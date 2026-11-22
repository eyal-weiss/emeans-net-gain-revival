"""Device-side pieces of the optimized GPU net-gain revival (FAISS PR #5915's PyTorch driver at 98deaca, and o-means).

ng_accumulate(rows, labels, distance, second, axes, centroids, sums)
    Adds, per centroid, the five sums the decision needs over the batch rows assigned to it:
      count, sum of margins m = max(d2 - d1, 0) (non-finite -> 0), sum of m^2, sum of p^2, sum of p^4,
    with p = (x - c_j) . v_j (c_j the centroids the batch was assigned against, v_j the split axis).
    `sums` is a (5, k) float64 tensor, updated in place. On CUDA: the per-row values are computed row-parallel in FP64,
    then segsum adds them per centroid over the label-sorted rows in a fixed order (no atomics: deterministic), so it
    can run inside the driver's captured CUDA graph and stays balanced when a few clusters hold most of a batch.
    Elsewhere: the same sums with index_add_ (deterministic on the CPU).

segsum(vals, labels, k, out, row_stride, col_stride)
    out[j * row_stride + c * col_stride] += sum of vals[r, c] over the rows r with labels[r] == j, in a fixed order.

nearest_other_all(centroids, exact=False)
    Each centroid's nearest other centroid and the squared distance to it (FP64 of the FP32 rows). On CUDA: FP32
    scores screen 8 candidates per centroid, compared in FP64 (ties to the lower index). exact=True uses the PR's
    rigorous Triton search instead (much slower when a few centroids are far from the rest, as on KDD Cup 99).
"""
import torch

try:
    import triton
    import triton.language as tl
except ImportError:                                   # CPU-only environments (tests of the fallback paths)
    triton = tl = None


if triton is not None:
    @triton.jit
    def _segsum_kernel(vals, order, offsets, out, ncol, row_stride, col_stride,
                       BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr):
        j = tl.program_id(0).to(tl.int64)
        cb = tl.program_id(1)
        start = tl.load(offsets + j).to(tl.int64)
        end = tl.load(offsets + j + 1).to(tl.int64)
        if start < end:
            cols = cb * BLOCK_C + tl.arange(0, BLOCK_C)
            cm = cols < ncol
            acc = tl.zeros((BLOCK_R, BLOCK_C), tl.float64)
            for b in range(start, end, BLOCK_R):
                lane = b + tl.arange(0, BLOCK_R)
                rm = lane < end
                r = tl.load(order + lane, mask=rm, other=0).to(tl.int64)
                acc += tl.load(vals + r[:, None] * ncol + cols[None, :], mask=rm[:, None] & cm[None, :], other=0.0)
            tot = tl.sum(acc, axis=0)
            dst = out + j * row_stride + cols * col_stride
            tl.store(dst, tl.load(dst, mask=cm, other=0.0) + tot, mask=cm)


if triton is not None:
    @triton.jit
    def _row_values_kernel(rows, idx, labels, distance, second, axes, C, out, pout, B, d, rs, sc,
                           HAS_IDX: tl.constexpr, BR: tl.constexpr, BD: tl.constexpr):
        """Per row r: out[r] = [1, m, m^2, p^2, p^4] (FP64), pout[r] = p, with y = x - c_lab, p = y . v_lab,
        m = max(second - distance, 0) (non-finite -> 0); x = rows[idx[r]] if HAS_IDX else rows[r]."""
        pid = tl.program_id(0)
        r = pid * BR + tl.arange(0, BR)
        rm = r < B
        lab = tl.load(labels + r, mask=rm, other=0).to(tl.int64)
        if HAS_IDX:
            src = tl.load(idx + r, mask=rm, other=0).to(tl.int64)
        else:
            src = r.to(tl.int64)
        gap = tl.load(second + r, mask=rm, other=0.0).to(tl.float64) - tl.load(distance + r, mask=rm, other=0.0).to(tl.float64)
        ok = rm & (gap == gap) & (gap < float("inf")) & (gap > float("-inf"))
        m = tl.where(ok, tl.maximum(gap, 0.0), 0.0)
        p = tl.zeros((BR,), tl.float64)
        for d0 in range(0, d, BD):
            dd = d0 + tl.arange(0, BD)
            mk = rm[:, None] & (dd < d)[None, :]
            x = tl.load(rows + src[:, None] * rs + dd[None, :], mask=mk, other=0.0).to(tl.float64)
            c = tl.load(C + lab[:, None] * sc + dd[None, :], mask=mk, other=0.0).to(tl.float64)
            a = tl.load(axes + lab[:, None] * d + dd[None, :], mask=mk, other=0.0).to(tl.float64)
            p += tl.sum((x - c) * a, axis=1)
        pp = p * p
        base = out + r.to(tl.int64) * 5
        tl.store(base, tl.where(rm, 1.0, 0.0).to(tl.float64), mask=rm)
        tl.store(base + 1, m, mask=rm)
        tl.store(base + 2, m * m, mask=rm)
        tl.store(base + 3, pp, mask=rm)
        tl.store(base + 4, pp * pp, mask=rm)
        tl.store(pout + r, p, mask=rm)

    @triton.jit
    def _segsum_py_kernel(rows, idx, order, offsets, p, C, out, d, rs, sc,
                          HAS_IDX: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr):
        """out[j, cols] += sum over the rows r of centroid j of p[r] (x_r - c_j)[cols] (FP64, fixed order)."""
        j = tl.program_id(0).to(tl.int64)
        cb = tl.program_id(1)
        start = tl.load(offsets + j).to(tl.int64)
        end = tl.load(offsets + j + 1).to(tl.int64)
        if start < end:
            cols = cb * BLOCK_C + tl.arange(0, BLOCK_C)
            cm = cols < d
            c = tl.load(C + j * sc + cols, mask=cm, other=0.0).to(tl.float64)
            acc = tl.zeros((BLOCK_R, BLOCK_C), tl.float64)
            for b in range(start, end, BLOCK_R):
                lane = b + tl.arange(0, BLOCK_R)
                rm = lane < end
                pos = tl.load(order + lane, mask=rm, other=0).to(tl.int64)
                if HAS_IDX:
                    src = tl.load(idx + pos, mask=rm, other=0).to(tl.int64)
                else:
                    src = pos
                x = tl.load(rows + src[:, None] * rs + cols[None, :], mask=rm[:, None] & cm[None, :],
                            other=0.0).to(tl.float64)
                pv = tl.load(p + pos, mask=rm, other=0.0)
                acc += tl.where(rm[:, None] & cm[None, :], pv[:, None] * (x - c[None, :]), 0.0)
            dst = out + j * d + cols
            tl.store(dst, tl.load(dst, mask=cm, other=0.0) + tl.sum(acc, axis=0), mask=cm)

    @triton.jit
    def _ins4(v, i, r0, r1, r2, r3, j0, j1, j2, j3):
        """Insert (v, i) into the ascending 4-list (r, j); strict <: an equal value goes after (lower index first)."""
        c0 = v < r0
        c1 = v < r1
        c2 = v < r2
        c3 = v < r3
        n3 = tl.where(c2, r2, tl.where(c3, v, r3))
        k3 = tl.where(c2, j2, tl.where(c3, i, j3))
        n2 = tl.where(c1, r1, tl.where(c2, v, r2))
        k2 = tl.where(c1, j1, tl.where(c2, i, j2))
        n1 = tl.where(c0, r0, tl.where(c1, v, r1))
        k1 = tl.where(c0, j0, tl.where(c1, i, j1))
        n0 = tl.where(c0, v, r0)
        k0 = tl.where(c0, i, j0)
        return n0, n1, n2, n3, k0, k1, k2, k3

    @triton.jit
    def _nn_top4_kernel(C, cn, out, sout, k, d, BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr):
        """For each centroid (row), the 4 other centroids with the smallest FP32 scores |c|^2 - 2 q.c (IEEE FP32
        products, no TF32), as int32 indices out[row, 0:4] (ascending score, ties to the lower index)."""
        pid = tl.program_id(0)
        rows = pid * BM + tl.arange(0, BM)
        rm = rows < k
        inf = float("inf")
        r0 = tl.full((BM,), inf, tl.float32)
        r1 = r0
        r2 = r0
        r3 = r0
        j0 = tl.zeros((BM,), tl.int32)
        j1 = j0
        j2 = j0
        j3 = j0
        lanes = tl.arange(0, BN)
        for n0 in range(0, k, BN):
            cols = n0 + lanes
            cmk = cols < k
            acc = tl.zeros((BM, BN), tl.float32)
            for d0 in range(0, d, BD):
                dd = d0 + tl.arange(0, BD)
                dm = dd < d
                q = tl.load(C + rows[:, None].to(tl.int64) * d + dd[None, :], mask=rm[:, None] & dm[None, :], other=0.0)
                c = tl.load(C + cols[:, None].to(tl.int64) * d + dd[None, :], mask=cmk[:, None] & dm[None, :], other=0.0)
                acc = tl.dot(q, tl.trans(c), acc=acc, input_precision="tf32x3")
            s = tl.load(cn + cols, mask=cmk, other=inf)[None, :] - 2.0 * acc
            s = tl.where(cmk[None, :] & (cols[None, :] != rows[:, None]) & (s == s), s, inf)
            # a tile can change a row's list only with a score below its 4th best (equal ones never enter: ties
            # keep the earlier, lower index); after the first tiles most tiles change nothing and are skipped
            if tl.sum((tl.min(s, axis=1) < r3).to(tl.int32), axis=0) > 0:
                for _t in tl.static_range(4):
                    v, jj = tl.min(s, axis=1, return_indices=True, return_indices_tie_break_left=True)
                    r0, r1, r2, r3, j0, j1, j2, j3 = _ins4(v, jj.to(tl.int32) + n0, r0, r1, r2, r3, j0, j1, j2, j3)
                    s = tl.where(lanes[None, :] == jj[:, None], inf, s)
        base = out + rows.to(tl.int64) * 4
        tl.store(base, j0, mask=rm)
        tl.store(base + 1, j1, mask=rm)
        tl.store(base + 2, j2, mask=rm)
        tl.store(base + 3, j3, mask=rm)
        tl.store(sout + rows.to(tl.int64) * 2, r0, mask=rm)
        tl.store(sout + rows.to(tl.int64) * 2 + 1, r3, mask=rm)

    @triton.jit
    def _nn_direct4_kernel(C, flags, out, sout, k, d, BM: tl.constexpr, BN: tl.constexpr):
        """For the flagged queries: the 4 smallest FP32 |q - c|^2 computed by direct differences (no cancellation:
        relative error at most (d + 2) 2^-24), with the best and 4th-best values; unflagged blocks return at once."""
        pid = tl.program_id(0)
        rows = pid * BM + tl.arange(0, BM)
        rm = rows < k
        fl = rm & (tl.load(flags + rows, mask=rm, other=0) != 0)
        if tl.sum(fl.to(tl.int32), axis=0) > 0:
            inf = float("inf")
            r0 = tl.full((BM,), inf, tl.float32)
            r1 = r0
            r2 = r0
            r3 = r0
            j0 = tl.zeros((BM,), tl.int32)
            j1 = j0
            j2 = j0
            j3 = j0
            lanes = tl.arange(0, BN)
            for n0 in range(0, k, BN):
                cols = n0 + lanes
                cmk = cols < k
                acc = tl.zeros((BM, BN), tl.float32)
                for dd in range(0, d):
                    q = tl.load(C + rows.to(tl.int64) * d + dd, mask=fl, other=0.0)
                    c = tl.load(C + cols.to(tl.int64) * d + dd, mask=cmk, other=0.0)
                    diff = q[:, None] - c[None, :]
                    acc += diff * diff
                acc = tl.where(cmk[None, :] & (cols[None, :] != rows[:, None]) & (acc == acc), acc, inf)
                if tl.sum((tl.min(acc, axis=1) < r3).to(tl.int32), axis=0) > 0:
                    for _t in tl.static_range(4):
                        v, jj = tl.min(acc, axis=1, return_indices=True, return_indices_tie_break_left=True)
                        r0, r1, r2, r3, j0, j1, j2, j3 = _ins4(v, jj.to(tl.int32) + n0, r0, r1, r2, r3, j0, j1, j2, j3)
                        acc = tl.where(lanes[None, :] == jj[:, None], inf, acc)
            base = out + rows.to(tl.int64) * 4
            tl.store(base, j0, mask=fl)
            tl.store(base + 1, j1, mask=fl)
            tl.store(base + 2, j2, mask=fl)
            tl.store(base + 3, j3, mask=fl)
            tl.store(sout + rows.to(tl.int64) * 2, r0, mask=fl)
            tl.store(sout + rows.to(tl.int64) * 2 + 1, r3, mask=fl)

    @triton.jit
    def _nn_exact_kernel(C, flags, nn, k, d, BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr):
        """For the flagged queries only: exact nearest other centroid by FP64 |q - c|^2 over all k (ties to the
        lower index); unflagged blocks return at once."""
        pid = tl.program_id(0)
        rows = pid * BM + tl.arange(0, BM)
        rm = rows < k
        fl = tl.load(flags + rows, mask=rm, other=0) != 0
        if tl.sum(fl.to(tl.int32), axis=0) > 0:
            best = tl.full((BM,), float("inf"), tl.float64)
            bi = tl.zeros((BM,), tl.int64)
            for n0 in range(0, k, BN):
                cols = n0 + tl.arange(0, BN)
                cmk = cols < k
                acc = tl.zeros((BM, BN), tl.float64)
                for dd in range(0, d):
                    q = tl.load(C + rows.to(tl.int64) * d + dd, mask=fl, other=0.0).to(tl.float64)
                    c = tl.load(C + cols.to(tl.int64) * d + dd, mask=cmk, other=0.0).to(tl.float64)
                    diff = q[:, None] - c[None, :]
                    acc += diff * diff
                acc = tl.where(cmk[None, :] & (cols[None, :] != rows[:, None]) & (acc == acc), acc, float("inf"))
                v, jj = tl.min(acc, axis=1, return_indices=True, return_indices_tie_break_left=True)
                better = v < best
                best = tl.where(better, v, best)
                bi = tl.where(better, jj.to(tl.int64) + n0, bi)
            tl.store(nn + rows, bi, mask=rm & fl)


def row_values_fused(rows, labels, distance, second, axes, C, idx=None, B=None):
    """((B, 5) FP64 per-row values, (B,) FP64 p) with one kernel (see _row_values_kernel)."""
    B = int(labels.shape[0]) if B is None else B
    d = axes.shape[1]
    out = torch.empty((B, 5), dtype=torch.float64, device=labels.device)
    p = torch.empty(B, dtype=torch.float64, device=labels.device)
    BR = 32
    with torch.cuda.device(labels.device):
        _row_values_kernel[(triton.cdiv(B, BR),)](rows, rows if idx is None else idx, labels, distance, second, axes, C,
                                                  out, p, B, d, rows.stride(0), C.stride(0), HAS_IDX=idx is not None,
                                                  BR=BR, BD=min(64, triton.next_power_of_2(d)), num_warps=4)
    return out, p


def _row_tile(B, k):
    """Rows per tile of the segmented sums: about the mean rows per centroid (4 to 64)."""
    return max(4, min(64, triton.next_power_of_2(max(1, B // max(k, 1)))))


def segsum_py(rows, idx, p, labels, k, C, out, index=None):
    """out[j] += sum over rows of centroid j of p (x - c_j), FP64 (k, d) (o-means' Oja statistic)."""
    d = out.shape[1]
    order, offsets = segment_index(labels, k) if index is None else index
    bc = min(128, triton.next_power_of_2(d))
    br = _row_tile(len(p), k)
    with torch.cuda.device(out.device):
        _segsum_py_kernel[(k, triton.cdiv(d, bc))](rows, rows if idx is None else idx, order, offsets, p, C, out, d,
                                                   rows.stride(0), C.stride(0), HAS_IDX=idx is not None, BLOCK_R=br,
                                                   BLOCK_C=bc, num_warps=4)
    return out


def segment_index(labels, k):
    """(order, offsets): rows sorted stably by label, and each label's segment [offsets[j], offsets[j + 1])."""
    dtype = torch.int16 if k < 2**15 else torch.int32                    # positions run to k inclusive
    sorted_labels, order = torch.sort(labels.to(dtype), stable=True)
    offsets = torch.searchsorted(sorted_labels, torch.arange(k + 1, device=labels.device, dtype=dtype))
    return order, offsets


def segsum(vals, labels, k, out, row_stride, col_stride, index=None):
    """Deterministic per-label sums of the rows of vals (B, ncol) FP64 into out (see the module docstring)."""
    assert vals.dtype == out.dtype == torch.float64 and vals.is_contiguous() and out.is_contiguous()
    B, ncol = vals.shape
    order, offsets = segment_index(labels, k) if index is None else index
    block_c = min(32, triton.next_power_of_2(ncol))
    with torch.cuda.device(vals.device):
        _segsum_kernel[(k, triton.cdiv(ncol, block_c))](vals, order, offsets, out, ncol, row_stride, col_stride,
                                                        BLOCK_R=_row_tile(B, k), BLOCK_C=block_c,
                                                        num_warps=1 if B // max(k, 1) < 32 else 4)
    return out


def row_values(rows, labels, distance, second, axes, centroids):
    """Per row: (lab, [1, m, m^2, p^2, p^4] FP64 (B, 5), y FP64 (B, d), p FP64 (B,))."""
    lab = labels.long()
    gap = second.double() - distance.double()
    m = torch.where(torch.isfinite(gap), gap.clamp_min(0.0), torch.zeros_like(gap))
    y = rows.double() - centroids[lab].double()
    p = (y * axes[lab].double()).sum(1)
    pp = p * p
    return lab, torch.stack((torch.ones_like(m), m, m * m, pp, pp * pp), dim=1).contiguous(), y, p


def ng_accumulate_torch(rows, labels, distance, second, axes, centroids, sums):
    """Reference / CPU path: the same sums with index_add_ (FP64)."""
    lab, v, _, _ = row_values(rows, labels, distance, second, axes, centroids)
    for c in range(5):
        sums[c].index_add_(0, lab, v[:, c])
    return sums


def ng_accumulate(rows, labels, distance, second, axes, centroids, sums):
    if triton is None or not rows.is_cuda:
        return ng_accumulate_torch(rows, labels, distance, second, axes, centroids, sums)
    assert sums.is_contiguous() and sums.dtype == torch.float64
    k = sums.shape[1]
    v, _p = row_values_fused(rows, labels, distance, second, axes, centroids)
    return segsum(v, labels, k, sums, 1, k)                               # sums is (5, k): column c at offset c * k


def nearest_other_all(centroids, exact=False):
    """(nearest other index int64 (k,), squared distance FP64 (k,)) for every centroid."""
    k, d = centroids.shape
    C = centroids.float().contiguous()
    if C.is_cuda and exact:
        from faiss.contrib.e_means.triton_assignment import nearest_other
        ids = torch.arange(k, device=C.device)
        nn = nearest_other(C, ids, torch.tensor(k, device=C.device))
    elif C.is_cuda and k <= 5:
        nn = _nearest_other_small(C)
    elif C.is_cuda:
        nn = _nearest_other_top4(C)
    else:
        nn = torch.empty(k, dtype=torch.int64, device=C.device)
        Cd = C.double()
        step = max(1, 2**24 // max(1, k * d))
        for a in range(0, k, step):
            q = Cd[a:a + step]
            dist = (q[:, None, :] - Cd[None, :, :]).square().sum(-1)
            dist[torch.arange(len(q)), torch.arange(a, a + len(q))] = float("inf")
            dist = torch.where(torch.isnan(dist), torch.full_like(dist, float("inf")), dist)
            nn[a:a + len(q)] = dist.argmin(dim=1)                          # first minimum: lowest index
    delta = (C[nn].double() - C.double()).square().sum(1)
    if k == 1:
        delta = torch.full_like(delta, float("inf"))            # no other centroid (the decision uses DMAX)
    return nn, delta


def _nearest_other_mm(C, screen=8, chunk=4096):
    """FP32 scores |c|^2 - 2 q.c screen the `screen` best candidates per query; FP64 |q - c|^2 picks among them
    (ties to the lower index)."""
    k = C.shape[0]
    cn = C.square().sum(1)
    nn = torch.empty(k, dtype=torch.int64, device=C.device)
    m = min(screen, k - 1)
    Cd = C.double()
    for a in range(0, k, chunk):
        q = C[a:a + chunk]
        s = torch.addmm(cn[None, :], q, C.t(), alpha=-2.0)
        r = torch.arange(len(q), device=C.device)
        s[r, a + r] = float("inf")
        s = torch.where(torch.isnan(s), torch.full_like(s, float("inf")), s)
        cand = torch.topk(s, m, dim=1, largest=False).indices                         # (chunk, m)
        exact = (Cd[cand] - Cd[a:a + chunk, None, :]).square().sum(-1)                   # (chunk, m) FP64
        exact = torch.where(torch.isnan(exact), torch.full_like(exact, float("inf")), exact)
        best = exact.min(dim=1, keepdim=True).values
        tie = torch.where(exact == best, cand, torch.full_like(cand, k))
        nn[a:a + chunk] = tie.min(dim=1).values
    return nn


def _nearest_other_small(C):
    """k <= 5: exact FP64 search (ties to the lower index)."""
    k = C.shape[0]
    Cd = C.double()
    D2 = (Cd[:, None, :] - Cd[None, :, :]).square().sum(-1)
    D2 = torch.where(torch.isnan(D2), torch.full_like(D2, float("inf")), D2)
    D2.fill_diagonal_(float("inf"))
    best = D2.min(dim=1, keepdim=True).values
    ids = torch.arange(k, device=C.device)
    return torch.where(D2 == best, ids[None, :], torch.full_like(D2, k, dtype=torch.int64)).min(dim=1).values


def _nearest_other_top4(C):
    """Fused screen (_nn_top4_kernel: 4 best FP32 scores per centroid, no k x k matrix) then FP64 |q - c|^2 among the 4
    (ties to the lower index)."""
    k, d = C.shape
    if k < 2:
        return torch.zeros(k, dtype=torch.int64, device=C.device)
    # distances are translation-invariant: centre on the coordinate-wise median (robust to far outliers, which
    # would drag a mean away from the bulk), so the FP32 scores lose less to the |c|^2 terms
    Cc = (C - C.median(dim=0, keepdim=True).values).contiguous()
    cn = Cc.square().sum(1)
    cand = torch.empty((k, 4), dtype=torch.int32, device=C.device)
    sc = torch.empty((k, 2), dtype=torch.float32, device=C.device)
    BM, BN = 64, 64
    with torch.cuda.device(C.device):
        _nn_top4_kernel[(triton.cdiv(k, BM),)](Cc, cn, cand, sc, k, d, BM=BM, BN=BN, BD=32, num_warps=4)
    # Rigour: a score s = |c|^2 - 2 q.c in FP32 (TF32x3 products, FP32 sums of d terms, FP32 norms) is off by at most
    # E(q) <= 2 [ (d + 4) 2^-23 smax + (d + 8) 2^-21 |q| cmax ] (smax = max |c|^2, cmax = max |c|; factor 2 for safety).
    # An unscreened centroid scores at least s4 in FP32, so it cannot be nearer than the best kept one unless
    # s4 - s1 <= 2 E(q); such queries (and any with a non-finite score) are searched exactly in FP64.
    if k > 5:
        cnorm = cn.double().sqrt()
        smax, cmax = cn.double().max(), cnorm.max()
        E = 2.0 * ((d + 4) * 2.0**-23 * smax + (d + 8) * 2.0**-21 * cnorm * cmax)
        s1, s4 = sc[:, 0].double(), sc[:, 1].double()
        flags = (~torch.isfinite(s1) | ~torch.isfinite(s4) | (s4 - s1 <= 2.0 * E)).to(torch.int8)
        # second tier for the flagged queries: FP32 direct differences (error relative to each distance, at most
        # (d + 2) 2^-24; doubled for safety): the 4 smallest, and the same gap test with this relative bound
        cand2 = cand.clone()
        sc2 = torch.zeros((k, 2), dtype=torch.float32, device=C.device)
        with torch.cuda.device(C.device):
            _nn_direct4_kernel[(triton.cdiv(k, 64),)](C, flags, cand2, sc2, k, d, BM=64, BN=64, num_warps=4)
        cand = torch.where(flags.bool()[:, None], cand2, cand)
        e = 2.0 * (d + 2) * 2.0**-24
        t1, t4 = sc2[:, 0].double(), sc2[:, 1].double()
        # The second tier bounds the result rigorously: the returned distance is within (1 + e) / (1 - e) of the true
        # minimum (about 5e-6 relative), since an unkept centroid's direct FP32 distance is at least t4 >= t1. Only
        # non-finite values (overflow) go to the exact FP64 scan.
        # Outside the relative-error model (overflow, or a direct minimum that is zero or below FP32's normal range,
        # where underflow breaks the relative bound) the query is searched exactly in FP64.
        tiny = 2.0**-100
        flags2 = (flags.bool() & (~torch.isfinite(t1) | ~torch.isfinite(t4) | (t1 < tiny))).to(torch.int8)
        exact_nn = torch.full((k,), -1, dtype=torch.int64, device=C.device)
        with torch.cuda.device(C.device):
            _nn_exact_kernel[(triton.cdiv(k, 64),)](C, flags2, exact_nn, k, d, BM=64, BN=64, BD=1, num_warps=4)
    cand = cand.long()
    cand = torch.where(cand == torch.arange(k, device=C.device)[:, None], (cand + 1) % k, cand)   # k < 5: padding
    Cd = C.double()
    exact = (Cd[cand] - Cd[:, None, :]).square().sum(-1)
    exact = torch.where(torch.isnan(exact), torch.full_like(exact, float("inf")), exact)
    best = exact.min(dim=1, keepdim=True).values
    tie = torch.where(exact == best, cand, torch.full_like(cand, k))
    nn = tie.min(dim=1).values
    if k > 5:
        nn = torch.where(exact_nn >= 0, exact_nn, nn)
    return nn


if triton is not None:
    @triton.jit
    def _ng_step_kernel(stale, cum_n, age, counts, cstride, ids, scal, decay_ptr, decay_val, k, n_ids,
                        DECAY_PTR: tl.constexpr, REPLACE: tl.constexpr, HAS_IDS: tl.constexpr,
                        LOGQ: tl.constexpr, BLOCK: tl.constexpr):
        """One step of the per-centroid state: stale *= g, cum_n += counts (by ids if given), age += 1, and in
        scal = [decay_prod, es_logq]: decay_prod *= g, es_logq += log g (g = decay, or 0 with replace_usage)."""
        pid = tl.program_id(0)
        if REPLACE:
            g = 0.0
        elif DECAY_PTR:
            g = tl.load(decay_ptr).to(tl.float64)
        else:
            g = decay_val
        i = pid * BLOCK + tl.arange(0, BLOCK)
        mk = i < k
        tl.store(stale + i, tl.load(stale + i, mask=mk, other=0.0) * g, mask=mk)
        tl.store(age + i, tl.load(age + i, mask=mk, other=0.0) + 1.0, mask=mk)
        if not HAS_IDS:
            # counts may be a strided column view (the driver's statistics table): index with its stride
            tl.store(cum_n + i, tl.load(cum_n + i, mask=mk, other=0.0)
                     + tl.load(counts + i.to(tl.int64) * cstride, mask=mk, other=0.0).to(tl.float64), mask=mk)
        if pid == 0:
            tl.store(scal, tl.load(scal) * g)
            if LOGQ:
                lg = tl.where(g > 0, tl.log(tl.maximum(g, 1e-300)), float("-inf"))
                tl.store(scal + 1, tl.load(scal + 1) + lg)


def ng_step(stale, cum_n, age, counts, ids, scal, decay, replace_usage, logq):
    """Device-only per-step update (CUDA-graph friendly); ids (sparse statistics) fall back to index_add_."""
    k = stale.shape[0]
    BLOCK = 1024
    is_t = torch.is_tensor(decay)
    with torch.cuda.device(stale.device):
        _ng_step_kernel[(triton.cdiv(k, BLOCK),)](stale, cum_n, age, counts, counts.stride(0),
                                                  cum_n if ids is None else ids, scal,
                                                  decay if is_t else stale, 0.0 if is_t else float(decay), k,
                                                  0 if ids is None else len(ids), DECAY_PTR=is_t,
                                                  REPLACE=bool(replace_usage), HAS_IDS=ids is not None, LOGQ=bool(logq),
                                                  BLOCK=BLOCK, num_warps=4)
    if ids is not None:
        cum_n.index_add_(0, ids, counts.double())

if triton is not None:
    @triton.jit
    def _ng_post_kernel(src, tgt, m_ptr, ng, ng_rs, cum, age, stale, moved_at, done_ptr, check_splits, scal,
                        stopped_ptr, es_step_ptr, step_ptr, cst, HAS_HO: tl.constexpr, ES: tl.constexpr,
                        BLOCK: tl.constexpr):
        """After a check's decision (src, tgt valid in [0, m); disjoint, distinct ids, so no write races): the
        statistics of both ends of every move restart (ng's 5 rows and the points and batches since birth to 0, the
        inherited weight to 1); (head handover) moved_at[t] = max(moved_at[t], done); program 0 adds m to the split
        counter and runs the early-stop clock: a check without moves once scal[1] (log prod(1 - alpha) since the last
        move) <= cst[0] (log 0.5, FP64) stops, and a check with moves restarts the clock."""
        pid = tl.program_id(0)
        m = tl.load(m_ptr)
        i = pid * BLOCK + tl.arange(0, BLOCK)
        live = i < m
        s = tl.load(src + i, mask=live, other=0)
        t = tl.load(tgt + i, mask=live, other=0)
        for r in tl.static_range(5):
            tl.store(ng + r * ng_rs + s, 0.0, mask=live)
            tl.store(ng + r * ng_rs + t, 0.0, mask=live)
        tl.store(cum + s, 0.0, mask=live)
        tl.store(cum + t, 0.0, mask=live)
        tl.store(age + s, 0.0, mask=live)
        tl.store(age + t, 0.0, mask=live)
        tl.store(stale + s, 1.0, mask=live)
        tl.store(stale + t, 1.0, mask=live)
        if HAS_HO:
            done = tl.load(done_ptr)
            old = tl.load(moved_at + t, mask=live, other=0)
            tl.store(moved_at + t, tl.maximum(old, done), mask=live)
        if pid == 0:
            tl.store(check_splits, tl.load(check_splits) + m)
            if ES:
                quiet = m == 0
                logq = tl.load(scal + 1)
                stopped = tl.load(stopped_ptr)
                stop = quiet & (logq <= tl.load(cst)) & (stopped == 0)
                tl.store(es_step_ptr, tl.where(stop, tl.load(step_ptr).to(tl.int64), tl.load(es_step_ptr)))
                tl.store(stopped_ptr, tl.where(stop, 1, stopped))
                tl.store(scal + 1, tl.where(quiet, logq, 0.0))


def ng_post(src, tgt, m, ng, cum, age, stale, moved_at, done, check_splits, scal, stopped, es_step, step, cst, es):
    """One launch for the state updates after a device decision (see _ng_post_kernel); ng is (5, k) row-major."""
    k = stale.shape[0]
    assert ng.stride(1) == 1 and all(t.is_contiguous() for t in (src, tgt, cum, age, stale))
    BLOCK = 256
    ho = moved_at is not None
    with torch.cuda.device(stale.device):
        _ng_post_kernel[(triton.cdiv(k, BLOCK),)](src, tgt, m, ng, ng.stride(0), cum, age, stale,
                                                 moved_at if ho else src, done if ho else m, check_splits, scal,
                                                 stopped, es_step, step, cst, HAS_HO=ho, ES=bool(es), BLOCK=BLOCK,
                                                 num_warps=4)

if triton is not None:
    @triton.jit
    def _nn_flags_kernel(cn, sc, flags, k, d, BLOCK: tl.constexpr):
        """The screen's rigour test (see _nearest_other_top4), one launch: smax = max |c|^2 (each program scans cn;
        a max is exact in any order), cmax = sqrt(smax) (= max |c|: sqrt is monotone and correctly rounded),
        E(q) = 2 ((d + 4) 2^-23 smax + (d + 8) 2^-21 |q| cmax) in FP64 with the torch path's operation order (launched
        without FMA contraction), flag = non-finite s1 or s4, or s4 - s1 <= 2 E(q)."""
        pid = tl.program_id(0)
        mx = tl.zeros((1024,), tl.float64)
        for b in range(0, k, 1024):
            j = b + tl.arange(0, 1024)
            mx = tl.maximum(mx, tl.load(cn + j, mask=j < k, other=0.0).to(tl.float64), propagate_nan=tl.PropagateNan.ALL)
        # a NaN norm makes smax NaN, as torch's max (review round 3: tl.maximum drops NaN by default)
        smax = tl.where(tl.max((mx != mx).to(tl.int32), axis=0) > 0, float("nan"), tl.max(mx, axis=0))
        cmax = tl.sqrt(smax)
        i = pid * BLOCK + tl.arange(0, BLOCK)
        mk = i < k
        cnorm = tl.sqrt(tl.load(cn + i, mask=mk, other=0.0).to(tl.float64))
        a1 = (d + 4) * 0.00000011920928955078125          # (d + 4) 2^-23, exact
        a2 = (d + 8) * 0.000000476837158203125            # (d + 8) 2^-21, exact
        E = 2.0 * (a1.to(tl.float64) * smax + (a2.to(tl.float64) * cnorm) * cmax)
        s1 = tl.load(sc + i * 2, mask=mk, other=0.0).to(tl.float64)
        s4 = tl.load(sc + i * 2 + 1, mask=mk, other=0.0).to(tl.float64)
        fin = (s1 == s1) & (tl.abs(s1) < float("inf")) & (s4 == s4) & (tl.abs(s4) < float("inf"))
        fl = ~fin | (s4 - s1 <= 2.0 * E)
        tl.store(flags + i, fl.to(tl.int8), mask=mk)

    @triton.jit
    def _nn_exact2_kernel(C, flags, sc2, nn, k, d, BM: tl.constexpr, BN: tl.constexpr):
        """Third tier, its flags computed here: a query flagged by the screen whose second-tier values are non-finite
        or whose direct minimum is below 2^-100 (outside the relative-error model) gets the exact nearest other
        centroid by FP64 |q - c|^2 over all k (ties to the lower index); every other query gets -1."""
        pid = tl.program_id(0)
        rows = pid * BM + tl.arange(0, BM)
        rm = rows < k
        f1 = tl.load(flags + rows, mask=rm, other=0) != 0
        t1 = tl.load(sc2 + rows * 2, mask=rm & f1, other=1.0).to(tl.float64)
        t4 = tl.load(sc2 + rows * 2 + 1, mask=rm & f1, other=1.0).to(tl.float64)
        fin = (t1 == t1) & (tl.abs(t1) < float("inf")) & (t4 == t4) & (tl.abs(t4) < float("inf"))
        fl = rm & f1 & (~fin | (t1 < 7.888609052210118e-31))          # 2^-100
        tl.store(nn + rows, tl.full((BM,), -1, tl.int64), mask=rm & ~fl)
        if tl.sum(fl.to(tl.int32), axis=0) > 0:
            best = tl.full((BM,), float("inf"), tl.float64)
            bi = tl.zeros((BM,), tl.int64)
            for n0 in range(0, k, BN):
                cols = n0 + tl.arange(0, BN)
                cmk = cols < k
                acc = tl.zeros((BM, BN), tl.float64)
                for dd in range(0, d):
                    q = tl.load(C + rows.to(tl.int64) * d + dd, mask=fl, other=0.0).to(tl.float64)
                    c = tl.load(C + cols.to(tl.int64) * d + dd, mask=cmk, other=0.0).to(tl.float64)
                    diff = q[:, None] - c[None, :]
                    acc += diff * diff
                acc = tl.where(cmk[None, :] & (cols[None, :] != rows[:, None]) & (acc == acc), acc, float("inf"))
                v, jj = tl.min(acc, axis=1, return_indices=True, return_indices_tie_break_left=True)
                better = v < best
                best = tl.where(better, v, best)
                bi = tl.where(better, jj.to(tl.int64) + n0, bi)
            tl.store(nn + rows, bi, mask=fl)

    @triton.jit
    def _nn_refine_kernel(C, cand, exact_nn, nn_out, delta_out, dup_out, k, d, BM: tl.constexpr,
                          BD: tl.constexpr):
        """Per query: FP64 |q - c|^2 to its 4 candidates (direct differences; NaN as inf; a candidate equal to the
        query itself is moved to the next index, as in the torch path), the nearest (ties to the lower index), or the
        third tier's answer when it has one; then the squared distance to it rounded to FP32 (as the rule's FP32
        distances) and whether the two centroids are equal coordinate by coordinate."""
        pid = tl.program_id(0)
        rows = pid * BM + tl.arange(0, BM)
        rm = rows < k
        r64 = rows.to(tl.int64)
        best = tl.full((BM,), float("inf"), tl.float64)
        bi = tl.full((BM,), k, tl.int64)
        for c4 in tl.static_range(4):
            j = tl.load(cand + r64 * 4 + c4, mask=rm, other=0).to(tl.int64)
            j = tl.where(j == r64, (j + 1) % k, j)
            acc = tl.zeros((BM,), tl.float64)
            for d0 in range(0, d, BD):
                dd = d0 + tl.arange(0, BD)
                dm = dd < d
                q = tl.load(C + r64[:, None] * d + dd[None, :], mask=rm[:, None] & dm[None, :], other=0.0)
                c = tl.load(C + j[:, None] * d + dd[None, :], mask=rm[:, None] & dm[None, :], other=0.0)
                df = c.to(tl.float64) - q.to(tl.float64)
                acc += tl.sum(df * df, axis=1)
            acc = tl.where(acc == acc, acc, float("inf"))
            take = (acc < best) | ((acc == best) & (j < bi))
            best = tl.where(take, acc, best)
            bi = tl.where(take, j, bi)
        ex = tl.load(exact_nn + rows, mask=rm, other=-1)
        nb = tl.where(ex >= 0, ex, bi)
        acc = tl.zeros((BM,), tl.float64)
        same = rm
        for d0 in range(0, d, BD):
            dd = d0 + tl.arange(0, BD)
            dm = dd < d
            q = tl.load(C + r64[:, None] * d + dd[None, :], mask=rm[:, None] & dm[None, :], other=0.0)
            c = tl.load(C + nb[:, None] * d + dd[None, :], mask=rm[:, None] & dm[None, :], other=0.0)
            df = c.to(tl.float64) - q.to(tl.float64)
            acc += tl.sum(df * df, axis=1)
            same = same & (tl.sum(((q != c) & dm[None, :]).to(tl.int32), axis=1) == 0)
        tl.store(nn_out + rows, nb, mask=rm)
        tl.store(delta_out + rows, acc.to(tl.float32).to(tl.float64), mask=rm)
        tl.store(dup_out + rows, same.to(tl.int8), mask=rm)


def nearest_other_state(centroids):
    """(nn int64, delta FP64 = FP32-rounded squared distance, dup int8) for the device decision, in a few launches:
    the same tiers as _nearest_other_top4 (screen, rigour test, second tier, third tier, FP64 refinement), with the
    torch glue between them fused. Falls back to nearest_other_all for k <= 5, CPU tensors or non-FP32 input."""
    k, d = centroids.shape
    if not (centroids.is_cuda and k > 5 and centroids.dtype == torch.float32):
        nn, delta = nearest_other_all(centroids)
        C = centroids
        return nn, delta.float().double(), (C == C[nn]).all(dim=1).to(torch.int8)
    C = centroids.contiguous()
    dev = C.device
    Cc = (C - C.median(dim=0, keepdim=True).values).contiguous()
    cn = Cc.square().sum(1)
    cand = torch.empty((k, 4), dtype=torch.int32, device=dev)
    sc = torch.empty((k, 2), dtype=torch.float32, device=dev)
    sc2 = torch.empty((k, 2), dtype=torch.float32, device=dev)
    flags = torch.empty(k, dtype=torch.int8, device=dev)
    exact_nn = torch.empty(k, dtype=torch.int64, device=dev)
    nn = torch.empty(k, dtype=torch.int64, device=dev)
    delta = torch.empty(k, dtype=torch.float64, device=dev)
    dup = torch.empty(k, dtype=torch.int8, device=dev)
    BM = 64
    with torch.cuda.device(dev):
        _nn_top4_kernel[(triton.cdiv(k, BM),)](Cc, cn, cand, sc, k, d, BM=BM, BN=64, BD=32, num_warps=4)
        _nn_flags_kernel[(triton.cdiv(k, 256),)](cn, sc, flags, k, d, BLOCK=256, num_warps=4, enable_fp_fusion=False)
        # the second tier writes the flagged queries' candidates over the screen's (cand2 = where(flags, ...))
        _nn_direct4_kernel[(triton.cdiv(k, 64),)](C, flags, cand, sc2, k, d, BM=64, BN=64, num_warps=4)
        _nn_exact2_kernel[(triton.cdiv(k, 64),)](C, flags, sc2, exact_nn, k, d, BM=64, BN=64, num_warps=4)
        _nn_refine_kernel[(triton.cdiv(k, 32),)](C, cand, exact_nn, nn, delta, dup, k, d, BM=32,
                                                 BD=min(64, triton.next_power_of_2(d)), num_warps=4)
    return nn, delta, dup
