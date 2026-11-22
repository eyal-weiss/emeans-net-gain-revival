"""Net-gain revival inside o-means (owlgebra-ai/faiss contrib/o_means at 9ecbf0c), optimized.

Differences from the proof of concept (research/omeans_netgain.py):
  * the margin d2 - d1 of every row comes from o-means' own two-level search: modified copies of its fine and reduce
    kernels keep the two best scores |c|^2 - 2 x.c among the probed centroids and the split children they check
    (|x|^2 cancels in the difference), instead of an exact search against all centroids;
  * per batch, the five per-centroid sums of the decision and the batch's Oja statistic sum p y for the split axis
    are computed row-parallel in FP64 and added per centroid by deterministic segmented sums (netgain_triton.segsum);
  * per check, every centroid's nearest other centroid is found on the device, the k-length state is copied to the
    host once, and the release's C++ decision (net_gain_revival.native.decide) chooses the moves; the split is the
    proof of concept's (o-means' split kernel for the bookkeeping, then the PR's axis split), without host reads.
Options: stat ("medg_pm" / "medg_fh"), freeze (True: no move once less than one pass remains), early_stop ("hl": end
the move phase after one EMA half-life without an accepted move; or None).
The second-nearest centroid is the best among the probed cells and the checked children, so a margin can exceed the
exact one when the true second-nearest lies in a cell the row does not probe.

The two kernels _fine2_kernel and _reduce2_kernel are modified copies of o-means' _fine_kernel and
_reduce_child_kernel (faiss/contrib/o_means/triton_two_level.py):
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the LICENSE file in the root directory of this source tree.
"""
import math
import time

import numpy as np
import torch
import triton
import triton.language as tl

import faiss.contrib.o_means.o_means as om_mod
from faiss.contrib.o_means.o_means import Omeans, _Training
from faiss.contrib.o_means.triton_kernels import accumulate, apply_splits, ema_update
from faiss.contrib.o_means.triton_two_level import (TwoLevelWorkspace, _launch_params, build_cells, permute_rows,
                                                    record_splits)
from faiss.contrib.e_means.base import _SPLIT_EPSILON, _epoch_seed
from net_gain_revival import native

import netgain_triton as NT

__all__ = ["OmeansNetGainFast"]


# ---------------------------------------------------------------------- two-level search with the second-best score
@triton.jit
def _fine2_kernel(x_ptr, idx_ptr, c_ptr, cn_ptr, off_ptr, cnt_ptr, bucket_ptr, best_ptr, sec_ptr, lab_ptr,
                  G, D, sx, sc, cap, sp,
                  HAS_IDX: tl.constexpr, P: tl.constexpr, GP: tl.constexpr,
                  BT: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr):
    t = tl.program_id(0)
    gid = tl.arange(0, GP)
    gm = gid < G
    cnt = tl.load(cnt_ptr + gid, mask=gm, other=0)
    nt = (cnt + BT - 1) // BT
    incl = tl.cumsum(nt, axis=0)
    c = tl.minimum(tl.sum((incl <= t).to(tl.int32), axis=0), G - 1)
    sel = gid == c
    excl_c = tl.sum(tl.where(sel, incl - nt, 0), axis=0)
    cnt_c = tl.sum(tl.where(sel, cnt, 0), axis=0)
    total = tl.sum(nt, axis=0)
    valid = t < total
    q = (t - excl_c) * BT + tl.arange(0, BT)
    qm = (q < cnt_c) & valid
    pair = tl.load(bucket_ptr + c.to(tl.int64) * cap + q, mask=qm, other=0)
    rl = pair // P
    s = pair - rl * P
    if HAS_IDX:
        rows = tl.load(idx_ptr + rl, mask=qm, other=0).to(tl.int64)
    else:
        rows = rl.to(tl.int64)
    xbase = x_ptr + rows[:, None] * sx
    k_lo = tl.load(off_ptr + c)
    k_hi = tl.load(off_ptr + c + 1)
    k_hi = tl.where(valid, k_hi, k_lo)
    best = tl.full((BT,), float("inf"), tl.float32)
    sec = tl.full((BT,), float("inf"), tl.float32)
    lab = tl.zeros((BT,), tl.int32)
    cols = tl.arange(0, BLOCK_K)
    for k0 in range(k_lo, k_hi, BLOCK_K):
        cid = k0 + cols
        cm = cid < k_hi
        cbase = c_ptr + cid[:, None].to(tl.int64) * sc
        acc = tl.zeros((BT, BLOCK_K), tl.float32)
        for d0 in range(0, D, BLOCK_D):
            col = d0 + tl.arange(0, BLOCK_D)
            xt = tl.load(xbase + col[None, :], mask=qm[:, None], other=0.0)
            ct = tl.load(cbase + col[None, :], mask=cm[:, None], other=0.0)
            acc = tl.dot(xt, tl.trans(ct), acc=acc, out_dtype=tl.float32)
        score = tl.load(cn_ptr + cid, mask=cm, other=float("inf"))[None, :] - 2.0 * acc
        bs, bl = tl.min(score, axis=1, return_indices=True, return_indices_tie_break_left=True)
        ss = tl.min(tl.where(cols[None, :] == bl[:, None], float("inf"), score), axis=1)
        # top two of {best, sec} and {bs, ss}
        sec = tl.minimum(tl.maximum(best, bs), tl.minimum(sec, ss))
        better = bs < best
        best = tl.where(better, bs, best)
        lab = tl.where(better, bl.to(tl.int32) + k0, lab)
    dst = s.to(tl.int64) * sp + rl
    tl.store(best_ptr + dst, best, mask=qm)
    tl.store(sec_ptr + dst, sec, mask=qm)
    tl.store(lab_ptr + dst, lab, mask=qm)


@triton.jit(do_not_specialize=["R"])
def _reduce2_kernel(best_ptr, sec_ptr, lab_ptr, out_ptr, obest_ptr, osec_ptr, child_ptr, x_ptr, idx_ptr, c_ptr,
                    cn_ptr, cnt_ptr, R, G, D, sx, sc, sp,
                    HAS_IDX: tl.constexpr, P: tl.constexpr, GP: tl.constexpr,
                    BLOCK: tl.constexpr, BLOCK_D: tl.constexpr):
    pid = tl.program_id(0)
    r = pid * BLOCK + tl.arange(0, BLOCK)
    m = r < R
    b = tl.load(best_ptr + r, mask=m, other=float("inf"))
    s = tl.load(sec_ptr + r, mask=m, other=float("inf"))
    lab = tl.load(lab_ptr + r, mask=m, other=0)
    for p in tl.static_range(1, P):
        bp = tl.load(best_ptr + p * sp + r, mask=m, other=float("inf"))
        sq = tl.load(sec_ptr + p * sp + r, mask=m, other=float("inf"))
        lp = tl.load(lab_ptr + p * sp + r, mask=m, other=0)
        s = tl.minimum(tl.maximum(b, bp), tl.minimum(s, sq))
        better = bp < b                    # strict: the lower probe rank wins ties
        b = tl.where(better, bp, b)
        lab = tl.where(better, lp, lab)
    c1 = tl.load(child_ptr + lab, mask=m, other=-1)
    c1 = tl.where(c1 < 0, lab, c1)
    c2 = tl.load(child_ptr + c1, mask=m, other=-1)
    c2 = tl.where(c2 < 0, c1, c2)
    n1 = m & (c1 != lab)
    n2 = m & (c2 != c1)
    # the second best must be another centroid: a cycle of split pointers can lead c2 back to the winner
    n2s = n2 & (c2 != lab)
    need = n1 | n2
    if tl.sum(need.to(tl.int32), axis=0) > 0:
        if HAS_IDX:
            rows = tl.load(idx_ptr + r, mask=need, other=0).to(tl.int64)
        else:
            rows = r.to(tl.int64)
        a1 = tl.zeros((BLOCK,), tl.float32)
        a2 = tl.zeros((BLOCK,), tl.float32)
        for d0 in range(0, D, BLOCK_D):
            col = d0 + tl.arange(0, BLOCK_D)
            xv = tl.load(x_ptr + rows[:, None] * sx + col[None, :], mask=need[:, None],
                         other=0.0).to(tl.float32)
            v1 = tl.load(c_ptr + c1[:, None].to(tl.int64) * sc + col[None, :],
                         mask=n1[:, None], other=0.0).to(tl.float32)
            v2 = tl.load(c_ptr + c2[:, None].to(tl.int64) * sc + col[None, :],
                         mask=n2[:, None], other=0.0).to(tl.float32)
            a1 += tl.sum(xv * v1, axis=1)
            a2 += tl.sum(xv * v2, axis=1)
        s1 = tl.load(cn_ptr + c1, mask=n1, other=float("inf")) - 2.0 * a1
        s2 = tl.load(cn_ptr + c2, mask=n2, other=float("inf")) - 2.0 * a2
        s = tl.where(n1, tl.minimum(tl.maximum(b, s1), s), s)
        t1 = n1 & (s1 < b)
        b = tl.where(t1, s1, b)
        lab = tl.where(t1, c1, lab)
        s = tl.where(n2s, tl.minimum(tl.maximum(b, s2), s), s)
        t2 = n2 & (s2 < b)
        b = tl.where(t2, s2, b)
        lab = tl.where(t2, c2, lab)
    tl.store(out_ptr + r, lab, mask=m)
    tl.store(obest_ptr + r, b, mask=m)
    tl.store(osec_ptr + r, s, mask=m)
    if pid == 0:
        gid = tl.arange(0, GP)
        tl.store(cnt_ptr + gid, tl.zeros((GP,), tl.int32), mask=gid < G)


class TwoLevelWorkspace2(TwoLevelWorkspace):
    def __init__(self, rows, G, max_p, device):
        super().__init__(rows, G, max_p, device)
        self.sec = torch.empty((max_p, rows), dtype=torch.float32, device=device)
        self.out_best = torch.empty(rows, dtype=torch.float32, device=device)
        self.out_sec = torch.empty(rows, dtype=torch.float32, device=device)


def two_level_assign2(x16, c16, cn, cells, ws, p, idx=None, rows=None):
    """o-means' two_level_assign, also returning each row's best and second-best score |c|^2 - 2 x.c."""
    R = int(idx.shape[0]) if idx is not None else int(rows if rows is not None else x16.shape[0])
    dp = x16.shape[1]
    G, GP = cells.G, cells.GP
    assert R <= ws.rows and p <= ws.max_p and G == ws.G and 1 <= p <= G
    assert x16.stride(1) == 1 and c16.stride(1) == 1 and dp % 32 == 0
    if R == 0:
        return ws.labels[:0], ws.out_best[:0], ws.out_sec[:0]
    xi = idx if idx is not None else x16
    coarse, coarse_rows, coarse_warps, block_k = _launch_params(GP, dp)
    coarse[(triton.cdiv(R, coarse_rows),)](
        x16, xi, cells.g16, cells.gn, ws.cnt, ws.bucket, R, dp, x16.stride(0),
        cells.g16.stride(0), ws.rows,
        HAS_IDX=idx is not None, P=p, GP=GP, BLOCK_B=coarse_rows, BLOCK_D=32,
        num_warps=coarse_warps)
    block_t = 64
    _fine2_kernel[(triton.cdiv(R * p, block_t) + G,)](
        x16, xi, c16, cn, cells.off, ws.cnt, ws.bucket, ws.best, ws.sec, ws.lab,
        G, dp, x16.stride(0), c16.stride(0), ws.rows, ws.rows,
        HAS_IDX=idx is not None, P=p, GP=GP, BT=block_t, BLOCK_K=block_k,
        BLOCK_D=32, num_warps=4, num_stages=2)
    _reduce2_kernel[(triton.cdiv(R, 128),)](
        ws.best, ws.sec, ws.lab, ws.labels, ws.out_best, ws.out_sec, cells.child, x16, xi, c16, cn, ws.cnt,
        R, G, dp, x16.stride(0), c16.stride(0), ws.rows,
        HAS_IDX=idx is not None, P=p, GP=GP, BLOCK=128, BLOCK_D=32, num_warps=4)
    return ws.labels[:R], ws.out_best[:R], ws.out_sec[:R]


# ---------------------------------------------------------------------- per-centroid sums and Oja statistic
def om_stats(src, idx, R, labels, best, sec, axes, C, sums, bU, k, d):
    """Adds to sums (kp, 5) FP64 [rows, sum m, sum m^2, sum p^2, sum p^4] and to bU (kp, d) FP64 sum p y, per centroid
    (table position), with y = x - c_j, p = y . v_j, m = s2 - s1 (the two best scores of the two-level search):
    one fused kernel for the per-row values, then deterministic segmented sums (netgain_triton)."""
    lab = labels[:R]
    v, p = NT.row_values_fused(src, lab, best, sec, axes, C, idx=idx, B=R)
    index = NT.segment_index(lab, k)
    NT.segsum(v, lab, k, sums, 5, 1, index=index)
    NT.segsum_py(src, idx, p, lab, k, C, bU, index=index)
    return index[1]


def _replay(cache, key, fn):
    """Run fn: eagerly the first time (compiles the kernels), then captured once as a CUDA graph and replayed
    (eagerly for good if the capture fails)."""
    g = cache.get(key)
    if g is None:
        fn()
        dev = torch.cuda.current_device()
        current, stream = torch.cuda.current_stream(dev), torch.cuda.Stream(dev)
        graph, pool = torch.cuda.CUDAGraph(), torch.cuda.MemPool()
        stream.wait_stream(current)
        try:
            with torch.cuda.stream(stream):
                graph.capture_begin(pool=pool.id, capture_error_mode="thread_local")
                try:
                    fn()
                finally:
                    graph.capture_end()
            cache[key] = (graph, pool)
        except RuntimeError as e:
            cache[key] = False
            cache.setdefault("_errors", {})[str(key)] = str(e)[:200]
        current.wait_stream(stream)
    elif g is False:
        fn()
    else:
        g[0].replay()


@triton.jit
def _om_step_kernel(U, bU, axes, stale, cum_n, age, offsets, counts, es_logq, a_ptr, k, d,
                    TRACK: tl.constexpr, ES: tl.constexpr, BLOCK_D: tl.constexpr):
    """Centroid j after o-means' update: with TRACK, U = (1 - a) U + a bU, axes = U / |U| where |U| > 0, bU = 0;
    stale *= 1 - a; cum_n += rows of j in this batch (segment offsets with TRACK, else counts); age += 1; program 0
    adds log(1 - a) to the device early-stop clock (ES)."""
    j = tl.program_id(0).to(tl.int64)
    a = tl.load(a_ptr)
    lq = tl.load(a_ptr + 1)
    col = tl.arange(0, BLOCK_D)
    cm = col < d
    if TRACK:
        u = tl.load(U + j * d + col, mask=cm, other=0.0) * (1.0 - a) + a * tl.load(bU + j * d + col, mask=cm, other=0.0)
        tl.store(U + j * d + col, u, mask=cm)
        nrm = tl.sqrt(tl.sum(u * u, axis=0))
        if nrm > 0:
            tl.store(axes + j * d + col, (u / tl.maximum(nrm, 1e-300)).to(tl.float32), mask=cm)
        tl.store(bU + j * d + col, tl.zeros((BLOCK_D,), tl.float64), mask=cm)
        n = (tl.load(offsets + j + 1) - tl.load(offsets + j)).to(tl.float64)
    else:
        n = tl.load(counts + j).to(tl.float64)
    tl.store(stale + j, tl.load(stale + j) * (1.0 - a))
    tl.store(cum_n + j, tl.load(cum_n + j) + n)
    tl.store(age + j, tl.load(age + j) + 1.0)
    if ES:
        if j == 0:
            tl.store(es_logq, tl.load(es_logq) + lq)


@triton.jit
def _om_post_split_kernel(src, tgt, m_ptr, spread, lo, hi, C, c16, cn, U, axes, ng, cum_n, age, stale, d, sc, s16,
                          BLOCK_D: tl.constexpr):
    """Pair i < m (after o-means' split kernel): the PR's axis split C[s], C[t] = lo_i, hi_i (where spread_i > 0) with
    the FP16 copies and norms; U[s] /= 2, U[t] = U[s], axes[t] = axes[s]; statistics of both ends restart."""
    i = tl.program_id(0)
    if i < tl.load(m_ptr):
        s = tl.load(src + i).to(tl.int64)
        t = tl.load(tgt + i).to(tl.int64)
        col = tl.arange(0, BLOCK_D)
        cm = col < d
        a = tl.load(lo + i * d + col, mask=cm, other=0.0)
        b = tl.load(hi + i * d + col, mask=cm, other=0.0)
        a16 = a.to(tl.float16)
        b16 = b.to(tl.float16)
        apart = tl.sum(tl.where(cm & (a16 != b16), 1, 0), axis=0) > 0   # distinct in the FP16 assignment copies
        if (tl.load(spread + i) > 0) & apart:
            tl.store(C + s * sc + col, a, mask=cm)
            tl.store(C + t * sc + col, b, mask=cm)
            tl.store(c16 + s * s16 + col, a16, mask=cm)
            tl.store(c16 + t * s16 + col, b16, mask=cm)
            af = a16.to(tl.float32)
            bf = b16.to(tl.float32)
            tl.store(cn + s, tl.sum(af * af, axis=0))
            tl.store(cn + t, tl.sum(bf * bf, axis=0))
        u = tl.load(U + s * d + col, mask=cm, other=0.0) / 2
        tl.store(U + s * d + col, u, mask=cm)
        tl.store(U + t * d + col, u, mask=cm)
        tl.store(axes + t * d + col, tl.load(axes + s * d + col, mask=cm, other=0.0), mask=cm)
        for r in tl.static_range(2):
            e = s if r == 0 else t
            five = tl.arange(0, 8)
            tl.store(ng + e * 5 + five, tl.zeros((8,), tl.float64), mask=five < 5)
            tl.store(cum_n + e, 0.0)
            tl.store(age + e, 0.0)
            tl.store(stale + e, 1.0)


# ---------------------------------------------------------------------- the run
class _FastTraining(_Training):
    stat, freeze, early_stop, check = "medg_pm", True, "hl", "device"

    def __init__(self, x, init, sched, seed):
        super().__init__(x, init, sched, seed)
        kp, d, dev = self.C.shape[0], sched.d, x.device
        self.device_check = self.check == "device" and self.stat == "medg_pm"
        if self.device_check or self.check == "both":
            import netgain_device as ND
            self.decider = ND.DeviceDecider(sched.k, dev)
            self.alpha_dev = torch.zeros((), dtype=torch.float64, device=dev)
            self.step_dev = torch.zeros((), dtype=torch.float64, device=dev)
            self.dprod_dev = torch.zeros((), dtype=torch.float64, device=dev)
            self.es_dev = torch.zeros((), dtype=torch.int32, device=dev)
            self.es_step_dev = torch.full((), -1, dtype=torch.int64, device=dev)
            self.es_logq_dev = torch.zeros((), dtype=torch.float64, device=dev)
            self.splits_dev = torch.zeros((), dtype=torch.int64, device=dev)
            self.ncand_dev = torch.zeros((), dtype=torch.int64, device=dev)
        self.es_host = torch.zeros((), dtype=torch.int32).pin_memory() if self.device_check else None
        self.es_event = None
        self.check_agree = self.check_total = 0
        # static buffers of the captured per-step graphs
        k = sched.k
        self.a_dev = torch.zeros(2, dtype=torch.float64, device=dev)                 # [alpha, log(1 - alpha)]
        self.idx_buf = torch.zeros(sched.batch_size, dtype=torch.int64, device=dev)
        self.off_buf = torch.zeros(k + 1, dtype=torch.int64, device=dev)   # row offsets (up to the batch size)
        self.cnt_buf = torch.zeros(k, dtype=torch.int64, device=dev)
        self.graphs = {}
        self.use_graphs = self.device_check
        self.ws = TwoLevelWorkspace2(sched.batch_size, sched.nlist, sched.nprobe, dev)
        g = torch.Generator(device=dev)
        g.manual_seed(seed * 31 + 7)
        axes = torch.randn(kp, d, dtype=torch.float32, device=dev, generator=g)
        self.axes = (axes / axes.norm(dim=1, keepdim=True)).contiguous()
        self.U = torch.zeros((kp, d), dtype=torch.float64, device=dev)
        self.bU = torch.zeros((kp, d), dtype=torch.float64, device=dev)
        self.ng = torch.zeros((kp, 5), dtype=torch.float64, device=dev)
        z = lambda: torch.zeros(kp, dtype=torch.float64, device=dev)
        self.cum_n, self.age, self.stale = z(), z(), z()
        self.decay_prod = 1.0
        self.es_logq, self.es_stopped, self.es_stop_step = 0.0, False, None
        self.prev_delta = None
        self.n_splits = self.n_checks = self.n_candidates = 0
        self.rule_seconds = 0.0

    def per_centroid(self):
        return [self.C, self.c16, self.cn, self.usage, self.close_sums, self.close_cnt, self.axes, self.U, self.ng,
                self.cum_n, self.age, self.stale]

    def rebuild_cells(self, first_pass):
        if self.cells is None:
            iters = om_mod._FIRST_BUILD_ITERS
        else:
            iters = om_mod._FIRST_PASS_REBUILD_ITERS if first_pass else om_mod._REBUILD_ITERS
        order, self.cells = build_cells(self.C, self.sched.k, self.sched.nlist, iters,
                                        prev=self.cells, generator=self.cell_rng)
        permute_rows(order, self.sched.k, self.per_centroid())
        self.ids = self.ids[order]
        if self.prev_delta is not None:
            self.prev_delta = self.prev_delta[order.cpu().numpy()]

    def after_update(self, alpha, track, labels, R, offsets=None):
        """Host bookkeeping, then the per-centroid update (_om_step_kernel; reads alpha from self.a_dev)."""
        a = float(alpha)
        self.decay_prod *= 1.0 - a
        lq = math.log(1.0 - a) if a < 1.0 else -math.inf
        self.es_logq += lq
        self._after_kernel(track, labels, R, offsets)

    def _after_kernel(self, track, labels, R, offsets):
        k, d = self.sched.k, self.sched.d
        es = self.device_check and self.early_stop == "hl"
        if not track:
            self.cnt_buf.zero_()
            self.cnt_buf.index_add_(0, labels[:R].long(), torch.ones(R, dtype=torch.int64, device=labels.device))
        _om_step_kernel[(k,)](self.U, self.bU, self.axes, self.stale, self.cum_n, self.age,
                              offsets if track else self.cnt_buf, self.cnt_buf,
                              self.es_logq_dev if es else self.cum_n, self.a_dev, k, d, TRACK=bool(track), ES=es,
                              BLOCK_D=triton.next_power_of_2(d))

    def _apply_moves(self, s, t, m, P, record=True):
        """Apply pairs i < m of (s, t) (device m, P >= m slots): o-means' split kernel for the bookkeeping, then
        the PR's axis split, U halved and copied, statistics restarted (_om_post_split_kernel)."""
        d = self.sched.d
        s, t = s[:P], t[:P]
        norm = self.U[s].norm(dim=1)
        mass = self.usage[s].double()
        valid = (norm > 0) & (mass > 0) & torch.isfinite(norm) & torch.isfinite(mass)
        spread = torch.sqrt(2.0 * torch.where(valid, norm / torch.where(valid, mass, 1.0), 0.0) / math.pi)
        c = self.C[s, :d].double()
        off = spread[:, None] * self.axes[s].double()
        lo, hi = (c - off).float().contiguous(), (c + off).float().contiguous()
        recency = self.sched.recency_close
        apply_splits(self.C, self.c16, self.cn, self.usage, s, t, m, d, _SPLIT_EPSILON, max_pairs=P,
                     wsum=self.close_sums if recency else None, wcnt=self.close_cnt if recency else None)
        _om_post_split_kernel[(P,)](s, t, m, spread.contiguous(), lo, hi, self.C, self.c16, self.cn, self.U, self.axes,
                                    self.ng, self.cum_n, self.age, self.stale, d, self.C.stride(0),
                                    self.c16.stride(0), BLOCK_D=triton.next_power_of_2(d))
        if record:
            record_splits(self.cells.child, s, t, m, P)

    def _device_decision(self, step_i, alpha):
        k, d = self.sched.k, self.sched.d
        Ck = self.C[:k, :d]
        # FP32-rounded distances (as the host path's decision gets) and the duplicate flags: netgain_triton's fused
        # search
        nn, delta, dup = NT.nearest_other_state(Ck)
        raw = (2.0 / math.pi) * torch.linalg.vector_norm(self.U[:k].float(), dim=1, dtype=torch.float64)
        if alpha is not None:
            self.alpha_dev.fill_(float(alpha))
            self.step_dev.fill_(float(step_i))
            self.dprod_dev.fill_(self.decay_prod)
        ng = self.ng[:k]
        st = dict(usage=self.usage[:k].double(), raw=raw, stale=self.stale[:k], age=self.age[:k], cum=self.cum_n[:k],
                  m_n=ng[:, 0], m_sum=ng[:, 1], m_sq=ng[:, 2], p2=ng[:, 3], p4=ng[:, 4], nn=nn, delta=delta, dup=dup,
                  decay_prod=self.dprod_dev, alpha=self.alpha_dev, step=self.step_dev, stopped=self.es_dev)
        return self.decider.decide(st)

    def revive(self, step_i, alpha):
        if self.device_check:
            return self.revive_device(step_i, alpha)
        k, d, dev = self.sched.k, self.sched.d, self.C.device
        dev_moves = None
        if self.check == "both":
            s_d, t_d, m_d, _nc = self._device_decision(step_i, alpha)
            md = int(m_d)
            dev_moves = (s_d[:md].cpu().numpy(), t_d[:md].cpu().numpy())
        Ck = self.C[:k, :d].contiguous()
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        nn, delta = NT.nearest_other_all(Ck)
        packed = torch.cat((self.usage[:k].double()[None], self.stale[:k][None], self.age[:k][None],
                            self.cum_n[:k][None], self.ng[:k].T, nn.double()[None], delta[None])).cpu().numpy()
        usage, stale, age, cum = (np.ascontiguousarray(packed[i]) for i in range(4))
        m_n, m_sum, m_sq, p2, p4 = (np.ascontiguousarray(packed[4 + i]) for i in range(5))
        D = np.zeros((k, 2), dtype=np.float32)
        D[:, 1] = packed[10]
        I = np.ascontiguousarray(np.stack((np.arange(k, dtype=np.int64), packed[9].astype(np.int64)), axis=1))
        rs = np.ascontiguousarray(self.U[:k].float().cpu().numpy())
        readout = np.ascontiguousarray(Ck.cpu().numpy())
        a = min(max(float(alpha), 1e-6), 1.0)
        src, tgt, self.prev_delta, info = native.decide(usage, rs, 2.0 / np.pi, stale, age, cum, self.decay_prod, a,
                                                        step_i, m_n, m_sum, m_sq, p2, p4, D, I, self.stat,
                                                        self.prev_delta, readout)
        self.n_checks += 1
        self.n_candidates += info["n_candidates"]
        if dev_moves is not None:
            hs = np.zeros(0, np.int64) if src is None else src
            ht = np.zeros(0, np.int64) if tgt is None else tgt
            self.check_total += 1
            self.check_agree += int(np.array_equal(hs, dev_moves[0]) and np.array_equal(ht, dev_moves[1]))
        if self.early_stop == "hl":
            if src is not None:
                self.es_logq = 0.0
            elif self.es_logq <= math.log(0.5):
                self.es_stopped = True
                self.es_stop_step = step_i
        self.rule_seconds += time.perf_counter() - t0
        if src is None:
            return
        npair = len(src)
        self._apply_moves(torch.from_numpy(src).to(dev), torch.from_numpy(tgt).to(dev),
                          torch.tensor(npair, device=dev), npair)
        self.n_splits += npair

    def revive_device(self, step_i, alpha):
        """The whole check on the device, captured as a CUDA graph after its first run (o-means' split pointers are
        recorded eagerly after it: the cell structure is rebuilt between checks)."""
        k = self.sched.k
        P = k // 2 + 1
        self.alpha_dev.fill_(float(alpha))
        self.step_dev.fill_(float(step_i))
        self.dprod_dev.fill_(self.decay_prod)
        _replay(self.graphs, ("check",), self._check_body) if self.use_graphs else self._check_body()
        record_splits(self.cells.child, self.decider.src[:P], self.decider.tgt[:P], self.decider.m, P)
        self.n_checks += 1
        if self.early_stop == "hl":
            self.es_host.copy_(self.es_dev, non_blocking=True)
            if self.es_event is None:
                self.es_event = torch.cuda.Event()
            self.es_event.record()

    def _check_body(self):
        k = self.sched.k
        s, t, m, nc = self._device_decision(None, None)
        self._apply_moves(s, t, m, k // 2 + 1, record=False)
        self.splits_dev.add_(m)
        self.ncand_dev.add_(nc)
        if self.early_stop == "hl":
            quiet = m == 0
            stop = quiet & (self.es_logq_dev <= math.log(0.5)) & (self.es_dev == 0)
            self.es_step_dev.copy_(torch.where(stop, self.step_dev.long(), self.es_step_dev))
            self.es_dev.copy_(torch.where(stop, 1, self.es_dev))
            self.es_logq_dev.copy_(torch.where(quiet, self.es_logq_dev, 0.0))

    def finish_device(self):
        if self.device_check:
            self.n_splits = int(self.splits_dev)
            self.n_candidates = int(self.ncand_dev)
            step = int(self.es_step_dev)
            self.es_stop_step = None if step < 0 else step

    def train(self, labels_out=None):
        # o-means' loop (o_means.py, _Training.train) plus the lines marked NG
        sched = self.sched
        n, k, d = sched.n, sched.k, sched.d
        last_epoch = len(sched.epoch_rows) - 1
        total_rows = sum(sched.epoch_rows)
        perm, epoch = None, -1
        rows_done, since_rebuild = 0, 0
        for step_i, step in enumerate(sched.steps):
            if step.epoch != epoch:
                epoch = step.epoch
                generator = torch.Generator(device=self.x.device)
                generator.manual_seed(_epoch_seed(self.seed, epoch))
                perm = torch.randperm(n, generator=generator, device=self.x.device)
            src, idx, R = self.batch_rows(perm, step.b0, step.b1)
            first_pass = rows_done < n
            every = om_mod._CHECK_UNUSED_EVERY if first_pass else sched.rebuild_every
            if self.cells is None or since_rebuild >= every:
                self.rebuild_cells(first_pass)
                since_rebuild = 0
            since_rebuild += 1
            labels, best, sec = two_level_assign2(src, self.c16, self.cn, self.cells, self.ws,     # NG
                                                  sched.nprobe, idx=idx, rows=R)
            if labels_out is not None and epoch == last_epoch:
                labels_out[perm[step.b0:step.b1]] = self.ids[labels]
            rows_done += R
            # NG: the freeze (no move once the remaining rows are less than one pass), or the early stop
            if self.es_event is not None and not self.es_stopped and self.es_event.query():
                self.es_stopped = bool(self.es_host.item())                                        # NG: lagged flag
            frozen = self.es_stopped or (self.freeze and len(sched.epoch_rows) > 1 and rows_done >= n
                                         and total_rows - rows_done < n)
            a_now = float(step.alpha)
            self.a_dev[0].fill_(a_now)                                                                # NG
            self.a_dev[1].fill_(math.log(1.0 - a_now) if a_now < 1.0 else -math.inf)
            offsets = None
            if self.use_graphs:                                                                       # NG: captured
                if idx is not None:
                    self.idx_buf[:R].copy_(idx)
                ib = None if idx is None else self.idx_buf[:R]
                if not frozen:
                    _replay(self.graphs, ("stats", R, idx is None), lambda: self.off_buf.copy_(
                        om_stats(src, ib, R, labels, best, sec, self.axes, self.C, self.ng, self.bU, k, d)))
                    offsets = self.off_buf
            elif not frozen:                                                                          # NG
                offsets = om_stats(src, idx, R, labels, best, sec, self.axes, self.C, self.ng, self.bU, k, d)
            accumulate(src, labels, self.sums, self.cnt, d, idx=idx)
            close = step.omega is not None
            ema_update(self.C, self.c16, self.cn, self.usage, self.sums, self.cnt, k, d,
                       step.alpha, self.close_sums if close else None,
                       self.close_cnt if close else None,
                       omega=step.omega if close else 1.0)
            if self.use_graphs:                                                                       # NG
                a = float(step.alpha)
                self.decay_prod *= 1.0 - a
                self.es_logq += math.log(1.0 - a) if a < 1.0 else -math.inf
                _replay(self.graphs, ("after", R, not frozen),
                        lambda tr=not frozen: self._after_kernel(tr, labels, R, offsets))
            else:
                self.after_update(step.alpha, not frozen, labels, R, offsets)                         # NG
            if step.check and not frozen:                                                             # NG
                self.revive(step_i, step.alpha)
        self.finish_device()
        return self.closing_mean()


class OmeansNetGainFast(Omeans):
    """o-means with the net-gain revival rule, optimized (see the module docstring)."""

    def __init__(self, d, k, nepochs=5, stat="medg_pm", freeze=True, early_stop="hl", check="device", **kw):
        assert stat in ("medg_pm", "medg_fh") and early_stop in (None, "hl") and check in ("device", "host", "both")
        super().__init__(d, k, nepochs=nepochs, **kw)
        if not native.available():
            raise RuntimeError("net_gain_revival/native/libnetgain.so is required (native/build.sh)")
        self.opts = dict(stat=stat, freeze=bool(freeze), early_stop=early_stop, check=check)
        self.revival_stats = None
        self.n_splits = self.n_checks = 0
        self.rule_seconds = 0.0
        self.es_stop_step = None

    def train(self, x, init_centroids=None):
        cls = type("_RuleTraining", (_FastTraining,), dict(self.opts))
        holder = {}
        init = cls.__init__

        def capture(run, *a, **kw):
            init(run, *a, **kw)
            holder["run"] = run
        cls.__init__ = capture
        original = om_mod._Training
        om_mod._Training = cls          # Omeans.train builds its run from the module's _Training
        try:
            out = super().train(x, init_centroids)
        finally:
            om_mod._Training = original
        run = holder["run"]
        self.n_splits, self.n_checks, self.rule_seconds = run.n_splits, run.n_checks, run.rule_seconds
        self.es_stop_step = run.es_stop_step
        self.check_agree, self.check_total = run.check_agree, run.check_total
        self.device_check = run.device_check
        self.graph_status = {str(k): ("captured" if v else "eager") for k, v in run.graphs.items() if k != "_errors"}
        self.graph_errors = run.graphs.get("_errors", {})
        torch.cuda.synchronize()
        run.graphs.clear()                    # free the graphs now (not during a later capture)
        self.revival_stats = dict(splits=run.n_splits, checks=run.n_checks)
        return out
