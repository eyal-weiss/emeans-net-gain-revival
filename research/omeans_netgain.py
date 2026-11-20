"""Proof of concept: the net-gain revival rule (medg_pm; also medg_fh) inside o-means (owlgebra-ai/faiss
contrib/o_means at 9ecbf0c). 2026-11-19 evening; not optimized.

o-means keeps e-means' count rule (dead = usage below tau times the mean, donors drawn by usage, epsilon split). Here
its revival is replaced by the rule of FAISS PR #5915 with our net-gain decision, everything else of o-means unchanged
(two-level assignment, cells, schedule, EMA kernels, closing mean):
  * per batch (naive torch ops, on the device): an exact two-nearest search of the batch rows against the centroid
    table before the update, giving each row's margin d2 - d1 relative to the centroid o-means assigned it to; its
    projection p on that centroid's split axis; the axis statistics U (EMA of sum p y, Oja's long axis, as the PR);
    per centroid since birth: rows, sums of margins, squared margins, p^2, p^4; points and batches since birth; the
    inherited weight since the last split;
  * revival checks at o-means' checks (every 5 batches), but none in the final pass (the PR's timing: o-means' closing
    mean is the last pass's mean, which a split would mix): netgain_rule.decide on the host; the split is the PR's axis
    split (copies at c -/+ s u, s = sqrt(2 |U| / (pi N))) after o-means' own split kernel has done the bookkeeping
    (usage shared, closing mass, FP16 table), with the cells' child pointers recorded as o-means does.
Units: o-means works in its scaled units (data times 2^e); the rule is invariant to that scale.
"""
import math

import numpy as np
import torch

import faiss.contrib.o_means.o_means as om_mod
from faiss.contrib.o_means.o_means import Omeans, _Training
from faiss.contrib.o_means.triton_kernels import accumulate, apply_splits, ema_update
from faiss.contrib.o_means.triton_two_level import build_cells, permute_rows, record_splits, two_level_assign
from faiss.contrib.e_means.base import _SPLIT_EPSILON, _epoch_seed

import netgain_rule as NR

__all__ = ["OmeansNetGain"]


def top2(rows, C, cn, chunk=4096):
    """Exact two smallest squared distances and their indices (FP32 products)."""
    D = torch.empty((len(rows), 2), dtype=torch.float32, device=rows.device)
    I = torch.empty((len(rows), 2), dtype=torch.int64, device=rows.device)
    for a in range(0, len(rows), chunk):
        q = rows[a:a + chunk]
        d2 = torch.addmm(cn[None, :], q, C.t(), alpha=-2.0) + q.square().sum(1, keepdim=True)
        v, i = torch.topk(d2, 2, dim=1, largest=False)
        D[a:a + chunk], I[a:a + chunk] = v, i
    return D, I


class _NetGainTraining(_Training):
    stat = "medg_pm"
    w = 1.0

    def __init__(self, x, init, sched, seed):
        super().__init__(x, init, sched, seed)
        kp, d, dev = self.C.shape[0], sched.d, x.device
        g = torch.Generator(device=dev)
        g.manual_seed(seed * 31 + 7)
        axes = torch.randn(kp, d, dtype=torch.float32, device=dev, generator=g)
        self.axes = axes / axes.norm(dim=1, keepdim=True)
        self.U = torch.zeros((kp, d), dtype=torch.float64, device=dev)
        self.bU = torch.zeros((kp, d), dtype=torch.float64, device=dev)
        z = lambda: torch.zeros(kp, dtype=torch.float64, device=dev)
        self.m_n, self.m_sum, self.m_sq, self.p2, self.p4 = z(), z(), z(), z(), z()
        self.cum_n, self.age, self.stale, self.bcnt = z(), z(), z(), z()
        self.decay_prod = 1.0
        self.prev_delta = None
        self.n_splits = self.n_checks = 0

    def per_centroid(self):
        return [self.C, self.c16, self.cn, self.usage, self.close_sums, self.close_cnt, self.axes, self.U,
                self.m_n, self.m_sum, self.m_sq, self.p2, self.p4, self.cum_n, self.age, self.stale]

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

    def stats_batch(self, rows, lab):
        """rows (R, d) float32 scaled, lab (R,) the centroid o-means assigned; table before the update."""
        k, d = self.sched.k, self.sched.d
        C = self.C[:k, :d]
        D, I = top2(rows, C, C.square().sum(1))
        own = (rows.double() - C[lab].double()).square().sum(1)
        other = torch.where(I[:, 0] == lab, D[:, 1], D[:, 0]).double()
        margin = torch.nan_to_num((other - own).clamp_min(0.0), nan=0.0, posinf=0.0)
        y = rows.double() - C[lab].double()
        p = (y * self.axes[lab].double()).sum(1)
        self.bU.index_add_(0, lab, p[:, None] * y)
        ones = torch.ones_like(margin)
        self.m_n.index_add_(0, lab, ones)
        self.bcnt.index_add_(0, lab, ones)
        self.m_sum.index_add_(0, lab, margin)
        self.m_sq.index_add_(0, lab, margin * margin)
        self.p2.index_add_(0, lab, p * p)
        self.p4.index_add_(0, lab, (p * p) ** 2)

    def after_update(self, alpha, track):
        a = float(alpha)
        if track:
            self.U.mul_(1.0 - a).add_(self.bU, alpha=a)
            nrm = self.U.norm(dim=1, keepdim=True)
            self.axes = torch.where(nrm > 0, (self.U / nrm.clamp_min(1e-300)).float(), self.axes)
            self.bU.zero_()
        self.stale.mul_(1.0 - a)
        self.decay_prod *= 1.0 - a
        self.cum_n += self.bcnt
        self.bcnt.zero_()
        self.age[:self.sched.k] += 1.0

    def revive(self, step_i, alpha):
        k, d, dev = self.sched.k, self.sched.d, self.C.device
        h = lambda t: t[:k].detach().double().cpu().numpy()
        Ck = self.C[:k, :d].detach().float().cpu().numpy()
        st = dict(k=k, step=step_i, alpha=alpha, usage=h(self.usage), unorm=h(self.U[:k].norm(dim=1)),
                  centroids=Ck, readout=Ck, m_n=h(self.m_n), m_sum=h(self.m_sum), m_sq=h(self.m_sq),
                  p2=h(self.p2), p4=h(self.p4), cum_n=h(self.cum_n), age=h(self.age), stale=h(self.stale),
                  decay_prod=self.decay_prod, prev_delta=self.prev_delta)
        src, tgt, self.prev_delta = NR.decide(st, self.stat, self.w)
        self.n_checks += 1
        if src is None:
            return
        s = torch.from_numpy(src).to(dev)
        t = torch.from_numpy(tgt).to(dev)
        npair = len(src)
        # the PR's axis split, computed before o-means' kernel halves the usage
        norm = self.U[s].norm(dim=1)
        mass = self.usage[s].double()
        valid = (norm > 0) & (mass > 0) & torch.isfinite(norm) & torch.isfinite(mass)
        spread = torch.sqrt(2.0 * torch.where(valid, norm / torch.where(valid, mass, 1.0), 0.0) / math.pi)
        c = self.C[s, :d].double()
        off = spread[:, None] * self.axes[s].double()
        lo, hi = (c - off).float(), (c + off).float()
        m = torch.tensor(npair, device=dev)
        recency = self.sched.recency_close
        apply_splits(self.C, self.c16, self.cn, self.usage, s, t, m, d, _SPLIT_EPSILON, max_pairs=npair,
                     wsum=self.close_sums if recency else None, wcnt=self.close_cnt if recency else None)
        keep = spread > 0              # spread 0: keep o-means' epsilon separation
        if keep.any():
            sk, tk = s[keep], t[keep]
            self.C[sk, :d], self.C[tk, :d] = lo[keep], hi[keep]
            for r in (sk, tk):
                self.c16[r, :d] = self.C[r, :d].to(self.c16.dtype)
                self.cn[r] = self.c16[r, :d].float().square().sum(1)
        record_splits(self.cells.child, s, t, m, npair)
        self.U[s] /= 2
        self.U[t] = self.U[s]
        self.axes[t] = self.axes[s]
        for a in (self.m_n, self.m_sum, self.m_sq, self.p2, self.p4, self.cum_n, self.age):
            a[s] = 0.0
            a[t] = 0.0
        self.stale[s] = 1.0
        self.stale[t] = 1.0
        self.n_splits += npair

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
            labels = two_level_assign(src, self.c16, self.cn, self.cells, self.ws,
                                      sched.nprobe, idx=idx, rows=R)
            if labels_out is not None and epoch == last_epoch:
                labels_out[perm[step.b0:step.b1]] = self.ids[labels]
            rows_done += R
            # NG: the PR's freeze (no move once the remaining rows are less than one pass)
            frozen = len(sched.epoch_rows) > 1 and rows_done >= n and total_rows - rows_done < n
            rows = (src[idx] if idx is not None else src[:R])[:, :d].float()                    # NG
            if not frozen:
                self.stats_batch(rows, labels[:R].long())                                        # NG
            else:
                self.bcnt.index_add_(0, labels[:R].long(), torch.ones(R, dtype=torch.float64, device=rows.device))
            accumulate(src, labels, self.sums, self.cnt, d, idx=idx)
            close = step.omega is not None
            ema_update(self.C, self.c16, self.cn, self.usage, self.sums, self.cnt, k, d,
                       step.alpha, self.close_sums if close else None,
                       self.close_cnt if close else None,
                       omega=step.omega if close else 1.0)
            self.after_update(step.alpha, not frozen)                                            # NG
            if step.check and not frozen:                                                        # NG
                self.revive(step_i, step.alpha)
        return self.closing_mean()


class OmeansNetGain(Omeans):
    """o-means with the net-gain revival rule (stat "medg_pm", "medg_fh" or "mean")."""

    def __init__(self, d, k, nepochs=5, stat="medg_pm", w=1.0, **kw):
        assert stat in NR.STATS
        super().__init__(d, k, nepochs=nepochs, **kw)
        self.stat, self.w = stat, float(w)
        self.revival_stats = None

    def train(self, x, init_centroids=None):
        cls = type("_RuleTraining", (_NetGainTraining,), dict(stat=self.stat, w=self.w))
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
        self.revival_stats = dict(splits=run.n_splits, checks=run.n_checks)
        return out
