"""Proof of concept: the net-gain revival rule (medg_pm; also medg_fh and the frozen ngp2_full) on the GPU class of
FAISS PR #5915 (contrib/e_means/e_means_torch.Emeans at commit 98deaca). Not optimized.

What it adds to the PR's class (everything else, including the assignment, the EMA updates, the axis statistics U and
the axis split, is the PR's own code):
  * per batch, on the device, while moves are allowed: for every row, the margin d2 - d1 between its second-nearest and
    its nearest centroid (the PR's assignment already returns d2 for its own removal cost) and its projection p on its
    centroid's split axis; per centroid since its birth: rows, sum of margins, of squared margins, of p^2 and of p^4;
    points and batches since birth; the inherited weight since its last split (cool-down);
  * per revival check (every 5 batches, as the PR): the k-length vectors and the centroids are read to the host and
    netgain_rule.decide (shared with the CPU check, which reproduces the CPU classes exactly) chooses the moves; the
    PR's own _apply_splits applies them on the device; the statistics of both ends restart.
Overriding the per-batch methods makes the PR's class train eagerly (no CUDA-graph replay): the timing is not
representative of an optimized implementation.
"""
import time

import numpy as np
import torch

from faiss.contrib.e_means.e_means_torch import Emeans

import netgain_rule as NR


class NetGainEmeans(Emeans):
    def __init__(self, d, k, *a, stat="medg_pm", w=1.0, **kw):
        assert stat in NR.STATS
        super().__init__(d, k, *a, **kw)
        self.stat, self.w = stat, float(w)
        self._m_n = None
        self.n_checks = 0
        self.rule_seconds = 0.0

    def begin(self, global_n, init_centroids):
        super().begin(global_n, init_centroids)
        assert self._unused_threshold is None, "cost-based revival (the PR's default) is required"
        z = lambda: torch.zeros(self.k, dtype=torch.float64, device=self._usage.device)
        self._m_n, self._m_sum, self._m_sq, self._p2, self._p4 = z(), z(), z(), z(), z()
        self._cum_n, self._age, self._stale = z(), z(), z()
        self._decay_prod = 1.0
        self._prev_delta = None

    # ------------------------------------------------------------------ statistics
    def _assign_rows(self, x, centroids, centroid_norm=None, distance_out=None, labels_out=None, second_out=None):
        distance, labels = super()._assign_rows(x, centroids, centroid_norm, distance_out, labels_out, second_out)
        # the PR asks for d2 exactly when moves are allowed and its removal statistics are tracked
        if second_out is not None and self._m_n is not None and len(x):
            lab = labels.long()
            margin = torch.nan_to_num((second_out.double() - distance.double()).clamp_min(0.0),
                                      nan=0.0, posinf=0.0, neginf=0.0)
            self._m_n.index_add_(0, lab, torch.ones_like(margin))
            self._m_sum.index_add_(0, lab, margin)
            self._m_sq.index_add_(0, lab, margin * margin)
            y = x.double() - self._readout()[lab].double()
            p = (y * self._axes[lab].double()).sum(1)
            p2 = p * p
            self._p2.index_add_(0, lab, p2)
            self._p4.index_add_(0, lab, p2 * p2)
        return distance, labels

    def _apply_statistics(self, ids, sums, counts, objective, alpha, decay, replace_usage=False):
        super()._apply_statistics(ids, sums, counts, objective, alpha, decay, replace_usage)
        g = 0.0 if replace_usage else float(decay)
        self._stale.mul_(g)
        self._decay_prod *= g
        c = counts.double()
        if ids is None:
            self._cum_n += c
        else:
            self._cum_n.index_add_(0, ids, c)
        self._age += 1.0

    # ------------------------------------------------------------------ revival
    def _choose_splits(self, rng):
        if self._frozen():
            return None, None
        t0 = time.perf_counter()
        h = lambda t: t.detach().double().cpu().numpy()
        C = self._readout().detach().float().cpu().numpy()
        st = dict(k=self.k, step=self._step, alpha0=self.recipe["alpha0"], total_steps=self.recipe["total_steps"],
                  usage=h(self._usage),
                  unorm=h(torch.linalg.vector_norm(self._removal_stats[:, :-1], dim=1, dtype=torch.float64)),
                  centroids=C, readout=C, m_n=h(self._m_n), m_sum=h(self._m_sum), m_sq=h(self._m_sq),
                  p2=h(self._p2), p4=h(self._p4), cum_n=h(self._cum_n), age=h(self._age), stale=h(self._stale),
                  decay_prod=self._decay_prod, prev_delta=self._prev_delta)
        src, tgt, self._prev_delta = NR.decide(st, self.stat, self.w)
        self.n_checks += 1
        self.rule_seconds += time.perf_counter() - t0
        if src is None:
            return None, None
        dev = self._usage.device
        return torch.from_numpy(src).to(dev), torch.from_numpy(tgt).to(dev)

    def _apply_splits(self, source, target):
        n = super()._apply_splits(source, target)
        for t in (self._m_n, self._m_sum, self._m_sq, self._p2, self._p4, self._cum_n, self._age):
            t[source] = 0.0
            t[target] = 0.0
        self._stale[source] = 1.0
        self._stale[target] = 1.0
        return n
