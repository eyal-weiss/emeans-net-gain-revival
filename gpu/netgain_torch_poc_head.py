"""Proof of concept of the net-gain rule on the GPU class of the PR head (79440bc): the independent reference for
NetGainEmeansFast on that driver. Not optimized.

It is research/netgain_torch.py (the 98deaca proof of concept) adapted to the head, written separately from the fast
class: per batch, plain torch statistics (index_add_ in FP64) while moves are allowed; per check (every 5 batches,
the driver's schedule), the k-length state and the centroids are read to the host and the NumPy reference decision
(netgain_rule.decide, which reproduces the CPU classes exactly) chooses the moves; the driver's own _apply_splits applies
them, and records them for its closing handover. Overriding the per-batch methods makes the driver train eagerly.

Options: stat; freeze (True: no move once less than one pass remains, 98deaca's rule, as the CPU class
ngp_head.NetGainHead; False: moves until the early stop or the end, the head's closing handover then sorting out the
rows of centroids moved in the last pass); early_stop ("hl": the release's rule: a move restarts the clock; a check
without moves once prod(1 - alpha) since the last move is at most 1/2 stops the revival; None).
"""
import math
import time

import numpy as np
import torch

from faiss.contrib.e_means.e_means_torch import Emeans
from faiss.contrib.e_means.base import _same_operation

import netgain_rule as NR

assert hasattr(Emeans, "_handover_start"), "the PR head's driver (79440bc) is required"


class NetGainEmeansPocHead(Emeans):
    def __init__(self, d, k, *a, stat="medg_pm", freeze=True, early_stop="hl", guard=True, **kw):
        assert stat in NR.STATS and early_stop in (None, "hl")
        super().__init__(d, k, *a, **kw)
        self.stat, self.freeze, self.early_stop, self.guard = stat, bool(freeze), early_stop, bool(guard)
        self._m_n = None
        self.n_checks = 0
        self.rule_seconds = 0.0
        self.es_stop_step = None

    def begin(self, global_n, init_centroids):
        super().begin(global_n, init_centroids)
        assert self._unused_threshold is None, "cost-based revival (the PR's default) is required"
        z = lambda: torch.zeros(self.k, dtype=torch.float64, device=self._usage.device)
        self._m_n, self._m_sum, self._m_sq, self._p2, self._p4 = z(), z(), z(), z(), z()
        self._cum_n, self._age, self._stale = z(), z(), z()
        self._decay_prod = 1.0
        self._prev_delta = None
        self._es_logq, self._es_stopped = 0.0, False
        self.es_stop_step = None

    def _frozen_ng(self):
        """98deaca's freeze (as ngp_head.NetGainHead): no move once less than one pass remains after this batch."""
        if not self.freeze:
            return False
        rows = self.recipe["epoch_rows"]
        done = sum(rows[:self._epoch]) + min(rows[self._epoch], self._epoch_rows + self.recipe["batch_size"])
        return len(rows) > 1 and done >= self._n and sum(rows) - done < self._n

    def _stopped(self):
        return self._es_stopped or self._frozen_ng()

    def _second_labels_supported(self):
        # _assign_rows below passes every output through: keep the driver's second-nearest handover route
        return getattr(self._execution_backend, "second_labels", False) and all(
            _same_operation(getattr(self, name), getattr(NetGainEmeansPocHead, name), self)
            for name in ("_assign_rows", "_accumulate_core"))

    # ------------------------------------------------------------------ statistics
    def _assign_rows(self, x, centroids, centroid_norm=None, distance_out=None, labels_out=None, second_out=None,
                     second_labels_out=None):
        distance, labels = super()._assign_rows(x, centroids, centroid_norm, distance_out, labels_out, second_out,
                                                second_labels_out)
        if second_out is not None and self._m_n is not None and len(x) and not self._stopped():
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
        if self._m_n is None:
            return
        g = 0.0 if replace_usage else float(decay)
        self._stale.mul_(g)
        self._decay_prod *= g
        self._es_logq += math.log(g) if g > 0 else -math.inf
        c = counts.double()
        if ids is None:
            self._cum_n += c
        else:
            self._cum_n.index_add_(0, ids, c)
        self._age += 1.0

    # ------------------------------------------------------------------ revival
    def _choose_splits(self, rng):
        if self._stopped():
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
        src, tgt, self._prev_delta = NR.decide(st, self.stat, 1.0, guard=self.guard)
        self.n_checks += 1
        if self.early_stop == "hl":
            if src is not None:
                self._es_logq = 0.0
            elif self._es_logq <= math.log(0.5):
                self._es_stopped = True
                self.es_stop_step = int(self._step)
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
