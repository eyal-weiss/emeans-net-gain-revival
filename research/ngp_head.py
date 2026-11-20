"""EXPLORATORY: our net-gain rule on the driver of the current FAISS PR #5915 head (79440bc, 2026-11-19), which removed
the freeze (moves continue through the last pass, whose rows are sorted out at the end by the closing handover and twin
reassignment, every centroid then returning the mean of its final last-pass rows) and the PR's pair guard.

NetGainHead(NumpyEmeans of the head; import with the head overlay first on sys.path):
  stat   "medg_pm" (default), "medg_fh" or "mean" (the frozen rule's gain)
  freeze True: no move once less than one pass remains (the timing of 98deaca and of all our runs so far);
         False: the head's timing (moves until the end, closing handover)
  guard  True: our matching's pair guard (as in all our runs so far); False: no guard (as the PR's rule now)
The decision is netgain_rule.decide (shown to reproduce the CPU classes' decisions exactly). The PR's cost-based mode
stays on (unused_threshold=None), so the head's handover applies, and _accumulate_core is not overridden (the head
enables the handover only for the plain class's _accumulate_core): our statistics come from _assign_rows, which the
head calls with the second-nearest distances whenever its removal statistics are tracked (always, for cost-based).
"""
import numpy as np

from faiss.contrib.e_means.e_means import NumpyEmeans

import netgain_rule as NR


class NetGainHead(NumpyEmeans):
    def __init__(self, d, k, *a, stat="medg_pm", w=1.0, freeze=False, guard=True, **kw):
        assert stat in NR.STATS
        super().__init__(d, k, *a, **kw)
        self.stat, self.w, self.freeze, self.guard = stat, float(w), bool(freeze), bool(guard)
        self._m_n = None
        self.n_checks = self.n_moves_last_pass = 0

    def begin(self, global_n, init_centroids):
        super().begin(global_n, init_centroids)
        assert self._unused_threshold is None, "cost-based mode (the PR's default) is required"
        z = lambda: np.zeros(self.k, dtype=np.float64)
        self._m_n, self._m_sum, self._m_sq, self._p2, self._p4 = z(), z(), z(), z(), z()
        self._cum_n, self._age, self._stale = z(), z(), z()
        self._decay_prod = 1.0
        self._prev_delta = None

    def _frozen_ng(self):
        """98deaca's freeze: no move once less than one pass remains after this batch."""
        if not self.freeze:
            return False
        rows = self.recipe["epoch_rows"]
        done = sum(rows[:self._epoch]) + min(rows[self._epoch], self._epoch_rows + self.recipe["batch_size"])
        return len(rows) > 1 and done >= self._n and sum(rows) - done < self._n

    def _last_pass(self):
        return self._epoch == len(self.recipe["epoch_rows"]) - 1

    # ------------------------------------------------------------------ statistics
    def _assign_rows(self, x, centroids, second_out=None, second_labels_out=None):
        distance, labels = super()._assign_rows(x, centroids, second_out, second_labels_out)
        if second_out is not None and self._m_n is not None and len(x) and not self._frozen_ng():
            k = self.k
            margin = np.maximum(np.asarray(second_out, dtype=np.float64) - np.asarray(distance, dtype=np.float64), 0.0)
            margin = np.where(np.isfinite(margin), margin, 0.0)
            self._m_n += np.bincount(labels, minlength=k)
            self._m_sum += np.bincount(labels, weights=margin, minlength=k)
            self._m_sq += np.bincount(labels, weights=margin * margin, minlength=k)
            y = np.asarray(x, dtype=np.float64) - centroids[labels].astype(np.float64)
            pr = np.einsum("ij,ij->i", y, self._axes[labels].astype(np.float64))
            self._p2 += np.bincount(labels, weights=pr * pr, minlength=k)
            self._p4 += np.bincount(labels, weights=pr ** 4, minlength=k)
        return distance, labels

    def _apply_statistics(self, ids, sums, counts, objective, alpha, decay, replace_usage=False):
        super()._apply_statistics(ids, sums, counts, objective, alpha, decay, replace_usage)
        g = 0.0 if replace_usage else decay
        self._stale *= g
        self._decay_prod *= g
        if ids is None:
            self._cum_n += np.asarray(counts, dtype=np.float64)
        else:
            np.add.at(self._cum_n, np.asarray(ids), np.asarray(counts, dtype=np.float64))
        self._age += 1.0

    # ------------------------------------------------------------------ revival
    def _choose_splits(self, rng):
        if self._frozen_ng():
            return None, None
        st = dict(k=self.k, step=self._step, alpha0=self.recipe["alpha0"], total_steps=self.recipe["total_steps"],
                  usage=np.array(self._usage, dtype=np.float64),
                  unorm=self._axis_scale * np.linalg.norm(self._removal_stats[:, :self.d].astype("float64"), axis=1),
                  centroids=np.ascontiguousarray(self._centroids), readout=np.array(self._readout(), copy=True),
                  m_n=self._m_n, m_sum=self._m_sum, m_sq=self._m_sq, p2=self._p2, p4=self._p4,
                  cum_n=self._cum_n, age=self._age, stale=self._stale, decay_prod=self._decay_prod,
                  prev_delta=self._prev_delta)
        src, tgt, self._prev_delta = NR.decide(st, self.stat, self.w, guard=self.guard)
        self.n_checks += 1
        if src is not None and self._last_pass():
            self.n_moves_last_pass += len(src)
        return src, tgt

    def _apply_splits(self, source, target):
        n = super()._apply_splits(source, target)
        for a in (self._m_n, self._m_sum, self._m_sq, self._p2, self._p4, self._cum_n, self._age):
            a[source] = 0.0
            a[target] = 0.0
        self._stale[source] = 1.0
        self._stale[target] = 1.0
        return n
