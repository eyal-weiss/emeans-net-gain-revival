"""NetGainRevivalEmeans: e-means with net-gain revival, on the e-means driver of FAISS pull request #5915 at its head
(commit 79440bc).

The driver keeps its schedule, EMA updates, split statistics U (Oja's rule), axis split, cost-based statistics and
closing handover: moves may continue through the last pass, whose rows held by a centre when it moved are handed to
their second-nearest centre or assigned again at the end, and every centre then returns the mean of its last-pass rows.
This class changes only the revival check (which centres to free and which clusters to split) and adds what the
decision needs:
  * while moves are allowed, every point's margin d2 - d1 (from the driver's own two-nearest search) and its
    projection on its cluster's split axis are added to per-centre sums since birth (counts, margins, squared margins,
    p^2, p^4);
  * per centre: batches and points since birth, and the coefficient h of statistics inherited at the last split;
  * at each check (every five batches): rule.decide, with a cool-down (a centre with h > 1/2 is neither freed nor used
    as a donor) and the pair guard; the split itself is the driver's (which also records it for the closing handover);
  * an automatic early end of the move phase once one EMA half-life has passed without a check that accepted a move
    (early_stop="hl", the default);
  * freeze=True: no move once less than one pass remains (the timing of the pull request at 98deaca, which the paper's
    first two rounds used). The default, freeze=False, keeps moving until the end, like the driver's own rule.

Identical decisions to the research code of the paper's third round on this driver (research/ngp_head.py with the
guard, early_stop=None), with the NumPy computations or with the optional C++ kernels (native/build.sh);
tests/test_netgain.py and research/verify_head.py check this.
"""
import math
import os

import numpy as np
import faiss
from faiss.contrib.e_means import NumpyEmeans
from faiss.contrib.e_means.base import _scheduled_alpha

from . import native, rule

if not hasattr(NumpyEmeans, "_handover_start"):
    raise ImportError("this module needs the e-means driver of PR #5915 at its head (79440bc); "
                      "net_gain_revival.emeans_98deaca serves the driver at 98deaca")


class NetGainRevivalEmeans(NumpyEmeans):
    """e-means with net-gain revival.

    stat: "medg_pm" (posterior median gain, Paule-Mandel prior; the paper's rule) or "medg_fh" (Fay-Herriot prior).
    early_stop: "hl" (end the move phase after one EMA half-life without an accepted move), a float eps (a check also
        counts as quiet when the predicted net gain of its moves is below eps times the last batch's squared error), or
        None (moves until the end, or until the freeze).
    freeze: True stops moves once less than one pass remains (default False).
    use_native: use the C++ kernels (default: when built). Other arguments are NumpyEmeans's (e.g. nepochs, seed).
    After train(): n_splits, n_checks, n_candidates, n_moves_last_pass, es_stop_step (None if the early stop did not
    fire), closing_handover_stats (the driver's).
    """

    def __init__(self, d, k, *args, stat="medg_pm", early_stop="hl", freeze=False, use_native=None, **kw):
        assert stat in rule.STATS, stat
        assert early_stop is None or early_stop == "hl" or (isinstance(early_stop, float) and early_stop > 0)
        if kw.get("unused_threshold") is not None:
            raise ValueError("net-gain revival replaces the revival rule and needs the driver's cost-based statistics; "
                             "do not set unused_threshold")
        super().__init__(d, k, *args, **kw)
        if self.spherical:
            raise ValueError("net-gain revival is defined for squared Euclidean distances (spherical=False)")
        self.stat, self.early_stop, self.freeze = stat, early_stop, bool(freeze)
        self.use_native = native.available() if use_native is None else bool(use_native)
        if self.use_native and not native.available():
            raise RuntimeError("C++ kernels not built: run net_gain_revival/native/build.sh")
        self.n_checks = self.n_candidates = self.n_moves_last_pass = 0
        self.es_stop_step = None
        self._m_n = None

    # ------------------------------------------------------------------ driver hooks
    def begin(self, global_n, init_centroids):
        super().begin(global_n, init_centroids)
        self.es_stop_step = None
        if self._unused_threshold is not None:      # k = 1: the driver revives nothing, and neither does this class
            self._m_n = None
            return
        k = self.k
        z = lambda: np.zeros(k, dtype=np.float64)
        self._stale, self._cum_n, self._age = z(), z(), z()              # h_j; points and batches since birth
        self._m_n, self._m_sum, self._m_sq, self._p2, self._p4 = z(), z(), z(), z(), z()
        self._decay_prod = 1.0                                            # prod(1 - alpha_t) since the start
        self._prev_delta = None
        self._es_stopped, self._es_logq, self._es_obj = False, 0.0, None

    def _moves_allowed(self):
        """False after the early stop, and with freeze=True once less than one pass remains after this batch."""
        if self._es_stopped:
            return False
        if not self.freeze:
            return True
        rows = self.recipe["epoch_rows"]
        done = sum(rows[:self._epoch]) + min(rows[self._epoch], self._epoch_rows + self.recipe["batch_size"])
        return not (len(rows) > 1 and done >= self._n and sum(rows) - done < self._n)

    def _assign_rows(self, x, centroids, second_out=None, second_labels_out=None):
        distance, labels = super()._assign_rows(x, centroids, second_out, second_labels_out)
        # The driver asks for the second-nearest distances whenever it tracks its cost-based statistics, that is for
        # every training row (not for the rows its closing handover assigns again).
        if second_out is None or self._m_n is None or len(x) == 0 or not self._moves_allowed():
            return distance, labels
        if self.use_native:
            native.margin_stats_labeled(
                np.ascontiguousarray(x, dtype=np.float32), np.ascontiguousarray(distance, dtype=np.float32),
                np.ascontiguousarray(second_out, dtype=np.float32), np.ascontiguousarray(labels, dtype=np.int64),
                np.ascontiguousarray(centroids, dtype=np.float32), self._axes, self._m_n, self._m_sum, self._m_sq,
                self._p2, self._p4, max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
            return distance, labels
        k = self.k
        margin = np.maximum(np.asarray(second_out, dtype=np.float64) - np.asarray(distance, dtype=np.float64), 0.0)
        margin = np.where(np.isfinite(margin), margin, 0.0)
        self._m_n += np.bincount(labels, minlength=k)
        self._m_sum += np.bincount(labels, weights=margin, minlength=k)
        self._m_sq += np.bincount(labels, weights=margin * margin, minlength=k)
        y = np.asarray(x, dtype=np.float64) - centroids[labels].astype(np.float64)
        p = np.einsum("ij,ij->i", y, self._axes[labels].astype(np.float64))
        self._p2 += np.bincount(labels, weights=p * p, minlength=k)
        self._p4 += np.bincount(labels, weights=p ** 4, minlength=k)
        return distance, labels

    def _apply_statistics(self, ids, sums, counts, objective, alpha, decay, replace_usage=False):
        super()._apply_statistics(ids, sums, counts, objective, alpha, decay, replace_usage)
        if self._m_n is None:
            return
        g = 0.0 if replace_usage else decay
        self._stale *= g
        self._decay_prod *= g
        if ids is None:
            self._cum_n += np.asarray(counts, dtype=np.float64)
        else:
            np.add.at(self._cum_n, np.asarray(ids), np.asarray(counts, dtype=np.float64))
        self._age += 1.0
        if self.early_stop is not None and not self._es_stopped:
            gg = 0.0 if replace_usage else float(decay)
            self._es_logq += math.log(gg) if gg > 0.0 else -math.inf
            self._es_obj = None if objective is None else float(objective)

    # ------------------------------------------------------------------ the check
    def _choose_splits(self, rng):
        if self._m_n is None:
            return super()._choose_splits(rng)
        if not self._moves_allowed():
            return None, None
        C = np.ascontiguousarray(self._centroids)
        D, I = faiss.knn(C, C, 2)
        D, I = np.ascontiguousarray(D, dtype=np.float32), np.ascontiguousarray(I, dtype=np.int64)
        alpha = float(_scheduled_alpha(self.recipe["alpha0"], self._step, self.recipe["total_steps"]))
        alpha = min(max(alpha, 1e-6), 1.0)
        readout = np.ascontiguousarray(self._readout(), dtype=np.float32)
        rs = self._removal_stats
        stats = (self._stale, self._age, self._cum_n, self._decay_prod, alpha, self._step,
                 self._m_n, self._m_sum, self._m_sq, self._p2, self._p4, D, I)
        if self.use_native:
            src, tgt, delta, info = native.decide(self._usage, rs, (2.0 / np.pi) * self._axis_scale, *stats,
                                                  self.stat, self._prev_delta, readout)
        else:
            raw = (2.0 / np.pi) * self._axis_scale * np.linalg.norm(rs[:, :self.d].astype("float64"), axis=1)
            src, tgt, delta, info = rule.decide(raw, self._usage, *stats, readout, self.stat, self._prev_delta)
        self._prev_delta = delta
        self.n_checks += 1
        self.n_candidates += info["n_candidates"]
        if src is not None and self._epoch == len(self.recipe["epoch_rows"]) - 1:
            self.n_moves_last_pass += len(src)
        if self.early_stop is not None:
            self._early_stop_check(src, tgt, info)
        return src, tgt

    def _early_stop_check(self, src, tgt, info):
        """A quiet check: no move ("hl"), or a net gain below eps * the last batch's error. The move phase ends at a
        quiet check once the clock (sum of log(1 - alpha) since the last non-quiet check) reaches log(1/2)."""
        if self.early_stop == "hl":
            quiet = src is None
        else:
            net = 0.0 if src is None else float((1.0 * info["gain"][src] - info["cost"][tgt]).sum())
            quiet = self._es_obj is not None and net < self.early_stop * self._es_obj
        if not quiet:
            self._es_logq = 0.0
        elif self._es_logq <= math.log(0.5):
            self._es_stopped = True
            self.es_stop_step = int(self._step)

    def _apply_splits(self, source, target):
        """The driver's axis split (which also records the move for the closing handover); evidence since birth
        restarts at both ends, and the cool-down starts."""
        n = super()._apply_splits(source, target)
        if self._m_n is not None:
            for arr in (self._cum_n, self._age, self._m_n, self._m_sum, self._m_sq, self._p2, self._p4):
                arr[source] = arr[target] = 0.0
            self._stale[source] = self._stale[target] = 1.0
        return n
