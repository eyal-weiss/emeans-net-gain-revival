"""NetGainRevivalEmeans: e-means with net-gain revival, on the e-means driver of FAISS pull request #5915 (98deaca).

The driver keeps its schedule, EMA updates, split statistics U (Oja's rule) and axis split. This class changes only the
revival check (which centres to free and which clusters to split) and adds what the decision needs:
  * while moves are allowed, every point's two nearest centres are found; its margin d2 - d1 and its projection on its
    cluster's split axis are added to per-centre sums since birth (counts, margins, squared margins, p^2, p^4);
  * per centre: batches and points since birth, and the coefficient h of statistics inherited at the last split;
  * at each check (every five batches): rule.decide, with a cool-down (a centre with h > 1/2 is neither freed nor used
    as a donor) and the pair guard; the split itself is the driver's axis split;
  * no move once less than one pass remains (the pull request's freeze), and an automatic early end of the move phase
    once one EMA half-life has passed without a check that accepted a move (early_stop="hl", the default).

Identical decisions to the research code of the paper (research/ngp_stop.py, ngp_fast.py, ngp_variants*.py,
ablation*.py with the paper's configuration), with the NumPy computations or with the optional C++ kernels
(native/build.sh); tests/test_netgain.py checks this.
"""
import math
import os

import numpy as np
import faiss
from faiss.contrib.e_means import NumpyEmeans
from faiss.contrib.e_means.base import _scheduled_alpha
from faiss.contrib.e_means.e_means import _separate_splits

from . import native, rule


class NetGainRevivalEmeans(NumpyEmeans):
    """e-means with net-gain revival.

    stat: "medg_pm" (posterior median gain, Paule-Mandel prior; the paper's rule) or "medg_fh" (Fay-Herriot prior).
    early_stop: "hl" (end the move phase after one EMA half-life without an accepted move), a float eps (a check also
        counts as quiet when the predicted net gain of its moves is below eps times the last batch's squared error), or
        None (moves until the freeze).
    use_native: use the C++ kernels (default: when built). Other arguments are NumpyEmeans's (e.g. nepochs, seed).
    After train(): n_splits, n_checks, n_candidates, es_stop_step (None if the early stop did not fire).
    """

    def __init__(self, d, k, *args, stat="medg_pm", early_stop="hl", use_native=None, **kw):
        assert stat in rule.STATS, stat
        assert early_stop is None or early_stop == "hl" or (isinstance(early_stop, float) and early_stop > 0)
        if "unused_threshold" in kw:
            raise ValueError("net-gain revival replaces the size threshold; do not set unused_threshold")
        # The driver runs in its count mode with the split statistics kept (its own second-nearest search is not
        # needed: this class searches the two nearest centres itself) and the freeze is applied by _frozen below.
        super().__init__(d, k, *args, unused_threshold=0.1, **kw)
        self.stat, self.early_stop = stat, early_stop
        self.use_native = native.available() if use_native is None else bool(use_native)
        if self.use_native and not native.available():
            raise RuntimeError("C++ kernels not built: run net_gain_revival/native/build.sh")
        self.n_checks = self.n_candidates = 0
        self.es_stop_step = None

    # ------------------------------------------------------------------ driver hooks
    def _keeps_removal_stats(self):
        return True

    def _frozen(self):
        """No move once less than one pass remains (the pull request's freeze), or after the early stop."""
        if getattr(self, "_es_stopped", False):
            return True
        rows = self.recipe["epoch_rows"]
        done = sum(rows[:self._epoch]) + min(rows[self._epoch], self._epoch_rows + self.recipe["batch_size"])
        return len(rows) > 1 and done >= self._n and sum(rows) - done < self._n

    def begin(self, global_n, init_centroids):
        super().begin(global_n, init_centroids)
        k = self.k
        self._moved = np.zeros(k, dtype=bool)          # centres moved by splits (for the driver's closing step)
        z = lambda: np.zeros(k, dtype=np.float64)
        self._stale, self._cum_n, self._age = z(), z(), z()              # h_j; points and batches since birth
        self._m_n, self._m_sum, self._m_sq, self._p2, self._p4 = z(), z(), z(), z(), z()
        self._decay_prod = 1.0                                            # prod(1 - alpha_t) since the start
        self._prev_delta = None
        self._es_stopped, self._es_logq, self._es_obj = False, 0.0, None
        self.es_stop_step = None

    def _assign_rows(self, x, centroids, second_out=None):
        if second_out is not None or self._centroids is None or len(x) == 0 or len(centroids) < 2 or self._frozen():
            return super()._assign_rows(x, centroids, second_out)
        xs = np.ascontiguousarray(x, dtype=np.float32)
        D, I = faiss.knn(xs, centroids, 2)
        if self.use_native:
            return native.margin_stats(xs, np.ascontiguousarray(D), np.ascontiguousarray(I),
                                       np.ascontiguousarray(centroids, dtype=np.float32), self._axes, self._m_n,
                                       self._m_sum, self._m_sq, self._p2, self._p4,
                                       max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
        lab = np.where(D[:, 0] == D[:, 1], I.min(axis=1), I[:, 0])            # ties go to the lowest index
        other = np.where(I[:, 0] == lab, D[:, 1], D[:, 0]).astype(np.float64)
        margin = np.maximum(other - D[:, 0].astype(np.float64), 0.0)
        margin = np.where(np.isfinite(margin), margin, 0.0)
        k = self.k
        self._m_n += np.bincount(lab, minlength=k)
        self._m_sum += np.bincount(lab, weights=margin, minlength=k)
        self._m_sq += np.bincount(lab, weights=margin * margin, minlength=k)
        y = xs.astype(np.float64) - centroids[lab].astype(np.float64)
        p = np.einsum("ij,ij->i", y, self._axes[lab].astype(np.float64))
        self._p2 += np.bincount(lab, weights=p * p, minlength=k)
        self._p4 += np.bincount(lab, weights=p ** 4, minlength=k)
        return np.ascontiguousarray(D[:, 0], dtype=np.float32), lab.astype(np.int64)

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
        if self.early_stop is not None and not self._es_stopped:
            gg = 0.0 if replace_usage else float(decay)
            self._es_logq += math.log(gg) if gg > 0.0 else -math.inf
            self._es_obj = None if objective is None else float(objective)

    # ------------------------------------------------------------------ the check
    def _choose_splits(self, rng):
        if self._frozen():
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

    def _split_positions(self, source):
        """The driver's axis split: c_s -/+ a_s u_s with a_s = sqrt(2 |U_s| / (pi N_s))."""
        centroids = self._centroids[source].astype("float64")
        norm = self._axis_scale * np.linalg.norm(self._removal_stats[source, :-1].astype("float64"), axis=1)
        mass = self._usage[source].astype("float64")
        valid = (norm > 0) & (mass > 0) & np.isfinite(norm) & np.isfinite(mass)
        spread = np.sqrt(2.0 * np.where(valid, norm / np.where(valid, mass, 1.0), 0.0) / np.pi)
        offset = spread[:, None] * self._axes[source].astype("float64")
        donor = (centroids - offset).astype("float32")
        child = (centroids + offset).astype("float32")
        _separate_splits(donor, child)
        return donor, child

    def _apply_splits(self, source, target):
        donor, child = self._split_positions(source)
        half_usage = self._usage[source] / 2.0
        self._usage[source] = half_usage
        self._centroids[source] = donor
        self._usage[target] = half_usage
        self._centroids[target] = child
        if self._removal_stats is not None:
            self._removal_stats[source] /= 2
            self._removal_stats[target] = self._removal_stats[source]
            self._axes[target] = self._axes[source]
        if self._pass_sums is not None and self._moved is not None:
            self._moved[source] = self._moved[target] = True
        self._stale[source] = self._stale[target] = 1.0                   # cool-down starts
        for arr in (self._cum_n, self._age, self._m_n, self._m_sum, self._m_sq, self._p2, self._p4):
            arr[source] = arr[target] = 0.0                               # evidence since birth restarts
        self.n_splits += len(source)
        self._invalidate_readout()
        return len(source)
