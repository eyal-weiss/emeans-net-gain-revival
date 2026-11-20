"""The posterior net-gain rule with its runtime-relevant parts in C++ (protocol_ngp.md section 2).

Same rule and same random streams as the reference (ablation.AblationEmeans); two parts are replaced by the kernels in
ngp_kernels.cpp:
  * the per-row statistics of the two-nearest search (labels, margins, projections on the split axes, per-cluster
    sums), which the reference computes with NumPy (and, with margin_stride 1, after copying the whole batch);
  * the greedy matching (_match_v2), a Python loop in the reference, together with the dictionary of nearest centres
    that the reference builds for it.
The search itself stays in FAISS (as in the reference and the PR), the driver is the PR's, and the per-check estimates
stay vectorized (O(k) per check). Usage: the runner imports this module instead of ablation when NGP_IMPL=cpp.
"""
import ctypes
import os
import time

import numpy as np
import faiss

import ablation as _ref
from ablation import CONFIGS, TIMES  # noqa: F401  (the same objects as in the reference module)

_HERE = os.path.dirname(os.path.abspath(__file__))
_lib = ctypes.CDLL(os.path.join(_HERE, "ngp_kernels.so"))
_P, _I = ctypes.c_void_p, ctypes.c_int64
_lib.ngp_margin_stats.restype = ctypes.c_int
_lib.ngp_margin_stats.argtypes = [_I, _I, _I] + [_P] * 12 + [ctypes.c_int]
_lib.ngp_match_v2.restype = _I
_lib.ngp_match_v2.argtypes = [_I, _P, _I, _P, _P, _P, ctypes.c_double, _P, _I, _I, _P, ctypes.c_int, _P, _P]


def _ptr(a, dtype):
    if a.dtype != dtype or not a.flags.c_contiguous:
        raise TypeError(f"expected a C-contiguous {np.dtype(dtype)} array, got {a.dtype} {a.flags.c_contiguous}")
    return a.ctypes.data


def _threads():
    return max(1, int(os.environ.get("OMP_NUM_THREADS", "1")))


def margin_stats(xs, D, I, C, A, m_n, m_sum, m_sq, p2, p4):
    """Labels, nearest distances and the per-cluster sums (added to m_n, m_sum, m_sq, p2, p4 in place)."""
    n, d = xs.shape
    k = len(C)
    labels = np.empty(n, dtype=np.int64)
    dist = np.empty(n, dtype=np.float32)
    rc = _lib.ngp_margin_stats(n, d, k, _ptr(xs, np.float32), _ptr(D, np.float32), _ptr(I, np.int64),
                               _ptr(C, np.float32), _ptr(A, np.float32), _ptr(labels, np.int64),
                               _ptr(dist, np.float32), _ptr(m_n, np.float64), _ptr(m_sum, np.float64),
                               _ptr(m_sq, np.float64), _ptr(p2, np.float64), _ptr(p4, np.float64), _threads())
    if rc != 0:
        raise RuntimeError("ngp_margin_stats: label out of range")
    return dist, labels


def match_v2(cand, donors, cost, gain, w, nearest, centroids, forbid_nearest):
    """(source, kept) arrays, or (None, None) if no move is accepted."""
    cand = np.ascontiguousarray(cand, dtype=np.int64)
    donors = np.ascontiguousarray(donors, dtype=np.int64)
    cost = np.ascontiguousarray(cost, dtype=np.float64)
    gain = np.ascontiguousarray(gain, dtype=np.float64)
    nearest = np.ascontiguousarray(nearest, dtype=np.int64)
    centroids = np.ascontiguousarray(centroids, dtype=np.float32)
    k, d = centroids.shape
    src = np.empty(len(cand), dtype=np.int64)
    kept = np.empty(len(cand), dtype=np.int64)
    m = _lib.ngp_match_v2(len(cand), _ptr(cand, np.int64), len(donors), _ptr(donors, np.int64),
                          _ptr(cost, np.float64), _ptr(gain, np.float64), float(w), _ptr(nearest, np.int64), k, d,
                          _ptr(centroids, np.float32), int(bool(forbid_nearest)), _ptr(src, np.int64),
                          _ptr(kept, np.int64))
    if m < 0:
        raise RuntimeError("ngp_match_v2: id out of range")
    if m == 0:
        return None, None
    return src[:m].copy(), kept[:m].copy()


class AblationEmeans(_ref.AblationEmeans):

    def _assign_rows_sampled(self, x, centroids):
        if self._pend is not None:                     # propose-then-confirm variants: reference path
            return super()._assign_rows_sampled(x, centroids)
        n = len(x)
        s = self.margin_stride
        off = int(self._margin_rng.randint(s))         # the reference's random stream, also for s = 1
        if s == 1:
            xs = np.ascontiguousarray(x, dtype=np.float32)          # every row: no copy, no scatter
            sidx = None
        else:
            smask = np.zeros(n, dtype=bool)
            smask[off::s] = True
            sidx, ridx = np.flatnonzero(smask), np.flatnonzero(~smask)
            distance = np.empty(n, dtype=np.float32)
            labels = np.empty(n, dtype=np.int64)
            t0 = time.perf_counter()
            if len(ridx):
                d1, l1 = _ref.NumpyEmeans._assign_rows(self, np.ascontiguousarray(x[ridx]), centroids)
                distance[ridx], labels[ridx] = d1, l1
            TIMES["search_top1"] += time.perf_counter() - t0
            if len(sidx) == 0:
                return distance, labels
            xs = np.ascontiguousarray(x[sidx])
        t0 = time.perf_counter()
        D, I = faiss.knn(xs, centroids, 2)
        TIMES["search_sampled_top2"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        dist, lab = margin_stats(xs, np.ascontiguousarray(D), np.ascontiguousarray(I),
                                 np.ascontiguousarray(centroids, dtype=np.float32), self._axes,
                                 self._m_n, self._m_sum, self._m_sq, self._p2, self._p4)
        TIMES["margin_stats"] += time.perf_counter() - t0
        if sidx is None:
            return dist, lab
        distance[sidx], labels[sidx] = dist, lab
        return distance, labels

    def _choose_net_gain(self):
        """The reference's _choose_net_gain for ng_post2 with matcher v2 and the exact delta search (every config
        of the confirmatory study), with the matching in C++. Any other combination uses the reference."""
        if not (self.removal == "ng_post2" and self.matcher == "v2" and self.knn == "exact"):
            return super()._choose_net_gain()
        usage = self._usage.astype("float64")
        gain = (2.0 / np.pi) * self._axis_scale * np.linalg.norm(
            self._removal_stats[:, :self.d].astype("float64"), axis=1)
        if self.cooldown == "none":
            cooling = np.zeros(self.k, dtype=bool)
        elif self.cooldown == "check":
            cooling = (self._step - self._split_step) <= 5
        else:
            cooling = self._stale > (0.5 if self.cooldown == "half" else 0.25)
        rate = self._posterior_rates()
        gain = self._posterior_gains(gain, rate)
        C = np.ascontiguousarray(self._centroids)
        t0 = time.perf_counter()
        D, I = faiss.knn(C, C, 2)
        TIMES["delta_search"] += time.perf_counter() - t0
        ids = np.arange(self.k)
        nn_other = np.where(I[:, 0] == ids, I[:, 1], I[:, 0])
        delta = np.where(I[:, 0] == ids, D[:, 1], D[:, 0]).astype("float64")
        missing = (nn_other < 0) | ~np.isfinite(delta)
        delta = np.where(missing, np.finfo("float64").max / 4, delta)
        nn_other = np.where(nn_other < 0, ids, nn_other)
        cost = rate * np.maximum(delta, 0.0)
        ratio, self._rho = self._margin_ratios(delta)
        cost = cost * ratio
        if self.trigger == "event":
            self._delta_ref = np.array(delta, dtype=np.float64, copy=True)
            self._pos_ref = np.array(self._centroids, dtype=np.float64, copy=True)
            self._r_ref = np.array(ratio, dtype=np.float64, copy=True)
            self._new_since = set()
        self._calib_pred = dict(cost=cost, gain=gain, delta=delta, unbiased=True, rate=rate)
        self._check_cache = dict(rate=rate, delta=delta)
        donors = np.flatnonzero((usage > 0) & ~cooling)
        if len(donors) == 0:
            return None, None
        donors = donors[np.argsort(-gain[donors], kind="stable")]
        cand = np.flatnonzero((cost < self.w * gain[donors[0]]) & ~cooling)     # nothing else can pass
        if len(cand) == 0:
            return None, None
        cand = cand[np.argsort(cost[cand], kind="stable")]
        self.n_candidates += len(cand)
        centroids = self._readout()
        t0 = time.perf_counter()
        out = match_v2(cand, donors, cost, gain, self.w, nn_other, centroids, self.no_fallback_donor)
        TIMES["match"] += time.perf_counter() - t0
        return out
