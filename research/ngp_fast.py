"""The posterior net-gain rule with the whole per-check decision in C++ (ngp_decide.cpp; 2026-11-19 night).

Subclass of ngp_variants2.AblationEmeans (run with NGP_IMPL=cpp, so the per-row statistics are the frozen C++ kernels
as in round 2). _choose_net_gain calls ngp_decide once per check: posterior rates, gain noise, posterior gains (stats
"mean" = frozen ngp2_full, "medg_pm", "medg_fh"), margin ratios, costs, donor and candidate order and the greedy
matching. The nearest-other-centre search stays in FAISS. Only the confirmatory configuration takes this path
(removal ng_post2, matcher v2, exact knn, cool-down "half", moderate False, gain_noise "fixed_n", gain_eb "normal",
periodic checks, no propose-then-confirm); every other configuration uses the parent.
Side effects kept: _rho, _calib_pred, _check_cache (medg_fh reads the previous check's delta), n_candidates. The
reference's diagnostics _gd and _ratio_prior are NOT produced (use the reference classes for diag_t5-style traces).
Exactness: see ngp_decide.cpp (NumPy 2.5.5 + the env's OpenBLAS; bit-identical estimates on CPUs without AVX-512).
usage: from ngp_fast import AblationEmeans, CONFIGS, TIMES   (same constructor as ngp_variants2.AblationEmeans)
"""
import ctypes
import os
import time

import numpy as np
import faiss

assert os.environ.get("NGP_IMPL") == "cpp", "ngp_fast extends the C++ implementation: set NGP_IMPL=cpp"
import ngp_variants2 as _V  # noqa: E402
from ngp_variants2 import CONFIGS, TIMES  # noqa: E402,F401
from faiss.contrib.e_means.base import _scheduled_alpha  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_lib = ctypes.CDLL(os.path.join(_HERE, "ngp_decide.so"))
_P, _I, _D = ctypes.c_void_p, ctypes.c_int64, ctypes.c_double
_lib.ngp_decide.restype = _I
_lib.ngp_decide.argtypes = ([_I, _I, _P, _P, _I, _D, _P, _P, _P, _D, _D, _D] + [_P] * 7 +
                            [ctypes.c_int, _P, _D, _P, ctypes.c_int] + [_P] * 8)
STAT = {"mean": 0, "medg_pm": 1, "medg_fh": 2}


def _f64(a):
    return np.ascontiguousarray(a, dtype=np.float64)


class AblationEmeans(_V.AblationEmeans):

    def _fast_ok(self):
        return (self.removal == "ng_post2" and self.matcher == "v2" and self.knn == "exact"
                and self.cooldown == "half" and not self.moderate and self.stat in STAT
                and getattr(self, "gain_noise", "fixed_n") == "fixed_n"
                and getattr(self, "gain_eb", "normal") == "normal"
                and getattr(self, "trigger", "periodic") == "periodic" and not getattr(self, "confirm", False))

    def _choose_net_gain(self):
        if not self._fast_ok():
            return super()._choose_net_gain()
        k, d = self.k, self.d
        C = np.ascontiguousarray(self._centroids)
        t0 = time.perf_counter()
        D, I = faiss.knn(C, C, 2)
        TIMES["delta_search"] += time.perf_counter() - t0
        D = np.ascontiguousarray(D, dtype=np.float32)
        I = np.ascontiguousarray(I, dtype=np.int64)
        alpha = float(_scheduled_alpha(self.recipe["alpha0"], self._step, self.recipe["total_steps"]))
        alpha = min(max(alpha, 1e-6), 1.0)
        prev = None
        if self.stat == "medg_fh":
            cache = getattr(self, "_check_cache", None)
            dp = None if cache is None else cache.get("delta")
            if dp is not None and len(dp) == k:
                prev = _f64(dp)
        rs = self._removal_stats
        assert rs.dtype == np.float32 and rs.flags.c_contiguous and rs.shape[0] == k and rs.shape[1] >= d
        gscale = float((2.0 / np.pi) * self._axis_scale)
        usage, stale, age, cum = _f64(self._usage), _f64(self._stale), _f64(self._age), _f64(self._cum_n)
        m_n, m_sum, m_sq, p2, p4 = (_f64(a) for a in (self._m_n, self._m_sum, self._m_sq, self._p2, self._p4))
        readout = np.ascontiguousarray(self._readout(), dtype=np.float32)
        rate, gain, delta, cost = (np.empty(k) for _ in range(4))
        rho = np.zeros(1)
        ncand = np.zeros(1, dtype=np.int64)
        src = np.empty(k, dtype=np.int64)
        kept = np.empty(k, dtype=np.int64)
        m = _lib.ngp_decide(k, d, usage.ctypes.data, rs.ctypes.data, rs.shape[1], gscale, stale.ctypes.data,
                            age.ctypes.data, cum.ctypes.data, float(self._decay_prod), alpha, float(self._step),
                            m_n.ctypes.data, m_sum.ctypes.data, m_sq.ctypes.data, p2.ctypes.data, p4.ctypes.data,
                            D.ctypes.data, I.ctypes.data, STAT[self.stat], None if prev is None else prev.ctypes.data,
                            float(self.w), readout.ctypes.data, int(bool(self.no_fallback_donor)), rate.ctypes.data,
                            gain.ctypes.data, delta.ctypes.data, cost.ctypes.data, rho.ctypes.data, ncand.ctypes.data,
                            src.ctypes.data, kept.ctypes.data)
        if m == -2:
            raise np.linalg.LinAlgError("SVD did not converge in Linear Least Squares (ngp_decide, medg_fh)")
        if m < 0:
            raise RuntimeError("ngp_decide: id out of range")
        self.n_fast = getattr(self, "n_fast", 0) + 1
        self._rho = float(rho[0])
        self._calib_pred = dict(cost=cost, gain=gain, delta=delta, unbiased=True, rate=rate)
        self._check_cache = dict(rate=rate, delta=delta)
        self.n_candidates += int(ncand[0])
        if m == 0:
            return None, None
        return src[:m].copy(), kept[:m].copy()
