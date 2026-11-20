"""ctypes bindings of the optional C++ kernels (libnetgain.so, built with build.sh). available() says whether they load."""
import ctypes
import os

import numpy as np

_LIB = None
_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libnetgain.so")


def available():
    global _LIB
    if _LIB is None and os.path.exists(_PATH):
        lib = ctypes.CDLL(_PATH)
        P, I, D = ctypes.c_void_p, ctypes.c_int64, ctypes.c_double
        lib.ngp_margin_stats.restype = ctypes.c_int
        lib.ngp_margin_stats.argtypes = [I, I, I] + [P] * 12 + [ctypes.c_int]
        lib.ngp_decide.restype = I
        lib.ngp_decide.argtypes = ([I, I, P, P, I, D, P, P, P, D, D, D] + [P] * 7 +
                                   [ctypes.c_int, P, D, P, ctypes.c_int] + [P] * 8)
        _LIB = lib
    return _LIB is not None


def _p(a, dtype):
    assert a.dtype == dtype and a.flags.c_contiguous
    return a.ctypes.data


def margin_stats(x, D, I, C, A, m_n, m_sum, m_sq, p2, p4, threads):
    """Labels (ties to the lowest index) and nearest distances; adds the per-cluster sums in place."""
    n, d = x.shape
    labels = np.empty(n, dtype=np.int64)
    dist = np.empty(n, dtype=np.float32)
    rc = _LIB.ngp_margin_stats(n, d, len(C), _p(x, np.float32), _p(D, np.float32), _p(I, np.int64),
                               _p(C, np.float32), _p(A, np.float32), _p(labels, np.int64), _p(dist, np.float32),
                               _p(m_n, np.float64), _p(m_sum, np.float64), _p(m_sq, np.float64), _p(p2, np.float64),
                               _p(p4, np.float64), threads)
    if rc != 0:
        raise RuntimeError("ngp_margin_stats: label out of range")
    return dist, labels


def decide(usage, removal_stats, gscale, stale, age, cum, decay_prod, alpha, step, m_n, m_sum, m_sq, p2, p4, D, I,
           stat, prev_delta, readout):
    """The whole check in C++ (same arguments as rule.decide, with the raw gain computed inside from removal_stats)."""
    k, d = readout.shape
    f64 = lambda a: np.ascontiguousarray(a, dtype=np.float64)
    usage, stale, age, cum = f64(usage), f64(stale), f64(age), f64(cum)
    m_n, m_sum, m_sq, p2, p4 = (f64(a) for a in (m_n, m_sum, m_sq, p2, p4))
    prev = None if prev_delta is None or len(prev_delta) != k else f64(prev_delta)
    rate, gain, delta, cost = (np.empty(k) for _ in range(4))
    rho, ncand = np.zeros(1), np.zeros(1, dtype=np.int64)
    src, tgt = np.empty(k, dtype=np.int64), np.empty(k, dtype=np.int64)
    m = _LIB.ngp_decide(k, d, usage.ctypes.data, _p(removal_stats, np.float32), removal_stats.shape[1], float(gscale),
                        stale.ctypes.data, age.ctypes.data, cum.ctypes.data, float(decay_prod), float(alpha),
                        float(step), m_n.ctypes.data, m_sum.ctypes.data, m_sq.ctypes.data, p2.ctypes.data,
                        p4.ctypes.data, _p(D, np.float32), _p(I, np.int64), {"medg_pm": 1, "medg_fh": 2}[stat],
                        None if prev is None else prev.ctypes.data, 1.0, _p(readout, np.float32), 0, rate.ctypes.data,
                        gain.ctypes.data, delta.ctypes.data, cost.ctypes.data, rho.ctypes.data, ncand.ctypes.data,
                        src.ctypes.data, tgt.ctypes.data)
    if m == -2:
        raise np.linalg.LinAlgError("SVD did not converge in Linear Least Squares")
    if m < 0:
        raise RuntimeError("ngp_decide: id out of range")
    info = dict(rate=rate, gain=gain, cost=cost, delta=delta, n_candidates=int(ncand[0]))
    if m == 0:
        return None, None, delta, info
    return src[:m].copy(), tgt[:m].copy(), delta, info
