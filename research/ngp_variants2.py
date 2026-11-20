"""EXPLORATORY variant with a covariate-dependent prior for the gain (2026-11-19 afternoon; not pre-registered).

Diagnosis (KDD 4096, diag_t5 out3): with one global prior, donors in KDD's sparse far tail (raw gains in the thousands,
under one point in the EMA's memory) are shrunk to ~0.2 and never split, while PR splits them by raw gain. Conversely, at
T = 5, centres without points were credited with the population's gain. Both point to the same modelling gap: the prior
mean of a centre's log gain should depend on well-measured facts about the centre, not be one global value.

stat = "medg_fh": Fay-Herriot empirical Bayes on the log gains, posterior median.
    log g_d = x_d' beta + u_d + e_d,  u_d ~ N(0, tau^2),  e_d ~ N(0, s2_d)  (s2_d as in the frozen rule)
    x_d = (1, log posterior rate_d, log delta_d), delta_d = squared distance to the nearest other centre at the previous
    check (the current one is computed after the gains; rate-only model at the first check). beta by weighted least
    squares with weights 1 / (tau^2 + s2_d); tau^2 by the Paule-Mandel equation sum w (y - X beta)^2 = m - p.
    Posterior median exp((tau^2 y + s2 x' beta) / (tau^2 + s2)).
Everything else is ngp2_full (via ngp_variants.AblationEmeans).
"""
import numpy as np

import ngp_variants as _V
from ngp_variants import TIMES  # noqa: F401
from faiss.contrib.e_means.base import _scheduled_alpha

CONFIGS = dict(_V.CONFIGS)
_FULL = dict(_V._base.CONFIGS["ngp2_full"][4])
CONFIGS["ngp2_full_medg_fh"] = _V._base.CONFIGS["ngp2_full"][:4] + (dict(_FULL, stat="medg_fh"),)
_base = _V._base


def fay_herriot(y, v, X, iters=60):
    """(beta, tau^2) of y = X beta + u + e, u ~ N(0, tau^2), e ~ N(0, v) (Paule-Mandel tau^2, WLS beta)."""
    m, p = X.shape

    def fit(t):
        w = 1.0 / (t + v)
        Xw = X * w[:, None]
        beta = np.linalg.lstsq(Xw.T @ X + 1e-12 * np.eye(p), Xw.T @ y, rcond=None)[0]
        r = y - X @ beta
        return float((w * r * r).sum()), beta

    q0, b0 = fit(0.0)
    if q0 <= m - p:
        return b0, 0.0
    lo, hi = -30.0, np.log(max(float(np.var(y)) * 4.0, 1e-12)) + 1.0
    while fit(np.exp(hi))[0] > m - p and hi < 60.0:
        hi += 2.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if fit(np.exp(mid))[0] > m - p:
            lo = mid
        else:
            hi = mid
    t = float(np.exp(0.5 * (lo + hi)))
    return fit(t)[1], t


class AblationEmeans(_V.AblationEmeans):
    def __init__(self, *a, stat="mean", **kw):
        fh = stat == "medg_fh"
        super().__init__(*a, stat="medg_pm" if fh else stat, **kw)
        self.fh = fh
        if fh:
            self.stat = "medg_fh"

    def _posterior_gains(self, raw, rate):
        if not self.fh:
            return super()._posterior_gains(raw, rate)
        g = raw / max(1.0 - self._decay_prod, 1e-12)
        alpha = float(_scheduled_alpha(self.recipe["alpha0"], self._step, self.recipe["total_steps"]))
        alpha = min(max(alpha, 1e-6), 1.0)
        window = min((2.0 - alpha) / alpha, float(self._step + 1))
        kv = self._gain_rel_var()
        s2 = kv / np.maximum(rate * window, 1e-3)
        out = np.zeros(self.k)
        pm = np.full(self.k, np.nan)
        pv = np.full(self.k, np.nan)
        pos = g > 0
        mu = tau2 = float("nan")
        beta = None
        if pos.sum() < 4:
            out[pos] = g[pos]
        else:
            y, v = np.log(g[pos]), s2[pos]
            cols = [np.ones(pos.sum()), np.log(np.maximum(rate[pos], 1e-6))]
            prev = getattr(self, "_check_cache", None)
            d_prev = None if prev is None else prev.get("delta")
            if d_prev is not None and len(d_prev) == self.k:
                cols.append(np.log(np.maximum(np.asarray(d_prev, dtype=np.float64)[pos], 1e-12)))
            X = np.column_stack(cols)
            beta, tau2 = fay_herriot(y, v, X)
            tau2 = max(tau2, 1e-6)
            pred = X @ beta
            pm[pos] = (tau2 * y + v * pred) / (tau2 + v)
            pv[pos] = tau2 * v / (tau2 + v)
            out[pos] = np.exp(pm[pos])
            mu = float(np.mean(pred))
        self._gd = dict(g=g, s2=s2, pm=pm, pv=pv, mu=mu, tau2=tau2, window=window, kv=kv, alpha=alpha,
                        beta=None if beta is None else [float(b) for b in beta])
        return out
