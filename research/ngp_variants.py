"""EXPLORATORY variants of the frozen rule ngp2_full (2026-11-19, after the confirmatory study; not pre-registered).

Diagnosis (diag_t5.py). Both failures of the frozen rule come from population quantities dominated by centres with
almost no evidence:
  * T = 5, KDD 1024: the posterior MEAN of the gain, exp(m + v/2) on the log scale, credits donors with no evidence
    (no point for several batches, effective sample size << 1) with the prior's mean exp(mu + tau^2/2); near-zero raw
    gains of decayed EMAs inflate tau^2 to 26-390, so empty centres become the top donors (runaway).
  * T = 2, KDD 256 seed 13: tau^2 is estimated by moments, var(log g) - mean(s2), and mean(s2) is dominated by centres with
    n_eff ~ 1 (s2 = (kappa - 1) / n_eff with a pooled kappa - 1 of 55-266): tau^2 hits its floor, every gain is shrunk to the
    same value, and the centre that absorbed an uncovered dense region is not recognised as the best donor.
Variants (stat=...):
  mean    : the frozen rule (identical computation)
  medg    : gain = posterior median exp(m) instead of the mean exp(m + v/2); prior by moments as in the frozen rule
  med     : medg, and the rate in the cost is the Gamma posterior median instead of the mean (ratio, delta unchanged)
  pm      : posterior mean, prior (mu, tau^2) by Paule-Mandel: precision-weighted mu with weights 1/(tau^2 + s2) and tau^2
            solving sum w (log g - mu)^2 = m - 1 (the standard random-effects estimator; low-evidence centres barely count)
  medg_pm : posterior median with the Paule-Mandel prior
The runtime-relevant parts are the frozen C++ ones (ablation_cpp) unless NGP_IMPL=py. self._gd keeps the last check's
gain estimates (diagnostics).
"""
import os

import numpy as np

if os.environ.get("NGP_IMPL", "py") == "cpp":
    import ablation_cpp as _base
else:
    import ablation as _base
from ablation import TIMES  # noqa: F401
from faiss.contrib.e_means.base import _scheduled_alpha

STATS = ("mean", "medg", "med", "pm", "medg_pm")
CONFIGS = dict(_base.CONFIGS)
_FULL = dict(_base.CONFIGS["ngp2_full"][4])
for _s in STATS[1:]:
    CONFIGS[f"ngp2_full_{_s}"] = _base.CONFIGS["ngp2_full"][:4] + (dict(_FULL, stat=_s),)


def paule_mandel(y, v, iters=100):
    """(mu, tau^2): random-effects prior of estimates y with known noise variances v (Paule and Mandel 1982).
    tau^2 = 0 if the generalized Q at tau^2 = 0 is already <= m - 1; else bisection on log tau^2 (Q is decreasing)."""
    m = len(y)

    def q(t):
        w = 1.0 / (t + v)
        mu = float((w * y).sum() / w.sum())
        return float((w * (y - mu) ** 2).sum()), mu

    q0, mu0 = q(0.0)
    if q0 <= m - 1:
        return mu0, 0.0
    lo, hi = -30.0, np.log(max(float(np.var(y)) * 4.0, 1e-12)) + 1.0
    while q(np.exp(hi))[0] > m - 1 and hi < 60.0:
        hi += 2.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if q(np.exp(mid))[0] > m - 1:
            lo = mid
        else:
            hi = mid
    t = float(np.exp(0.5 * (lo + hi)))
    return q(t)[1], t


class AblationEmeans(_base.AblationEmeans):
    def __init__(self, *a, stat="mean", **kw):
        assert stat in STATS, stat
        super().__init__(*a, **kw)
        self.stat = stat
        self._gd = None

    def _posterior_gains(self, raw, rate):
        out = super()._posterior_gains(raw, rate)
        # the parent's estimate (normal gain_eb, fixed-n noise), recomputed for the diagnostics and the variants
        assert getattr(self, "gain_eb", "normal") == "normal" and getattr(self, "gain_noise", "fixed_n") == "fixed_n"
        g = raw / max(1.0 - self._decay_prod, 1e-12)
        alpha = float(_scheduled_alpha(self.recipe["alpha0"], self._step, self.recipe["total_steps"]))
        alpha = min(max(alpha, 1e-6), 1.0)
        window = min((2.0 - alpha) / alpha, float(self._step + 1))
        kv = self._gain_rel_var()
        s2 = kv / np.maximum(rate * window, 1e-3)
        pm = np.full(self.k, np.nan)
        pv = np.full(self.k, np.nan)
        mu = tau2 = float("nan")
        pos = g > 0
        if pos.sum() >= 2:
            lg, s2p = np.log(g[pos]), s2[pos]
            if self.stat in ("pm", "medg_pm"):
                mu, tau2 = paule_mandel(lg, s2p)
                tau2 = max(tau2, 1e-6)
            else:
                mu = float(lg.mean())
                tau2 = max(float(lg.var()) - float(s2p.mean()), 1e-6)
            pm[pos] = (tau2 * lg + s2p * mu) / (tau2 + s2p)
            pv[pos] = tau2 * s2p / (tau2 + s2p)
            if self.stat != "mean":
                out = np.zeros(self.k)
                if self.stat in ("medg", "med", "medg_pm"):
                    out[pos] = np.exp(pm[pos])
                else:
                    out[pos] = np.exp(pm[pos] + pv[pos] / 2.0)
        self._gd = dict(g=g, s2=s2, pm=pm, pv=pv, mu=mu, tau2=tau2, window=window, kv=kv, alpha=alpha)
        return out

    def _margin_ratios(self, delta):
        r_hat, rho = super()._margin_ratios(delta)
        if self.stat != "med":
            return r_hat, rho
        from scipy.special import gammaincinv
        age, cum = self._age, self._cum_n
        seen = age >= 1
        if seen.sum() >= 2:                              # the parent's Gamma prior (_posterior_rates)
            r = cum[seen] / age[seen]
            m = float(r.mean())
            if m > 0:
                v = max(float(r.var()) - float(np.mean(r / age[seen])), 1e-6 * m * m)
                a, b = m * m / v, m / v
                mean = (a + cum) / (b + age)
                med = gammaincinv(a + cum, 0.5) / (b + age)
                r_hat = r_hat * np.where(mean > 0, med / np.where(mean > 0, mean, 1.0), 1.0)
        return r_hat, rho
