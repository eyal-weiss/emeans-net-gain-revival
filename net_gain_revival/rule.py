"""The net-gain revival decision as a pure function of per-centre statistics.

One call decides one revival check: which centres to free (targets) and which clusters to split (sources). Notation as
in the paper ("Revive What Pays", Sections 3-4 and Appendix A): rates lambda_j (points per batch), squared distance
delta_j to the nearest other centre, margin ratio r_j, removal cost C_j = lambda_j r_j delta_j, split gain G_s.

Every estimate is shrunk toward the population of centres according to its evidence:
  rates          Gamma-Poisson posterior mean, prior fitted by moments;
  margin ratios  normal-normal empirical Bayes toward the point-weighted mean ratio;
  split gains    log scale, noise (kappa - 1) / n_eff with a pooled kurtosis, prior by Paule-Mandel ("medg_pm") or by a
                 Fay-Herriot regression on log rate and log previous delta ("medg_fh"); the posterior MEDIAN is used.
Moves are matched greedily (cheapest removals with the largest gains, accepted while C_j < G_s) with the pair guard.

The arithmetic follows the research implementation operation by operation, so the decisions are identical to it.
"""
import numpy as np

STATS = ("medg_pm", "medg_fh")
DMAX = np.finfo("float64").max / 4          # distance of a centre with no other centre found


def posterior_rates(age, cum, k):
    """Gamma-Poisson posterior mean of each centre's points per batch, from cum points in age batches since birth."""
    seen = age >= 1
    if seen.sum() < 2:
        return np.where(age > 0, cum / np.maximum(age, 1.0), 0.0)
    r = cum[seen] / age[seen]
    m = float(r.mean())
    if not m > 0:
        return np.zeros(k)
    v = max(float(r.var()) - float(np.mean(r / age[seen])), 1e-6 * m * m)      # between-centre variance
    a, b = m * m / v, m / v
    return (a + cum) / (b + age)


def gain_rel_var(m_n, p2, p4):
    """Pooled kurtosis minus one of the projections on the split axes (2 if no cluster qualifies)."""
    ok = (m_n >= 2) & (p2 > 0)
    if not ok.any():
        return 2.0
    n = m_n[ok]
    kappa = (p4[ok] / n) / (p2[ok] / n) ** 2
    return max(float(np.average(kappa, weights=n)) - 1.0, 1e-6)


def margin_ratios(delta, m_n, m_sum, m_sq, k):
    """Shrunk ratios r_j = (mean margin) / delta_j, toward the point-weighted mean ratio."""
    ok = (m_n > 0) & np.isfinite(delta) & (delta > 0)
    r_hat = np.ones(k)
    if not ok.any():
        return r_hat
    n, dl = m_n[ok], delta[ok]
    r = (m_sum[ok] / n) / dl
    rho = float(np.average(r, weights=n))
    if ok.sum() < 2:
        return np.full(k, max(rho, 0.0))
    many = n >= 2
    if many.any():
        within = (m_sq[ok][many] - m_sum[ok][many] ** 2 / n[many]) / dl[many] ** 2
        sigma2 = max(float(within.sum()) / float((n[many] - 1).sum()), 1e-12)
    else:
        sigma2 = max(float(r.var()), 1e-12)
    tau2 = max(float(r.var()) - float(np.mean(sigma2 / n)), 1e-6 * rho * rho, 1e-12)
    noise = sigma2 / n
    r_hat[:] = rho
    r_hat[ok] = (tau2 * r + noise * rho) / (tau2 + noise)
    return np.maximum(r_hat, 0.0)


def paule_mandel(y, v, iters=100):
    """(mu, tau^2) of y = theta + noise(v), theta ~ N(mu, tau^2): precision-weighted mean and the Paule-Mandel root."""
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


def fay_herriot(y, v, X, iters=60):
    """(beta, tau^2) of y = X beta + eta + noise(v), eta ~ N(0, tau^2): WLS beta and the Paule-Mandel-type root."""
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


def posterior_gains(raw, rate, decay_prod, alpha, step, m_n, p2, p4, stat, prev_delta=None):
    """Posterior median of each centre's split gain (0 for a centre with no raw gain)."""
    k = len(raw)
    g = raw / max(1.0 - decay_prod, 1e-12)                       # start-up bias of the EMA
    alpha = min(max(float(alpha), 1e-6), 1.0)
    window = min((2.0 - alpha) / alpha, float(step + 1))         # memory of the EMA in batches (approximation)
    s2 = gain_rel_var(m_n, p2, p4) / np.maximum(rate * window, 1e-3)
    out = np.zeros(k)
    pos = g > 0
    if stat == "medg_fh":
        if pos.sum() < 4:
            out[pos] = g[pos]
            return out
        y, v = np.log(g[pos]), s2[pos]
        cols = [np.ones(pos.sum()), np.log(np.maximum(rate[pos], 1e-6))]
        if prev_delta is not None and len(prev_delta) == k:
            cols.append(np.log(np.maximum(np.asarray(prev_delta, dtype=np.float64)[pos], 1e-12)))
        X = np.column_stack(cols)
        beta, tau2 = fay_herriot(y, v, X)
        tau2 = max(tau2, 1e-6)
        out[pos] = np.exp((tau2 * y + v * (X @ beta)) / (tau2 + v))
        return out
    if pos.sum() < 2:
        out[pos] = g[pos]
        return out
    lg, s2p = np.log(g[pos]), s2[pos]
    mu, tau2 = paule_mandel(lg, s2p)
    tau2 = max(tau2, 1e-6)
    out[pos] = np.exp((tau2 * lg + s2p * mu) / (tau2 + s2p))
    return out


def match(cand, donors, cost, gain, nearest, readout):
    """Greedy matching with the pair guard; returns (sources, targets) or (None, None)."""
    kept, source, kept_ids, protected, used = [], [], set(), set(), set()
    pos = 0
    for j in cand.tolist():
        while pos < len(donors) and (int(donors[pos]) in kept_ids or int(donors[pos]) in used):
            pos += 1
        q = pos
        while q < len(donors) and (int(donors[q]) in kept_ids or int(donors[q]) in used or int(donors[q]) == j):
            q += 1
        if q == len(donors):
            if q == pos:
                break
            continue
        d = int(donors[q])
        if not cost[j] < gain[d]:
            if q == pos:
                break                          # costs rise, gains fall: no later candidate can pass
            continue                           # the better donor was j itself
        nb = int(nearest[j])
        back = nb in kept_ids and not (readout[j] == readout[nb]).all()
        if j in protected or back or j in used:
            continue
        kept.append(j); kept_ids.add(j); protected.add(nb)
        source.append(d); used.add(d)
    if not kept:
        return None, None
    return np.array(source, dtype="int64"), np.array(kept, dtype="int64")


def decide(raw, usage, stale, age, cum, decay_prod, alpha, step, m_n, m_sum, m_sq, p2, p4, D, I, readout,
           stat="medg_pm", prev_delta=None):
    """One revival check.

    raw: (2/pi) |U_s| per centre (computed by the caller as ((2/pi) * axis_scale) * |rows of U|); usage: EMA usage;
    stale: inherited coefficients h_j (cool-down at 1/2); age, cum: batches and points since birth; decay_prod:
    prod(1 - alpha) since the start; alpha, step: this batch's step size and zero-based index; m_n, m_sum, m_sq, p2, p4:
    margin and projection sums since birth; D, I: two-nearest search of the centres among themselves (k x 2);
    readout: the centres used by the pair guard's duplicate test; prev_delta: the previous check's delta (medg_fh).
    Returns (sources, targets, delta, info): the moves (or None, None), this check's delta, and the estimates.
    """
    assert stat in STATS
    k = len(raw)
    usage = np.asarray(usage, dtype=np.float64)
    cooling = stale > 0.5
    rate = posterior_rates(age, cum, k)
    gain = posterior_gains(raw, rate, decay_prod, alpha, step, m_n, p2, p4, stat, prev_delta)
    ids = np.arange(k)
    nn_other = np.where(I[:, 0] == ids, I[:, 1], I[:, 0])
    delta = np.where(I[:, 0] == ids, D[:, 1], D[:, 0]).astype("float64")
    missing = (nn_other < 0) | ~np.isfinite(delta)
    delta = np.where(missing, DMAX, delta)
    nn_other = np.where(nn_other < 0, ids, nn_other)
    cost = rate * np.maximum(delta, 0.0)
    cost = cost * margin_ratios(delta, m_n, m_sum, m_sq, k)
    info = dict(rate=rate, gain=gain, cost=cost, delta=delta, n_candidates=0)
    donors = np.flatnonzero((usage > 0) & ~cooling)
    if len(donors) == 0:
        return None, None, delta, info
    donors = donors[np.argsort(-gain[donors], kind="stable")]
    cand = np.flatnonzero((cost < gain[donors[0]]) & ~cooling)
    if len(cand) == 0:
        return None, None, delta, info
    cand = cand[np.argsort(cost[cand], kind="stable")]
    info["n_candidates"] = len(cand)
    src, tgt = match(cand, donors, cost, gain, nn_other, readout)
    return src, tgt, delta, info
