"""The net-gain revival decision of ngp2_full and its variants medg_pm / medg_fh, as one pure NumPy function of the
per-centre state (2026-11-19 evening; proof of concept for other implementations: GPU, o-means).

It transcribes the code paths that the CPU classes use for these configs (ablation.py: ng_post2, matcher v2,
cooldown "half", knn "exact", margin_stride 1, moderate False, gain_noise "fixed_n", gain_eb "normal", w = 1;
ablation_cpp.py::_choose_net_gain; ngp_variants.py stat "medg_pm"; ngp_variants2.py stat "medg_fh"), so that
decide(state) returns exactly the moves the CPU class makes at a check (netgain_rule_check.py verifies this).
paule_mandel and fay_herriot are verbatim copies (this module must import without the PR's CPU classes).

state (dict; arrays of length k unless noted):
  k, step, alpha0, total_steps      ints / float (the run's recipe and the current batch index)
  alpha                             optional: this step's EMA step (else from the PR's schedule)
  usage                             EMA masses N_j (float64)
  unorm                             |U_j| (the PR's gain statistic, already scaled; raw gain = (2/pi) unorm)
  centroids, readout                (k, d) float32: positions for the delta search, and for the pair guard
  m_n, m_sum, m_sq, p2, p4          sums since birth over the rows assigned while moves are allowed: count,
                                    margin d2 - d1, its square, projection p on the split axis squared, p^4
  cum_n, age                        points and batches since birth
  stale                             inherited weight since the last split (cool-down: stale > 0.5)
  decay_prod                        prod(1 - alpha_t) since the start
  prev_delta                        (medg_fh) the previous check's delta, or None
Returns (source, target, delta): donor and freed-centre ids (or None, None) and this check's delta (for prev_delta).
"""
import numpy as np

STATS = ("mean", "medg_pm", "medg_fh")


def scheduled_alpha(alpha0, step, total_steps):
    from faiss.contrib.e_means.base import _scheduled_alpha
    return float(_scheduled_alpha(alpha0, step, total_steps))


def posterior_rates(age, cum, k):
    seen = age >= 1
    if seen.sum() < 2:
        return np.where(age > 0, cum / np.maximum(age, 1.0), 0.0)
    r = cum[seen] / age[seen]
    m = float(r.mean())
    if not m > 0:
        return np.zeros(k)
    v_lam = float(r.var()) - float(np.mean(r / age[seen]))
    v_lam = max(v_lam, 1e-6 * m * m)
    a, b = m * m / v_lam, m / v_lam
    return (a + cum) / (b + age)


def gain_rel_var(m_n, p2, p4):
    ok = (m_n >= 2) & (p2 > 0)
    if not ok.any():
        return 2.0
    n = m_n[ok]
    kappa = (p4[ok] / n) / (p2[ok] / n) ** 2
    kbar = float(np.average(kappa, weights=n))
    return max(kbar - 1.0, 1e-6)


def margin_ratios(delta, m_n, m_sum, m_sq, k):
    ok = (m_n > 0) & np.isfinite(delta) & (delta > 0)
    r_hat = np.ones(k)
    if not ok.any():
        return r_hat, 1.0
    n, dl = m_n[ok], delta[ok]
    r = (m_sum[ok] / n) / dl
    rho = float(np.average(r, weights=n))
    if ok.sum() < 2:
        return np.full(k, max(rho, 0.0)), rho
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
    return np.maximum(r_hat, 0.0), rho


def paule_mandel(y, v, iters=100):
    """Verbatim copy of ngp_variants.paule_mandel."""
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
    """Verbatim copy of ngp_variants2.fay_herriot."""
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


def posterior_gains(raw, rate, st, stat):
    g = raw / max(1.0 - st["decay_prod"], 1e-12)
    alpha = st.get("alpha")                       # given by implementations with their own schedule (o-means)
    if alpha is None:
        alpha = scheduled_alpha(st["alpha0"], st["step"], st["total_steps"])
    alpha = min(max(float(alpha), 1e-6), 1.0)
    window = min((2.0 - alpha) / alpha, float(st["step"] + 1))
    kv = gain_rel_var(st["m_n"], st["p2"], st["p4"])
    s2 = kv / np.maximum(rate * window, 1e-3)
    k = len(raw)
    out = np.zeros(k)
    pos = g > 0
    if stat == "medg_fh":
        if pos.sum() < 4:
            out[pos] = g[pos]
            return out
        y, v = np.log(g[pos]), s2[pos]
        cols = [np.ones(pos.sum()), np.log(np.maximum(rate[pos], 1e-6))]
        prev = st.get("prev_delta")
        if prev is not None and len(prev) == k:
            cols.append(np.log(np.maximum(np.asarray(prev, dtype=np.float64)[pos], 1e-12)))
        X = np.column_stack(cols)
        beta, tau2 = fay_herriot(y, v, X)
        tau2 = max(tau2, 1e-6)
        out[pos] = np.exp((tau2 * y + v * (X @ beta)) / (tau2 + v))
        return out
    if pos.sum() < 2:
        out[pos] = g[pos]
        return out
    lg, s2p = np.log(g[pos]), s2[pos]
    if stat == "medg_pm":
        mu, tau2 = paule_mandel(lg, s2p)
        tau2 = max(tau2, 1e-6)
    else:
        mu = float(lg.mean())
        tau2 = max(float(lg.var()) - float(s2p.mean()), 1e-6)
    pm = (tau2 * lg + s2p * mu) / (tau2 + s2p)
    pv = tau2 * s2p / (tau2 + s2p)
    out[pos] = np.exp(pm) if stat == "medg_pm" else np.exp(pm + pv / 2.0)
    return out


def match_v2(cand, donors, cost, gain, nearest, centroids, w=1.0, guard=True):
    """ablation.AblationEmeans._match_v2 with forbid_nearest=False (the confirmatory configs). guard=False drops the
    pair guard (a freed centre protects its nearest centre; no freed centre's nearest may be freed unless it is an exact
    duplicate), as PR #5915 did for its own rule on 2026-11-19 (commit 79440bc)."""
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
        if not cost[j] < w * gain[d]:
            if q == pos:
                break
            continue
        nb = int(nearest[j])
        if guard:
            back = nb in kept_ids and not (centroids[j] == centroids[nb]).all()
            if j in protected or back:
                continue
        if j in used:
            continue
        kept.append(j); kept_ids.add(j)
        if guard:
            protected.add(nb)
        source.append(d); used.add(d)
    if not kept:
        return None, None
    return np.array(source, dtype="int64"), np.array(kept, dtype="int64")


def nearest_two(C):
    """(delta, nearest other) by an exact two-nearest search, as the CPU classes (faiss.knn, FP32)."""
    import faiss
    D, I = faiss.knn(np.ascontiguousarray(C, dtype=np.float32), np.ascontiguousarray(C, dtype=np.float32), 2)
    return D, I


def decide(st, stat="medg_pm", w=1.0, knn=nearest_two, guard=True):
    assert stat in STATS
    k = st["k"]
    usage = np.asarray(st["usage"], dtype=np.float64)
    raw = (2.0 / np.pi) * np.asarray(st["unorm"], dtype=np.float64)
    cooling = st["stale"] > 0.5
    rate = posterior_rates(st["age"], st["cum_n"], k)
    gain = posterior_gains(raw, rate, st, stat)
    D, I = knn(st["centroids"])
    ids = np.arange(k)
    nn_other = np.where(I[:, 0] == ids, I[:, 1], I[:, 0])
    delta = np.where(I[:, 0] == ids, D[:, 1], D[:, 0]).astype("float64")
    missing = (nn_other < 0) | ~np.isfinite(delta)
    delta = np.where(missing, np.finfo("float64").max / 4, delta)
    nn_other = np.where(nn_other < 0, ids, nn_other)
    cost = rate * np.maximum(delta, 0.0)
    ratio, _ = margin_ratios(delta, st["m_n"], st["m_sum"], st["m_sq"], k)
    cost = cost * ratio
    donors = np.flatnonzero((usage > 0) & ~cooling)
    if len(donors) == 0:
        return None, None, delta
    donors = donors[np.argsort(-gain[donors], kind="stable")]
    cand = np.flatnonzero((cost < w * gain[donors[0]]) & ~cooling)
    if len(cand) == 0:
        return None, None, delta
    cand = cand[np.argsort(cost[cand], kind="stable")]
    src, tgt = match_v2(cand, donors, cost, gain, nn_other, st["readout"], w, guard)
    return src, tgt, delta
