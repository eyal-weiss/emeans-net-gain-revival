"""The net-gain revival decision (stat "medg_pm") entirely on the device: no host reads, fixed shapes, so a whole check
can be captured as a CUDA graph.

decide_device(st) follows net_gain_revival.rule.decide operation by operation:
  rates          Gamma-Poisson posterior means (masked FP64 torch operations);
  gains          posterior medians on the log scale with the Paule-Mandel prior: the root search (bracket expansion
                 by +2 up to 60, then 100 bisections, as the reference) runs in one single-program Triton kernel;
  margin ratios  normal-normal empirical Bayes (masked torch operations);
  matching       the reference's greedy loop with the pair guard, in one single-program Triton kernel (sequential by
                 nature: cheapest candidate first, best remaining donor, accepted while C_j < G_s).
Sums are taken in a different order than NumPy's pairwise sums, so estimates agree with the reference to rounding;
a decision differs only when a comparison sits within rounding of its threshold (tests count how often).

st: dict of device tensors (k = number of centres): usage, raw ((2/pi)|U|), stale, age, cum, m_n, m_sum, m_sq, p2, p4
(FP64, k); nn (int64, k) and delta (FP64, k) from the nearest-other search; dup (int8, k): centre equal to its
nearest; decay_prod, alpha, step (FP64 0-d); stopped (int32 0-d, early stop: no move when set).
Returns (src, tgt, m, n_cand): int64 (k,) donors and freed centres, valid in [0, m); m and n_cand int64 0-d tensors.
"""
import os

import torch
import triton
import triton.language as tl

DMAX = torch.finfo(torch.float64).max / 4


# ---------------------------------------------------------------------- Paule-Mandel root (one program)
@triton.jit
def _pm_eval_mem(y, v, k, t, BLOCK: tl.constexpr):
    """(Q(t), dQ/dt, mu(t)) streaming y, v from memory (w = 1 / (t + v); masked entries carry v = inf, w = 0).
    dQ/dt = -sum w^2 (y - mu)^2: mu minimizes the weighted sum, so its derivative drops out."""
    sw = tl.zeros((BLOCK,), tl.float64)
    swy = tl.zeros((BLOCK,), tl.float64)
    for b in range(0, k, BLOCK):
        i = b + tl.arange(0, BLOCK)
        mk = i < k
        w = 1.0 / (t + tl.load(v + i, mask=mk, other=float("inf")))
        sw += w
        swy += w * tl.load(y + i, mask=mk, other=0.0)
    mu = tl.sum(swy, axis=0) / tl.sum(sw, axis=0)
    q = tl.zeros((BLOCK,), tl.float64)
    dq = tl.zeros((BLOCK,), tl.float64)
    for b in range(0, k, BLOCK):
        i = b + tl.arange(0, BLOCK)
        mk = i < k
        w = 1.0 / (t + tl.load(v + i, mask=mk, other=float("inf")))
        e = tl.load(y + i, mask=mk, other=0.0) - mu
        we2 = tl.where(w > 0, w * e * e, 0.0)
        q += we2
        dq += w * we2
    return tl.sum(q, axis=0), -tl.sum(dq, axis=0), mu


@triton.jit
def _pm_eval(y, v, k, t, yv, vv, REG: tl.constexpr, BLOCK: tl.constexpr):
    """As _pm_eval_mem; with REG the whole y, v sit in registers (yv, vv) and are not reloaded."""
    if REG:
        w = 1.0 / (t + vv)
        mu = tl.sum(w * yv, axis=0) / tl.sum(w, axis=0)
        e = yv - mu
        we2 = tl.where(w > 0, w * e * e, 0.0)
        return tl.sum(we2, axis=0), -tl.sum(w * we2, axis=0), mu
    else:
        return _pm_eval_mem(y, v, k, t, BLOCK)


@triton.jit
def _pm_eval32(y32, v32, t32):
    """FP32 (Q(t), dQ/dt) from register-resident FP32 copies (the FP32 pre-solve only: it locates the root for the
    FP64 Newton steps and decides nothing itself)."""
    w = 1.0 / (t32 + v32)
    mu = tl.sum(w * y32, axis=0) / tl.sum(w, axis=0)
    e = y32 - mu
    we2 = tl.where(w > 0, w * e * e, 0.0)
    return tl.sum(we2, axis=0), -tl.sum(w * we2, axis=0)


@triton.jit
def _pm_kernel(y, v, k, out, cst, MAXIT: tl.constexpr, ITERS: tl.constexpr, BISECT: tl.constexpr, REG: tl.constexpr,
               BLOCK: tl.constexpr, REPLAY: tl.constexpr, PRE32: tl.constexpr):
    """The reference's Paule-Mandel estimate (net_gain_revival.rule.paule_mandel), exactly: Q(0) <= m - 1 gives
    tau2 = 0; otherwise the bracket [-30, hi] in log t (hi from 4 var(y), raised by 2 while Q(e^hi) > m - 1 and
    hi < 60) and the result of the reference's 100 bisections in it. The bisection path is replayed: safeguarded
    Newton steps find the root x first; a midpoint then goes up iff it lies below x, decided arithmetically outside a
    band where rounding of Q could decide (band from the rounding error of Q, about 1e-15 m, over |dQ/dx| at x), and
    by evaluating Q as the reference does inside it (repeated midpoints, once the bracket is down to an ulp, are
    evaluated once). BISECT evaluates Q at every midpoint (tests).
    2026-11-21: REPLAY=False (default) skips the replay: the result is Newton's root x itself, which agrees with the
    replayed bisection to the rounding of Q (inside the band the replay's comparisons are rounding-decided anyway, by
    sums in a different order than the reference's); about 17 of the ~30 FP64 evaluations of Q are saved. PRE32 (with
    REG) first runs Newton in FP32 (its own bracket; it decides nothing), so the FP64 Newton steps start next to the
    root (about 3 instead of 8 FP64 evaluations). cst: FP64 constants (cst[1] = 1e-12)."""
    if REG:
        i0 = tl.arange(0, BLOCK)
        yv = tl.load(y + i0, mask=i0 < k, other=0.0)
        vv = tl.load(v + i0, mask=i0 < k, other=float("inf"))
        live = vv < float("inf")
        m = tl.sum(tl.where(live, 1.0, 0.0), axis=0)
        mean = tl.sum(tl.where(live, yv, 0.0), axis=0) / m
        e0 = yv - mean
        var = tl.sum(tl.where(live, e0 * e0, 0.0), axis=0) / m
    else:
        yv = tl.zeros((BLOCK,), tl.float64)
        vv = yv
        cs = tl.zeros((BLOCK,), tl.float64)
        ss = tl.zeros((BLOCK,), tl.float64)
        for b in range(0, k, BLOCK):
            i = b + tl.arange(0, BLOCK)
            mk = i < k
            lv = mk & (tl.load(v + i, mask=mk, other=float("inf")) < float("inf"))
            cs += tl.where(lv, 1.0, 0.0)
            ss += tl.where(lv, tl.load(y + i, mask=mk, other=0.0), 0.0)
        m = tl.sum(cs, axis=0)
        mean = tl.sum(ss, axis=0) / m
        vs = tl.zeros((BLOCK,), tl.float64)
        for b in range(0, k, BLOCK):
            i = b + tl.arange(0, BLOCK)
            mk = i < k
            lv = mk & (tl.load(v + i, mask=mk, other=float("inf")) < float("inf"))
            e = tl.load(y + i, mask=mk, other=0.0) - mean
            vs += tl.where(lv, e * e, 0.0)
        var = tl.sum(vs, axis=0) / m
    target = m - 1.0
    q0, _d0, mu0 = _pm_eval(y, v, k, 0.0, yv, vv, REG, BLOCK)
    mu_out = mu0
    t_out = 0.0 * mu0
    if q0 > target:
        lo = -30.0 + 0.0 * mu0
        hi = tl.log(tl.maximum(var * 4.0, tl.load(cst + 1))) + 1.0
        qh, _d, _u = _pm_eval(y, v, k, tl.exp(hi), yv, vv, REG, BLOCK)
        while (qh > target) & (hi < 60.0):
            hi += 2.0
            qh, _d, _u = _pm_eval(y, v, k, tl.exp(hi), yv, vv, REG, BLOCK)
        ql, _d, _u = _pm_eval(y, v, k, tl.exp(lo), yv, vv, REG, BLOCK)
        hi0 = hi                                    # the reference's bracket, kept for the replay below
        slope = 0.0 * mu0
        if ql <= target:
            x = lo                                  # root below the bracket: the bisection ends at lo
        elif qh > target:
            x = hi                                  # root above it: the bisection ends at hi
        else:
            x = 0.5 * (lo + hi)
            if PRE32:
                y32 = yv.to(tl.float32)
                v32 = vv.to(tl.float32)
                lo3 = lo
                hi3 = hi
                x3 = x
                done3 = x != x
                for _it in range(40):
                    if done3 == 0:
                        tq3, dq3 = _pm_eval32(y32, v32, tl.exp(x3.to(tl.float32)))
                        f3 = tq3.to(tl.float64) - target
                        up3 = f3 > 0
                        lo3 = tl.where(up3, x3, lo3)
                        hi3 = tl.where(up3, hi3, x3)
                        df3 = tl.exp(x3) * dq3.to(tl.float64)
                        nw3 = x3 - f3 / df3
                        ok3 = (df3 < 0) & (nw3 > lo3) & (nw3 < hi3)
                        xn3 = tl.where(f3 == 0.0, x3, tl.where(ok3, nw3, 0.5 * (lo3 + hi3)))
                        tol3 = 1e-6 * tl.maximum(tl.abs(x3), 1.0)
                        done3 = (tl.abs(xn3 - x3) <= tol3) | (hi3 - lo3 <= tol3) | (f3 == 0.0)
                        x3 = xn3
                x = tl.where((x3 > lo) & (x3 < hi), x3, x)  # FP64 Newton (safeguarded, own bracket) from here
            done = x != x
            for _it in range(MAXIT):
                if done == 0:
                    tq, dq, _u = _pm_eval(y, v, k, tl.exp(x), yv, vv, REG, BLOCK)
                    f = tq - target
                    up = f > 0
                    lo = tl.where(up, x, lo)
                    hi = tl.where(up, hi, x)
                    df = tl.exp(x) * dq
                    slope = df
                    newton = x - f / df
                    ok = (df < 0) & (newton > lo) & (newton < hi)
                    xn = tl.where(f == 0.0, x, tl.where(ok, newton, 0.5 * (lo + hi)))   # f = 0: x is the root
                    tol = 4e-16 * tl.maximum(tl.abs(x), 1.0)
                    done = (tl.abs(xn - x) <= tol) | (hi - lo <= tol) | (f == 0.0)
                    x = xn
        # band where Q's rounding (about 1e-15 m, here 1e-12 m for safety) could decide a comparison
        band = tl.where(slope < 0, 1e-12 * m / tl.abs(slope), 1e-6 * tl.maximum(tl.abs(x), 1.0))
        band = tl.maximum(band, 1e-14 * tl.maximum(tl.abs(x), 1.0))
        lo_b = -30.0 + 0.0 * mu0
        hi_b = hi0
        if REPLAY == 0:
            lo_b = x
            hi_b = x
        c1 = 0.0 * mu0 + float("nan")               # cache of the last two evaluated midpoints
        c2 = c1
        u1 = c1 != c1
        u2 = u1
        for _it in range(ITERS if REPLAY else 0):
            mid = 0.5 * (lo_b + hi_b)
            near = (tl.abs(mid - x) <= band) | BISECT
            up = mid < x
            if near:
                if mid == c1:
                    up = u1
                elif mid == c2:
                    up = u2
                else:
                    qm, _d, _u = _pm_eval(y, v, k, tl.exp(mid), yv, vv, REG, BLOCK)
                    up = qm > target
                    c2 = c1
                    u2 = u1
                    c1 = mid
                    u1 = up
            lo_b = tl.where(up, mid, lo_b)
            hi_b = tl.where(up, hi_b, mid)
        t_out = tl.exp(0.5 * (lo_b + hi_b))
        _q, _d, mu_out = _pm_eval(y, v, k, t_out, yv, vv, REG, BLOCK)
    tl.store(out, mu_out)
    tl.store(out + 1, t_out)


_PM_CST = {}


def paule_mandel_into(y, v, out, iters=100, replay=None, pre32=None):
    """(mu, tau2) into out[0], out[1] (FP64); y, v contiguous FP64, excluded entries carry v = inf.
    replay (default: NG_PM_REPLAY=1, else off) replays the reference's bisection; pre32 (default on unless
    NG_PM_PRE32=0; register-resident sizes only) runs Newton in FP32 first."""
    dev = y.device
    if dev not in _PM_CST:
        _PM_CST[dev] = torch.tensor((1e-6, 1e-12), dtype=torch.float64, device=dev)
    bisect = os.environ.get("NG_PM_BISECT") == "1"
    replay = (os.environ.get("NG_PM_REPLAY") == "1" or bisect) if replay is None else bool(replay)
    pre32 = os.environ.get("NG_PM_PRE32", "1") == "1" if pre32 is None else bool(pre32)
    with torch.cuda.device(dev):
        k = len(y)
        reg = k <= 16384                                  # y, v held in registers (measured faster up to 16384)
        block = triton.next_power_of_2(k) if reg else 1024
        warps = 4 if block <= 1024 else 8 if block <= 4096 else 32
        # at least one element per thread: with fewer (replicated layouts), the FP32 pre-solve was miscompiled
        # (wrong roots and means for k <= 7, found 2026-11-21 by test_pm.py; padding fixes it)
        block = max(block, 32 * warps) if reg else block
        _pm_kernel[(1,)](y, v, k, out, _PM_CST[dev], MAXIT=80, ITERS=iters, BISECT=bisect, REG=reg, BLOCK=block,
                         REPLAY=replay, PRE32=pre32 and reg, num_warps=warps)
    return out


def paule_mandel_device(lg, s2, pos, iters=100):
    """(mu, tau2) as 0-d FP64 tensors; entries with pos False are excluded."""
    y = torch.where(pos, lg, torch.zeros_like(lg)).contiguous()
    v = torch.where(pos, s2, torch.full_like(s2, float("inf"))).contiguous()
    out = torch.empty(2, dtype=torch.float64, device=lg.device)
    paule_mandel_into(y, v, out, iters)
    return out[0], out[1]


# ---------------------------------------------------------------------- greedy matching (one program)
@triton.jit
def _match_kernel(cand, nc_ptr, donors, nd_ptr, cost, gain, nearest, dup, kept, used, prot, src_out, tgt_out,
                  m_out, stopped_ptr, G, nc_out, PARTS: tl.constexpr, BLOCK_G: tl.constexpr):
    if PARTS:                                       # counts given as G block partials (fused estimates)
        o = tl.arange(0, BLOCK_G)
        nc = tl.sum(tl.load(nc_ptr + o, mask=o < G, other=0), axis=0)
        nd = tl.sum(tl.load(nd_ptr + o, mask=o < G, other=0), axis=0)
        tl.store(nc_out, nc)
    else:
        nc = tl.load(nc_ptr)
        nd = tl.load(nd_ptr)
    go = tl.load(stopped_ptr) == 0
    m = nc * 0
    pos = nc * 0
    i = nc * 0
    while go & (i < nc):
        j = tl.load(cand + i)
        # first donor that is neither a freed centre nor already used
        adv = pos < nd
        while adv:
            dp = tl.load(donors + pos)
            bad = (tl.load(kept + dp) != 0) | (tl.load(used + dp) != 0)
            pos = tl.where(bad, pos + 1, pos)
            adv = bad & (pos < nd)
        q = pos
        adv = q < nd
        while adv:
            dq = tl.load(donors + q)
            bad = (tl.load(kept + dq) != 0) | (tl.load(used + dq) != 0) | (dq == j)
            q = tl.where(bad, q + 1, q)
            adv = bad & (q < nd)
        if q >= nd:
            go = q != pos                                      # q == pos: no donor left (break), else continue
        else:
            d = tl.load(donors + q)
            if tl.load(cost + j) < tl.load(gain + d):
                nb = tl.load(nearest + j)
                back = (tl.load(kept + nb) != 0) & (tl.load(dup + j) == 0)
                skip = (tl.load(prot + j) != 0) | back | (tl.load(used + j) != 0)
                if skip == 0:
                    tl.store(kept + j, 1)
                    tl.store(prot + nb, 1)
                    tl.store(used + d, 1)
                    tl.store(src_out + m, d)
                    tl.store(tgt_out + m, j)
                    m += 1
                    tl.debug_barrier()
            else:
                go = q != pos                                  # costs rise, gains fall: break at the best donor
        i += 1
    tl.store(m_out, m)


# ---------------------------------------------------------------------- estimates (torch, FP64, masked)
def _safe(x, ok, fill=1.0):
    return torch.where(ok, x, torch.full_like(x, fill))


def posterior_rates_device(age, cum):
    seen = age >= 1
    ns = seen.sum()
    cnt = ns.double().clamp_min(1.0)
    r = torch.where(seen, cum / age.clamp_min(1.0), torch.zeros_like(cum))
    m = r.sum() / cnt
    var = torch.where(seen, (r - m) ** 2, torch.zeros_like(r)).sum() / cnt
    mra = torch.where(seen, r / age.clamp_min(1.0), torch.zeros_like(r)).sum() / cnt
    v = torch.maximum(var - mra, 1e-6 * m * m)
    a, b = m * m / v, m / v
    full = (a + cum) / (b + age)
    fallback = torch.where(age > 0, cum / age.clamp_min(1.0), torch.zeros_like(cum))
    return torch.where(ns < 2, fallback, torch.where(m > 0, full, torch.zeros_like(full)))


def gain_rel_var_device(m_n, p2, p4):
    ok = (m_n >= 2) & (p2 > 0)
    n = torch.where(ok, m_n, torch.zeros_like(m_n))
    nn_ = _safe(m_n, ok)
    kappa = torch.where(ok, (p4 / nn_) / (_safe(p2, ok) / nn_) ** 2, torch.zeros_like(m_n))
    kbar = (kappa * n).sum() / n.sum().clamp_min(1e-300)
    return torch.where(ok.any(), torch.clamp_min(kbar - 1.0, 1e-6), torch.full_like(kbar, 2.0))


def posterior_gains_device(raw, rate, decay_prod, alpha, step, m_n, p2, p4):
    g = raw / torch.clamp_min(1.0 - decay_prod, 1e-12)
    alpha = alpha.clamp(1e-6, 1.0)
    window = torch.minimum((2.0 - alpha) / alpha, step + 1.0)
    s2 = gain_rel_var_device(m_n, p2, p4) / torch.clamp_min(rate * window, 1e-3)
    pos = g > 0
    lg = torch.log(_safe(g, pos))
    mu, tau2 = paule_mandel_device(lg, s2, pos)
    tau2 = torch.clamp_min(tau2, 1e-6)
    post = torch.exp((tau2 * lg + s2 * mu) / (tau2 + s2))
    few = pos.sum() < 2
    return torch.where(pos, torch.where(few, g, post), torch.zeros_like(g))


def margin_ratios_device(delta, m_n, m_sum, m_sq):
    ok = (m_n > 0) & torch.isfinite(delta) & (delta > 0)
    nok = ok.sum()
    n = torch.where(ok, m_n, torch.zeros_like(m_n))
    nsafe, dl = _safe(m_n, ok), _safe(delta, ok)
    r = torch.where(ok, (m_sum / nsafe) / dl, torch.zeros_like(m_n))
    rho = (r * n).sum() / n.sum().clamp_min(1e-300)
    cnt = nok.double().clamp_min(1.0)
    rmean = r.sum() / cnt
    rvar = torch.where(ok, (r - rmean) ** 2, torch.zeros_like(r)).sum() / cnt
    many = ok & (m_n >= 2)
    within = torch.where(many, (m_sq - m_sum ** 2 / nsafe) / dl ** 2, torch.zeros_like(m_n))
    dof = torch.where(many, m_n - 1.0, torch.zeros_like(m_n)).sum()
    sigma2 = torch.where(many.any(), torch.clamp_min(within.sum() / dof.clamp_min(1e-300), 1e-12),
                         torch.clamp_min(rvar, 1e-12))
    noise_mean = torch.where(ok, sigma2 / nsafe, torch.zeros_like(m_n)).sum() / cnt
    tau2 = torch.maximum(torch.maximum(rvar - noise_mean, 1e-6 * rho * rho), torch.full_like(rho, 1e-12))
    noise = sigma2 / nsafe
    r_hat = torch.where(ok, (tau2 * r + noise * rho) / (tau2 + noise), rho.expand_as(r))
    out = torch.where(nok < 2, torch.clamp_min(rho, 0.0).expand_as(r), torch.clamp_min(r_hat, 0.0))
    return torch.where(nok == 0, torch.ones_like(out), out)


class DeviceDecider:
    """Workspace for decide_device (flags and outputs allocated once; graph-capture friendly)."""

    def __init__(self, k, device, fused=None):
        self.k = k
        self.kept = torch.zeros(k, dtype=torch.int8, device=device)
        self.used = torch.zeros(k, dtype=torch.int8, device=device)
        self.prot = torch.zeros(k, dtype=torch.int8, device=device)
        self.src = torch.zeros(k, dtype=torch.int64, device=device)
        self.tgt = torch.zeros(k, dtype=torch.int64, device=device)
        self.m = torch.zeros((), dtype=torch.int64, device=device)
        # fused estimates (netgain_fused: five kernels) unless NG_FUSED=0 (the torch operations below, for tests)
        self.fused = (os.environ.get("NG_FUSED", "1") == "1") if fused is None else bool(fused)
        if self.fused:
            import netgain_fused as NF
            self._fe = NF.FusedEstimates(k, device)
            self.nc = torch.zeros((), dtype=torch.int64, device=device)

    def decide(self, st):
        if self.fused:
            fe = self._fe
            cost, gain, donors, cand = fe.run(st, self.kept, self.used, self.prot, paule_mandel_into)
            with torch.cuda.device(cost.device):
                _match_kernel[(1,)](cand, fe.pe, donors, fe.pd, cost, gain, st["nn"], st["dup"], self.kept,
                                    self.used, self.prot, self.src, self.tgt, self.m, st["stopped"], fe.G, self.nc,
                                    PARTS=True, BLOCK_G=fe.BLOCK_G, num_warps=1)
            return self.src, self.tgt, self.m, self.nc
        k = self.k
        rate = posterior_rates_device(st["age"], st["cum"])
        gain = posterior_gains_device(st["raw"], rate, st["decay_prod"], st["alpha"], st["step"], st["m_n"],
                                      st["p2"], st["p4"])
        delta = torch.where(torch.isfinite(st["delta"]), st["delta"], torch.full_like(st["delta"], DMAX))
        cost = rate * torch.clamp_min(delta, 0.0) * margin_ratios_device(delta, st["m_n"], st["m_sum"], st["m_sq"])
        cooling = st["stale"] > 0.5
        isdonor = (st["usage"] > 0) & ~cooling
        donors = torch.sort(torch.where(isdonor, -gain, torch.full_like(gain, float("inf"))), stable=True).indices
        nd = isdonor.sum()
        g0 = gain.index_select(0, donors[:1]).squeeze(0)            # no host read (graph capture)
        iscand = (cost < g0) & ~cooling & (nd > 0)
        cand = torch.sort(torch.where(iscand, cost, torch.full_like(cost, float("inf"))), stable=True).indices
        nc = iscand.sum()
        self.kept.zero_()
        self.used.zero_()
        self.prot.zero_()
        with torch.cuda.device(cost.device):
            _match_kernel[(1,)](cand, nc, donors, nd, cost, gain, st["nn"], st["dup"], self.kept, self.used,
                                self.prot, self.src, self.tgt, self.m, st["stopped"], 1, self.m, PARTS=False,
                                BLOCK_G=2, num_warps=1)
        return self.src, self.tgt, self.m, nc
