"""The device decision's estimates in five fused Triton kernels (instead of about 150 small torch kernels per check).

Same formulas as netgain_device's torch functions (posterior_rates_device, gain_rel_var_device, posterior_gains_device,
margin_ratios_device and DeviceDecider.decide's costs and keys), in FP64; global sums are block partials (tl.sum over a
block of rows) added in a fixed order, so results are deterministic and equal the torch path to rounding.

  _est_a   per centre: rates' and margins' first sums, the gain variance's sums, counts       -> partials A
  _est_b   per centre: the centred second sums (rates' variance, margin ratios' variance)    -> partials B
  _est_c   per centre: posterior rate, the Paule-Mandel inputs y = log g, v = s^2           (then the PM kernel)
  _est_d   per centre: posterior gain, margin ratio, cost, donor key; clears the matching flags -> partials D (donors)
  _est_e   per centre: candidate key (cost below the best donor's gain)                         -> partials E (candidates)
Every program reduces the partials it needs itself (a few dozen values), so no separate reduction launches.
"""
import torch
import triton
import triton.language as tl

DMAX = torch.finfo(torch.float64).max / 4
NPA = 16          # partial slots of pass A


# FP64 constants, read from memory (a literal that fits FP32, such as 1e-6, would become an FP32 constant)
C_E6, C_E12, C_E300, C_E3, C_DMAX = 0, 1, 2, 3, 4
CONSTS = (1e-6, 1e-12, 1e-300, 1e-3, DMAX)


@triton.jit
def _mx(a, b):
    """max propagating NaN, as torch.maximum / clamp_min (tl.maximum's default returns the other operand)."""
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _mn(a, b):
    """min propagating NaN, as torch.minimum / clamp_max."""
    return tl.minimum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _isfinite(x):
    return (x == x) & (tl.abs(x) < float("inf"))


@triton.jit
def _red(part, j, G, NP: tl.constexpr, BLOCK_G: tl.constexpr):
    """Sum of slot j over the G block partials (fixed order)."""
    o = tl.arange(0, BLOCK_G)
    return tl.sum(tl.load(part + o * NP + j, mask=o < G, other=0.0), axis=0)


@triton.jit
def _scal_a(pa, G, cst, BLOCK_G: tl.constexpr):
    """Scalars from partials A: rates (m, mra, ns), gain variance (gvar), margins (rho, rmean, nok, sigma2 parts),
    positive gains count."""
    ns = _red(pa, 0, G, 16, BLOCK_G)
    s_r = _red(pa, 1, G, 16, BLOCK_G)
    s_mra = _red(pa, 2, G, 16, BLOCK_G)
    s_okg = _red(pa, 3, G, 16, BLOCK_G)
    s_kn = _red(pa, 4, G, 16, BLOCK_G)
    s_ng = _red(pa, 5, G, 16, BLOCK_G)
    nok = _red(pa, 6, G, 16, BLOCK_G)
    s_rn = _red(pa, 7, G, 16, BLOCK_G)
    s_nm = _red(pa, 8, G, 16, BLOCK_G)
    s_rm = _red(pa, 9, G, 16, BLOCK_G)
    s_w = _red(pa, 10, G, 16, BLOCK_G)
    s_dof = _red(pa, 11, G, 16, BLOCK_G)
    s_many = _red(pa, 12, G, 16, BLOCK_G)
    s_inv = _red(pa, 13, G, 16, BLOCK_G)
    npos = _red(pa, 14, G, 16, BLOCK_G)
    cnt_r = _mx(ns, 1.0)
    m = s_r / cnt_r
    mra = s_mra / cnt_r
    e6, e300 = tl.load(cst + 0), tl.load(cst + 2)
    kbar = s_kn / _mx(s_ng, e300)
    gvar = tl.where(s_okg > 0, _mx(kbar - 1.0, e6), 2.0)
    rho = s_rn / _mx(s_nm, e300)
    cnt_m = _mx(nok, 1.0)
    rmean = s_rm / cnt_m
    return ns, m, mra, cnt_r, gvar, nok, rho, cnt_m, rmean, s_w, s_dof, s_many, s_inv, npos


@triton.jit
def _scal_b(pb, G, m, mra, cnt_r, cnt_m, s_w, s_dof, s_many, s_inv, rho, cst, BLOCK_G: tl.constexpr):
    """Scalars from partials B: the rates' prior (a, b) and the margin ratios' sigma2, tau2."""
    e6, e12, e300 = tl.load(cst + 0), tl.load(cst + 1), tl.load(cst + 2)
    var = _red(pb, 0, G, 2, BLOCK_G) / cnt_r
    v = _mx(var - mra, e6 * m * m)
    a = m * m / v
    b = m / v
    rvar = _red(pb, 1, G, 2, BLOCK_G) / cnt_m
    sigma2 = tl.where(s_many > 0, _mx(s_w / _mx(s_dof, e300), e12), _mx(rvar, e12))
    noise_mean = sigma2 * s_inv / cnt_m
    tau2m = _mx(_mx(rvar - noise_mean, e6 * rho * rho), e12)
    return a, b, sigma2, tau2m


@triton.jit
def _rows(i, mk, age, cum, m_n, m_sum, delta, cst):
    ag = tl.load(age + i, mask=mk, other=0.0)
    cu = tl.load(cum + i, mask=mk, other=0.0)
    n = tl.load(m_n + i, mask=mk, other=0.0)
    ms = tl.load(m_sum + i, mask=mk, other=0.0)
    d0 = tl.load(delta + i, mask=mk, other=0.0)
    dd = tl.where(_isfinite(d0), d0, tl.load(cst + 4))                 # DMAX = float64 max / 4
    seen = mk & (ag >= 1.0)
    agc = _mx(ag, 1.0)
    r = tl.where(seen, cu / agc, 0.0)
    okm = mk & (n > 0.0) & _isfinite(dd) & (dd > 0.0)
    nsafe = tl.where(okm, n, 1.0)
    dl = tl.where(okm, dd, 1.0)
    rm = tl.where(okm, (ms / nsafe) / dl, 0.0)
    return ag, cu, n, ms, dd, seen, agc, r, okm, nsafe, dl, rm


@triton.jit
def _est_a(age, cum, m_n, m_sum, m_sq, p2, p4, raw, delta, decay_prod, pa, cst, k, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    mk = i < k
    ag, cu, n, ms, dd, seen, agc, r, okm, nsafe, dl, rm = _rows(i, mk, age, cum, m_n, m_sum, delta, cst)
    q2 = tl.load(p2 + i, mask=mk, other=0.0)
    q4 = tl.load(p4 + i, mask=mk, other=0.0)
    okg = mk & (n >= 2.0) & (q2 > 0.0)
    ng = tl.where(okg, n, 1.0)
    q2s = tl.where(okg, q2, 1.0)
    kap = tl.where(okg, (q4 / ng) / ((q2s / ng) * (q2s / ng)), 0.0)
    nmg = tl.where(okg, n, 0.0)
    msq = tl.load(m_sq + i, mask=mk, other=0.0)
    many = okm & (n >= 2.0)
    within = tl.where(many, (msq - ms * ms / nsafe) / (dl * dl), 0.0)
    g = tl.load(raw + i, mask=mk, other=0.0) / _mx(1.0 - tl.load(decay_prod), tl.load(cst + 1))
    base = pa + pid * 16
    one = tl.where(seen, 1.0, 0.0)
    tl.store(base + 0, tl.sum(one, axis=0))
    tl.store(base + 1, tl.sum(r, axis=0))
    tl.store(base + 2, tl.sum(tl.where(seen, r / agc, 0.0), axis=0))
    tl.store(base + 3, tl.sum(tl.where(okg, 1.0, 0.0), axis=0))
    tl.store(base + 4, tl.sum(kap * nmg, axis=0))
    tl.store(base + 5, tl.sum(nmg, axis=0))
    tl.store(base + 6, tl.sum(tl.where(okm, 1.0, 0.0), axis=0))
    tl.store(base + 7, tl.sum(rm * tl.where(okm, n, 0.0), axis=0))
    tl.store(base + 8, tl.sum(tl.where(okm, n, 0.0), axis=0))
    tl.store(base + 9, tl.sum(rm, axis=0))
    tl.store(base + 10, tl.sum(within, axis=0))
    tl.store(base + 11, tl.sum(tl.where(many, n - 1.0, 0.0), axis=0))
    tl.store(base + 12, tl.sum(tl.where(many, 1.0, 0.0), axis=0))
    tl.store(base + 13, tl.sum(tl.where(okm, 1.0 / nsafe, 0.0), axis=0))
    tl.store(base + 14, tl.sum(tl.where(mk & (g > 0.0), 1.0, 0.0), axis=0))
    tl.store(base + 15, 0.0)


@triton.jit
def _est_b(age, cum, m_n, m_sum, delta, pa, pb, cst, k, G, BLOCK: tl.constexpr, BLOCK_G: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    mk = i < k
    ns, m, mra, cnt_r, gvar, nok, rho, cnt_m, rmean, s_w, s_dof, s_many, s_inv, npos = _scal_a(pa, G, cst, BLOCK_G)
    ag, cu, n, ms, dd, seen, agc, r, okm, nsafe, dl, rm = _rows(i, mk, age, cum, m_n, m_sum, delta, cst)
    er = r - m
    em = rm - rmean
    tl.store(pb + pid * 2, tl.sum(tl.where(seen, er * er, 0.0), axis=0))
    tl.store(pb + pid * 2 + 1, tl.sum(tl.where(okm, em * em, 0.0), axis=0))


@triton.jit
def _est_c(age, cum, raw, decay_prod, alpha, step, pa, pb, rate_out, y_out, v_out, cst, k, G,
           BLOCK: tl.constexpr, BLOCK_G: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    mk = i < k
    ns, m, mra, cnt_r, gvar, nok, rho, cnt_m, rmean, s_w, s_dof, s_many, s_inv, npos = _scal_a(pa, G, cst, BLOCK_G)
    a, b, sigma2, tau2m = _scal_b(pb, G, m, mra, cnt_r, cnt_m, s_w, s_dof, s_many, s_inv, rho, cst, BLOCK_G)
    ag = tl.load(age + i, mask=mk, other=0.0)
    cu = tl.load(cum + i, mask=mk, other=0.0)
    full = (a + cu) / (b + ag)
    fallback = tl.where(ag > 0.0, cu / _mx(ag, 1.0), 0.0)
    rate = tl.where(ns < 2.0, fallback, tl.where(m > 0.0, full, 0.0))
    al = _mn(_mx(tl.load(alpha), tl.load(cst + 0)), 1.0)
    window = _mn((2.0 - al) / al, tl.load(step) + 1.0)
    s2 = gvar / _mx(rate * window, tl.load(cst + 3))
    g = tl.load(raw + i, mask=mk, other=0.0) / _mx(1.0 - tl.load(decay_prod), tl.load(cst + 1))
    pos = g > 0.0
    lg = tl.log(tl.where(pos, g, 1.0))
    tl.store(rate_out + i, rate, mask=mk)
    tl.store(y_out + i, tl.where(pos, lg, 0.0), mask=mk)
    tl.store(v_out + i, tl.where(pos, s2, float("inf")), mask=mk)


@triton.jit
def _est_d(age, cum, m_n, m_sum, delta, raw, decay_prod, stale, usage, rate, y, v, pm, pa, pb,
           gain_out, cost_out, dkey_out, kept, used, prot, pd, cst, k, G, BLOCK: tl.constexpr,
           BLOCK_G: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    mk = i < k
    ns, m, mra, cnt_r, gvar, nok, rho, cnt_m, rmean, s_w, s_dof, s_many, s_inv, npos = _scal_a(pa, G, cst, BLOCK_G)
    a, b, sigma2, tau2m = _scal_b(pb, G, m, mra, cnt_r, cnt_m, s_w, s_dof, s_many, s_inv, rho, cst, BLOCK_G)
    ag, cu, n, ms, dd, seen, agc, r, okm, nsafe, dl, rm = _rows(i, mk, age, cum, m_n, m_sum, delta, cst)
    # posterior gain
    g = tl.load(raw + i, mask=mk, other=0.0) / _mx(1.0 - tl.load(decay_prod), tl.load(cst + 1))
    pos = g > 0.0
    lg = tl.load(y + i, mask=mk, other=0.0)
    s2 = tl.load(v + i, mask=mk, other=1.0)
    mu = tl.load(pm)
    tau2 = _mx(tl.load(pm + 1), tl.load(cst + 0))
    post = tl.exp((tau2 * lg + s2 * mu) / (tau2 + s2))
    gain = tl.where(pos, tl.where(npos < 2.0, g, post), 0.0)
    # margin ratio and cost
    noise = sigma2 / nsafe
    r_hat = tl.where(okm, (tau2m * rm + noise * rho) / (tau2m + noise), rho)
    ratio = tl.where(nok == 0.0, 1.0, tl.where(nok < 2.0, _mx(rho, 0.0), _mx(r_hat, 0.0)))
    cost = tl.load(rate + i, mask=mk, other=0.0) * _mx(dd, 0.0) * ratio
    cooling = tl.load(stale + i, mask=mk, other=0.0) > 0.5
    isdonor = mk & (tl.load(usage + i, mask=mk, other=0.0) > 0.0) & ~cooling
    tl.store(gain_out + i, gain, mask=mk)
    tl.store(cost_out + i, cost, mask=mk)
    tl.store(dkey_out + i, tl.where(isdonor, -gain, float("inf")), mask=mk)
    zero = tl.zeros((BLOCK,), tl.int8)
    tl.store(kept + i, zero, mask=mk)
    tl.store(used + i, zero, mask=mk)
    tl.store(prot + i, zero, mask=mk)
    tl.store(pd + pid, tl.sum(tl.where(isdonor, 1, 0).to(tl.int64), axis=0))


@triton.jit
def _est_e(cost, gain, stale, donors, pd, ckey_out, pe, k, G, BLOCK: tl.constexpr, BLOCK_G: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    mk = i < k
    o = tl.arange(0, BLOCK_G)
    nd = tl.sum(tl.load(pd + o, mask=o < G, other=0), axis=0)
    g0 = tl.load(gain + tl.load(donors))
    c = tl.load(cost + i, mask=mk, other=0.0)
    cooling = tl.load(stale + i, mask=mk, other=0.0) > 0.5
    iscand = mk & (c < g0) & ~cooling & (nd > 0)
    tl.store(ckey_out + i, tl.where(iscand, c, float("inf")), mask=mk)
    tl.store(pe + pid, tl.sum(tl.where(iscand, 1, 0).to(tl.int64), axis=0))


class FusedEstimates:
    """Workspace (allocated once: graph-capture friendly) and launcher of the fused estimate kernels."""

    def __init__(self, k, device):
        self.k = k
        self.BLOCK = 256 if k <= 16384 else 1024
        self.G = triton.cdiv(k, self.BLOCK)
        self.BLOCK_G = max(triton.next_power_of_2(self.G), 2)
        f = lambda *s: torch.zeros(*s, dtype=torch.float64, device=device)
        self.pa, self.pb = f(self.G * NPA), f(self.G * 2)
        self.rate, self.y, self.v, self.gain, self.cost, self.dkey, self.ckey = (f(k) for _ in range(7))
        self.pm = f(2)
        self.pd = torch.zeros(self.G, dtype=torch.int64, device=device)
        self.pe = torch.zeros(self.G, dtype=torch.int64, device=device)
        self.cst = torch.tensor(CONSTS, dtype=torch.float64, device=device)

    def run(self, st, kept, used, prot, pm_fn):
        """Estimates, costs and the two sort keys; returns (cost, gain, donors, cand). pm_fn(y, v, out) writes
        (mu, tau2) to out."""
        k, B, G, BG = self.k, self.BLOCK, self.G, self.BLOCK_G
        # the kernels index k-vectors with unit stride: strided views (o-means passes columns of its (k, 5) table)
        # are copied first (found 2026-11-21: without this, o-means' decisions were wrong)
        st = {key: (v.contiguous() if torch.is_tensor(v) and v.dim() == 1 and not v.is_contiguous() else v)
              for key, v in st.items()}
        grid = (G,)
        _est_a[grid](st["age"], st["cum"], st["m_n"], st["m_sum"], st["m_sq"], st["p2"], st["p4"], st["raw"],
                     st["delta"], st["decay_prod"], self.pa, self.cst, k, BLOCK=B, num_warps=4)
        _est_b[grid](st["age"], st["cum"], st["m_n"], st["m_sum"], st["delta"], self.pa, self.pb, self.cst, k, G,
                     BLOCK=B, BLOCK_G=BG, num_warps=4)
        _est_c[grid](st["age"], st["cum"], st["raw"], st["decay_prod"], st["alpha"], st["step"], self.pa, self.pb,
                     self.rate, self.y, self.v, self.cst, k, G, BLOCK=B, BLOCK_G=BG, num_warps=4)
        pm_fn(self.y, self.v, self.pm)
        _est_d[grid](st["age"], st["cum"], st["m_n"], st["m_sum"], st["delta"], st["raw"], st["decay_prod"],
                     st["stale"], st["usage"], self.rate, self.y, self.v, self.pm, self.pa, self.pb, self.gain,
                     self.cost, self.dkey, kept, used, prot, self.pd, self.cst, k, G, BLOCK=B, BLOCK_G=BG,
                     num_warps=4)
        donors = torch.sort(self.dkey, stable=True).indices
        _est_e[grid](self.cost, self.gain, st["stale"], donors, self.pd, self.ckey, self.pe, k, G,
                     BLOCK=B, BLOCK_G=BG, num_warps=4)
        cand = torch.sort(self.ckey, stable=True).indices
        return self.cost, self.gain, donors, cand
