"""The fused estimates (netgain_fused) against the torch path (netgain_device, NG_FUSED=0): decisions identical,
rates / gains / costs equal to rounding, on random states and edge cases. usage: python test_fused.py"""
import math
import sys

import torch

import netgain_device as ND
import netgain_fused as NF

dev = torch.device("cuda")
DMAX = ND.DMAX


def torch_parts(st):
    rate = ND.posterior_rates_device(st["age"], st["cum"])
    gain = ND.posterior_gains_device(st["raw"], rate, st["decay_prod"], st["alpha"], st["step"], st["m_n"], st["p2"],
                                     st["p4"])
    delta = torch.where(torch.isfinite(st["delta"]), st["delta"], torch.full_like(st["delta"], DMAX))
    cost = rate * torch.clamp_min(delta, 0.0) * ND.margin_ratios_device(delta, st["m_n"], st["m_sum"], st["m_sq"])
    return rate, gain, cost


def rel(a, b):
    a, b = a.double(), b.double()
    both_nan = torch.isnan(a) & torch.isnan(b)
    same = (a == b) | both_nan
    d = ((a - b).abs() / torch.maximum(a.abs(), b.abs()).clamp_min(1e-300))
    d = torch.where(same, torch.zeros_like(d), d)
    return float(d.max()) if len(d) else 0.0


def make(k, g, kind="random"):
    u = lambda *s: torch.rand(*s, generator=g, device=dev, dtype=torch.float64)
    m_n = torch.floor(u(k) * 200)
    m_sum = m_n * u(k)
    s = lambda v: torch.tensor(v, dtype=torch.float64, device=dev)
    st = dict(usage=torch.floor(u(k) * 1000), raw=u(k) * 3, stale=(u(k) < 0.05).double(), age=torch.floor(u(k) * 300),
              cum=torch.floor(u(k) * 3e4), m_n=m_n, m_sum=m_sum, m_sq=m_sum * m_sum / m_n.clamp_min(1) * (1 + u(k)),
              p2=m_n * u(k), p4=m_n * u(k) * 3, delta=u(k) * 2, decay_prod=s(float(u(1)) * 0.9),
              alpha=s(0.002 + 0.05 * float(u(1))), step=s(float(torch.floor(u(1) * 2000))),
              stopped=torch.zeros((), dtype=torch.int32, device=dev),
              nn=torch.randint(0, k, (k,), generator=g, device=dev), dup=torch.zeros(k, dtype=torch.int8, device=dev))
    z, o = torch.zeros(k, dtype=torch.float64, device=dev), torch.ones(k, dtype=torch.float64, device=dev)
    if kind == "unseen":
        st["age"] = z.clone()
    elif kind == "one_seen":
        st["age"] = z.clone(); st["age"][0] = 5.0
    elif kind == "no_margins":
        st["m_n"] = z.clone(); st["m_sum"] = z.clone(); st["m_sq"] = z.clone()
    elif kind == "single_rows":
        st["m_n"] = torch.where(st["m_n"] > 0, o, z); st["m_sum"] = st["m_n"] * 0.3; st["m_sq"] = st["m_sum"] ** 2
    elif kind == "no_p2":
        st["p2"] = z.clone()
    elif kind == "no_gain":
        st["raw"] = z.clone()
    elif kind == "one_gain":
        st["raw"] = z.clone(); st["raw"][k // 2] = 1.5
    elif kind == "all_cooling":
        st["stale"] = o.clone()
    elif kind == "no_donors":
        st["usage"] = z.clone()
    elif kind == "inf_delta":
        st["delta"][::3] = float("inf"); st["delta"][1::7] = float("nan")
    elif kind == "all_inf_delta":
        st["delta"] = torch.full_like(z, float("inf"))
    elif kind == "zero_delta":
        st["delta"][::2] = 0.0
    elif kind == "decay_one":
        st["decay_prod"] = s(1.0)
    elif kind == "alpha_zero":
        st["alpha"] = s(0.0)
    elif kind == "stopped":
        st["stopped"] = torch.ones((), dtype=torch.int32, device=dev)
    elif kind == "cheap":                       # many candidates: costs well below gains
        st["delta"] = st["delta"] * 1e-4
    elif kind == "strided":                     # o-means passes columns of a (k, 5) table
        tab = torch.stack([st[c] for c in ("m_n", "m_sum", "m_sq", "p2", "p4")], dim=1).contiguous()
        for j, c in enumerate(("m_n", "m_sum", "m_sq", "p2", "p4")):
            st[c] = tab[:, j]
    return st


g = torch.Generator(device=dev).manual_seed(1)
kinds = ["random", "unseen", "one_seen", "no_margins", "single_rows", "no_p2", "no_gain", "one_gain", "all_cooling",
         "no_donors", "inf_delta", "all_inf_delta", "zero_delta", "decay_one", "alpha_zero", "stopped", "cheap",
         "strided"]
fails, worst, n = 0, 0.0, 0
for k in (2, 3, 5, 17, 256, 1000, 4096, 16384, 20000):
    for kind in kinds:
        for rep in range(3 if kind in ("random", "cheap", "strided") else 1):
            st = make(k, g, kind)
            dt = ND.DeviceDecider(k, dev, fused=False)
            df = ND.DeviceDecider(k, dev, fused=True)
            s1, t1, m1, c1 = dt.decide(st)
            s2, t2, m2, c2 = df.decide(st)
            m1, m2, c1, c2 = int(m1), int(m2), int(c1), int(c2)
            same = m1 == m2 and c1 == c2 and torch.equal(s1[:m1], s2[:m2]) and torch.equal(t1[:m1], t2[:m2])
            rate, gain, cost = torch_parts(st)
            e = max(rel(rate, df._fe.rate), rel(gain, df._fe.gain), rel(cost, df._fe.cost))
            worst = max(worst, e)
            n += 1
            if not same or e > 1e-12:
                fails += 1
                print(f"FAIL k={k} {kind} rep={rep}: moves torch {m1} fused {m2}, cand {c1}/{c2}, same={same}, "
                      f"max rel diff {e:.2e}", flush=True)
# review round 3: finite inputs whose gain variance is 0/0 (p2 underflows when squared); torch propagates the NaN
s = lambda v: torch.tensor(v, dtype=torch.float64, device=dev)
codex = dict(age=s([10.0, 10.0]), cum=s([2.0, 2.0]), usage=s([1.0, 1.0]), raw=s([1.0, 1.0]), delta=s([1.0, 1.0]),
             stale=s([0.0, 0.0]), m_sum=s([0.0, 0.0]), m_sq=s([0.0, 0.0]), m_n=s([2.0, 2.0]), p2=s([2.0**-559, 2.0]),
             p4=s([0.0, 2.0]), nn=torch.tensor([1, 0], device=dev), dup=torch.zeros(2, dtype=torch.int8, device=dev),
             decay_prod=s(0.0), alpha=s(0.1), step=s(10.0), stopped=torch.zeros((), dtype=torch.int32, device=dev))
cases = [("review round 3 state", codex)]
for k in (64, 4096):
    st = make(k, g, "random")
    st["m_n"][0], st["p2"][0], st["p4"][0] = 2.0, 2.0**-559, 0.0
    cases.append((f"p2 underflow k={k}", st))
    st = make(k, g, "random")
    st["delta"][::5] = float("nan")
    cases.append((f"NaN delta k={k}", st))
for name, st in cases:
    k = len(st["age"])
    r1 = ND.DeviceDecider(k, dev, fused=False).decide(st)
    r2 = ND.DeviceDecider(k, dev, fused=True).decide(st)
    m1, m2, c1, c2 = int(r1[2]), int(r2[2]), int(r1[3]), int(r2[3])
    same = m1 == m2 and c1 == c2 and torch.equal(r1[0][:m1], r2[0][:m2]) and torch.equal(r1[1][:m1], r2[1][:m2])
    n += 1
    fails += 0 if same else 1
    print(f"{name}: moves torch {m1} fused {m2}, candidates {c1}/{c2} -> {'same' if same else 'DIFFERENT'}")
print(f"{n} states, {fails} failures, worst relative difference of rate/gain/cost {worst:.2e}")
sys.exit(1 if fails else 0)
