"""Paule-Mandel kernel: the default (Newton's root, FP32 pre-solve) against the replayed bisection and NumPy's
reference (net_gain_revival.rule.paule_mandel); time per call. usage: python test_pm.py RULE_PY"""
import sys, time
from importlib import util
import numpy as np, torch
import netgain_device as ND
spec = util.spec_from_file_location("rule", sys.argv[1]); rule = util.module_from_spec(spec); spec.loader.exec_module(rule)
rng = np.random.default_rng(0)
dev = torch.device("cuda")


def run(y, v, pos, **kw):
    yy = torch.tensor(np.where(pos, y, 0.0), device=dev).contiguous()
    vv = torch.tensor(np.where(pos, v, np.inf), device=dev).contiguous()
    out = torch.empty(2, dtype=torch.float64, device=dev)
    ND.paule_mandel_into(yy, vv, out, **kw)
    return out.tolist(), (yy, vv, out)


def timed(args, **kw):
    yy, vv, out = args
    best = 1e9
    for _ in range(30):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        ND.paule_mandel_into(yy, vv, out, **kw)
        torch.cuda.synchronize(); best = min(best, time.perf_counter() - t0)
    return 1e3 * best


def rd(a, b):
    return 0.0 if a == b else abs(a - b) / max(abs(a), abs(b), 1e-300)


worst_by = {}
worst = {"new_vs_replay_mu": 0, "new_vs_replay_t": 0, "new_vs_numpy_mu": 0, "new_vs_numpy_t": 0,
         "replay_vs_numpy_t": 0}
cases = 0
for k in (2, 3, 7, 64, 256, 1024, 4096, 16384, 20000):
    for kind in ("normal", "wide", "tight_v", "huge_v", "equal_y", "few_pos", "heavy"):
        for rep in range(4):
            y = rng.normal(-3, 1.5, k); v = rng.gamma(2.0, 0.3, k) * rng.uniform(0.05, 3)
            pos = rng.random(k) < 0.9
            if kind == "wide": y = rng.normal(0, 8, k)
            if kind == "tight_v": v = v * 1e-8
            if kind == "huge_v": v = v * 1e6
            if kind == "equal_y": y = np.full(k, -2.5)
            if kind == "few_pos": pos = rng.random(k) < 3.0 / k
            if kind == "heavy": y = rng.standard_t(2, k) * 2
            if pos.sum() < 2: continue
            (mn, tn), args = run(y, v, pos)
            (mr, tr), _ = run(y, v, pos, replay=True, pre32=False)
            mu_ref, t_ref = rule.paule_mandel(y[pos], v[pos])
            cases += 1
            worst["new_vs_replay_mu"] = max(worst["new_vs_replay_mu"], rd(mn, mr))
            worst["new_vs_replay_t"] = max(worst["new_vs_replay_t"], rd(tn, tr))
            worst["new_vs_numpy_mu"] = max(worst["new_vs_numpy_mu"], rd(mn, mu_ref))
            worst["new_vs_numpy_t"] = max(worst["new_vs_numpy_t"], rd(tn, t_ref))
            worst["replay_vs_numpy_t"] = max(worst["replay_vs_numpy_t"], rd(tr, t_ref))
            wb = worst_by.setdefault(k, [0.0, 0.0])
            wb[0] = max(wb[0], rd(tn, t_ref)); wb[1] = max(wb[1], rd(mn, mu_ref))
            if rd(tn, t_ref) > 1e-10 or rd(mn, mu_ref) > 1e-10:
                print(f"LARGE k={k} {kind}: tau2 new {tn:.17g} replay {tr:.17g} numpy {t_ref:.17g}; mu {mn} {mr} {mu_ref}")
    y = rng.normal(-3, 1.5, k); v = rng.gamma(2.0, 0.3, k); pos = rng.random(k) < 0.9
    _, args = run(y, v, pos)
    print(f"k={k:6d}: new {timed(args):.3f} ms, old (replay, FP64 Newton) {timed(args, replay=True, pre32=False):.3f} ms",
          flush=True)
print("worst new vs numpy by k (tau2, mu):", {a: (f"{b[0]:.1e}", f"{b[1]:.1e}") for a, b in worst_by.items()})
print(cases, "cases; worst relative differences:", {a: f"{b:.1e}" for a, b in worst.items()})
