"""Equivalence and accuracy summary of runs on the PR head driver (fast_run.py records).
usage: python check_head.py FILE.jsonl [...]
Bit-identity pairs (final centroids' sha): the fast class trained eagerly against the head proof of concept (both
eager: exact), and the device decision against the host (C++) decision within the fast class (both replayed).
"""
import collections, json, math, sys

R = collections.defaultdict(dict)
for f in sys.argv[1:]:
    for line in open(f):
        if line.startswith("{"):
            r = json.loads(line)
            R[(r["dataset"], r["k"], r["seed"], r["T"])][r["config"]] = r

PAIRS = [("poch0", "fdev_e", "freeze, no early stop: PoC = fast (eager)"),
         ("pochn0", "fdevn_e", "no freeze, no early stop: PoC = fast (eager)"),
         ("poch", "fdevh_e", "freeze, early stop: PoC = fast (eager)"),
         ("pochn", "fdevnh_e", "no freeze, early stop: PoC = fast (eager)"),
         ("pochn0_x", "fdevn_x_e", "no freeze, no handover: PoC = fast (eager)"),
         ("fboth", "fdev", "freeze: host-applied (C++) = device decision (replay)"),
         ("fbothn", "fdevn", "no freeze: host-applied (C++) = device decision (replay)")]
for a, b, label in PAIRS:
    n = same = 0
    bad = []
    for key, v in sorted(R.items()):
        if a in v and b in v:
            n += 1
            if v[a]["sha"] == v[b]["sha"]:
                same += 1
            else:
                bad.append((key, v[a].get("test_mse"), v[b].get("test_mse"), v[a]["n_splits"], v[b]["n_splits"]))
    print(f"{label}: {same}/{n} identical")
    for x in bad:
        print("   differs:", x)
for c in ("fboth", "fbothn"):
    agree = sum(v[c]["check_agree"] for v in R.values() if c in v)
    total = sum(v[c]["check_total"] for v in R.values() if c in v)
    print(f"{c}: device decision = C++ decision in {agree}/{total} checks")
es = [(k, v[c]["es_stop_step"]) for k, v in R.items() for c in ("fdevh", "fdevnh") if c in v and v[c]["es_stop_step"]]
print(f"early stops among fdevh/fdevnh runs: {len(es)}")


def gm(v):
    return math.exp(sum(map(math.log, v)) / len(v)) if v else float("nan")


print("\nAccuracy and time (geometric means over runs; replayed runs)")
for T in sorted({k[3] for k in R}):
    keys = [k for k in R if k[3] == T and all(c in R[k] for c in ("pr", "fdevh", "fdevnh"))]
    if not keys:
        continue
    for m in ("test_mse", "train_mse"):
        print(f"T={T} {m}: no freeze / freeze {gm([R[k]['fdevnh'][m] / R[k]['fdevh'][m] for k in keys]):.4f}, "
              f"freeze / pr {gm([R[k]['fdevh'][m] / R[k]['pr'][m] for k in keys]):.4f}, "
              f"no freeze / pr {gm([R[k]['fdevnh'][m] / R[k]['pr'][m] for k in keys]):.4f}  ({len(keys)} runs)")
    print(f"T={T} seconds (single fits, not timing runs): fdevh / pr "
          f"{gm([R[k]['fdevh']['seconds'] / R[k]['pr']['seconds'] for k in keys]):.2f}, fdevnh / pr "
          f"{gm([R[k]['fdevnh']['seconds'] / R[k]['pr']['seconds'] for k in keys]):.2f}")
    for k in sorted(keys):
        v = R[k]
        print(f"   {k[0]:10s} {k[1]:6d} s{k[2]} test nf/f {v['fdevnh']['test_mse'] / v['fdevh']['test_mse']:.4f} "
              f"train nf/f {v['fdevnh']['train_mse'] / v['fdevh']['train_mse']:.4f}  f/pr test "
              f"{v['fdevh']['test_mse'] / v['pr']['test_mse']:.4f} train {v['fdevh']['train_mse'] / v['pr']['train_mse']:.4f}"
              f"  checks f/nf {v['fdevh']['n_checks']}/{v['fdevnh']['n_checks']}  handover {v['fdevnh']['handover_stats']}")
