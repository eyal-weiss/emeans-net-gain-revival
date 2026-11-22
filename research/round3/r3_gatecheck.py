"""Third confirmatory round, gate G on one host (run by launch_r3.sh after G, before I).

Every gate run (interval_grid.py, medg_pm@5, seeds 30-31) on this host must equal the round-2 record of
ngp2_full_medg_pm at the same cell and seed (test and training error exactly; same host, same data hashes, expected
fields), and every cell of this host must have both gate runs. Exit 0 if so; otherwise exit 1 and the host stops (confirm3/PROTOCOL.md, Section 3).
usage: python r3_gatecheck.py --host H --gate DIR/r3_gate --round2 DIR/cfm2_grid
"""
import argparse, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import confirm2_grid as C2  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--host", required=True, choices=sorted(set(C2.HOST.values()))); p.add_argument("--gate", required=True); p.add_argument("--round2", required=True)
a = p.parse_args()
cells = [(ds, k) for ds, k in C2.CELLS34 if C2.HOST[ds] == a.host]
assert cells, "no cells on this host"
ok = bad = missing = 0
for ds, k in cells:
    for s in (30, 31):
        g = C2._run_file(a.gate, f"ivl|medg_pm@5|{ds}|k{k}|s{s}|random|T2")
        o = C2._run_file(a.round2, f"cfm2|ngp2_full_medg_pm|{ds}|k{k}|s{s}|random|T2")
        if not (os.path.exists(g) and os.path.exists(o)):
            missing += 1; print("MISSING", g if not os.path.exists(g) else o); continue
        rg, ro = json.load(open(g)), json.load(open(o))
        fields = (rg["config"], rg["dataset"], rg["k"], rg["seed"], rg["init"], rg["T"], rg["impl"])
        if fields != ("medg_pm@5", ds, k, s, "random", 2, "cpp") or rg["host"] != ro["host"] \
                or not rg["host"].startswith(a.host) or None in (rg.get("data_sha"), rg.get("data_test_sha")) \
                or (rg.get("data_sha"), rg.get("data_test_sha")) != (ro.get("data_sha"), ro.get("data_test_sha")) \
                or not all(math.isfinite(float(rg[m])) and float(rg[m]) > 0 for m in ("test_mse", "train_mse")):
            bad += 1; print("WRONG FIELDS", g, fields, rg["host"], ro["host"]); continue
        if rg["test_mse"] == ro["test_mse"] and rg["train_mse"] == ro["train_mse"]:
            ok += 1
        else:
            bad += 1; print("DIFFERENT", ds, k, s, rg["test_mse"], ro["test_mse"], rg["train_mse"], ro["train_mse"])
print(f"gate {a.host}: {ok} equal, {bad} different, {missing} missing (expected {2 * len(cells)} equal)")
sys.exit(0 if ok == 2 * len(cells) else 1)
