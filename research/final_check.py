"""Checks of the final comparison against round 3 (run on each host, where both record sets live):
  R  : ours without the early stop = round 3's ngh79440bc_medg_pm_{freeze,nofreeze}_guard (test and training error)
  pr : the PR's runs of F2/F5 = round 3's pr_head79440bc at the same keys
usage: python final_check.py FINAL_DIR ROUND3_DIR"""
import json, os, sys

fin, r3 = sys.argv[1], sys.argv[2]
MAP = {"rel79440bc_freeze_none": "ngh79440bc_medg_pm_freeze_guard",
       "rel79440bc_nofreeze_none": "ngh79440bc_medg_pm_nofreeze_guard", "pr_head79440bc": "pr_head79440bc"}
n = bad = missing = 0
for f in sorted(os.listdir(os.path.join(fin, "runs"))):
    if not f.endswith(".json"):
        continue
    r = json.load(open(os.path.join(fin, "runs", f)))
    if r["config"] not in MAP:
        continue
    g = os.path.join(r3, "runs", f"prh__{MAP[r['config']]}__{r['dataset']}__k{r['k']}__s{r['seed']}__random__T{r['T']}.json")
    if not os.path.exists(g):
        missing += 1
        continue
    q = json.load(open(g))
    n += 1
    same = (r["test_mse"], r["train_mse"]) == (q["test_mse"], q["train_mse"]) and r["n_splits"] == q["n_splits"]
    if not same:
        bad += 1
        print("DIFF", r["key"], r["test_mse"], q["test_mse"], r["train_mse"], q["train_mse"], r["n_splits"], q["n_splits"])
print(f"compared {n}, identical {n - bad}, different {bad}, no round-3 record {missing}")
sys.exit(1 if bad or missing else 0)
