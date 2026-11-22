"""Analysis of the final comparison (docs/protocols/final_comparison_plan.md): checks against round 3, then estimates.

usage: python final_analyze.py --final DIR [DIR ...] --round3 DIR [DIR ...]
  DIR: directories holding runs/<key>.json (one per host) of final_grid.py and of round 3's prhead_grid.py (part F).
Prints Markdown. Exits with code 1 if a check fails (then no estimate is printed).
"""
import argparse
import collections
import glob
import json
import math
import os
import statistics as st
import sys

import numpy as np
from scipy import stats as sps

R3 = {"rel79440bc_freeze_none": "ngh79440bc_medg_pm_freeze_guard",
      "rel79440bc_nofreeze_none": "ngh79440bc_medg_pm_nofreeze_guard", "pr_head79440bc": "pr_head79440bc",
      "rel79440bc_freeze_hl": "ngh79440bc_medg_pm_freeze_guard",
      "rel79440bc_nofreeze_hl": "ngh79440bc_medg_pm_nofreeze_guard"}
PARTS = {"R": (2, range(200, 202), ["rel79440bc_freeze_none", "rel79440bc_nofreeze_none"]),
         "F2": (2, range(200, 260), ["pr_head79440bc", "rel79440bc_freeze_hl", "rel79440bc_nofreeze_hl"]),
         "F5": (5, range(200, 240), ["pr_head79440bc", "rel79440bc_freeze_hl", "rel79440bc_nofreeze_hl"])}
NAMES = {"pr_head79440bc": "the PR's rule", "rel79440bc_freeze_hl": "ours, freeze", "rel79440bc_nofreeze_hl": "ours, no freeze",
         "ngh79440bc_medg_pm_freeze_guard": "round 3: ours, freeze, no early stop",
         "ngh79440bc_medg_pm_nofreeze_guard": "round 3: ours, no freeze, no early stop"}


def load(dirs):
    out = {}
    for d in dirs:
        for f in glob.glob(os.path.join(d, "runs", "*.json")):
            r = json.load(open(f))
            out[(r["config"], r["dataset"], r["k"], r["seed"], r["T"])] = r
    return out


def est(A, a, B, b, T, seeds, cells, metric):
    """exp(mean over seeds of L_s) - 1 with a 95% t interval; L_s = mean over cells of log(metric_a / metric_b)."""
    L = np.array([np.mean([math.log(A[(a, ds, k, s, T)][metric] / B[(b, ds, k, s, T)][metric]) for ds, k in cells])
                  for s in seeds])
    m, se = L.mean(), L.std(ddof=1) / math.sqrt(len(L))
    h = sps.t.ppf(0.975, len(L) - 1) * se
    p = sps.wilcoxon(L).pvalue if np.any(L != 0) else 1.0
    f = lambda x: 100 * (math.exp(x) - 1)
    return f"{f(m):+.2f}% [{f(m - h):+.2f}%, {f(m + h):+.2f}%]", int((L < 0).sum()), len(L), p


def time_ratio(A, a, B, b, T, seeds, cells):
    return st.median(A[(a, ds, k, s, T)]["seconds"] / B[(b, ds, k, s, T)]["seconds"] for s in seeds for ds, k in cells)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--final", nargs="+", required=True)
    p.add_argument("--round3", nargs="+", required=True)
    a = p.parse_args()
    F, R = load(a.final), load(a.round3)
    cells = sorted({(r["dataset"], r["k"]) for r in F.values()})
    print("# Final comparison on the released implementation: analysis\n")
    print(f"{len(F)} runs; {len(cells)} cells; round-3 records loaded: {len(R)}\n")
    fails = []
    # completeness
    for part, (T, seeds, cfgs) in PARTS.items():
        miss = [(c, ds, k, s) for c in cfgs for s in seeds for ds, k in cells if (c, ds, k, s, T) not in F]
        print(f"- part {part}: {len(cfgs) * len(seeds) * len(cells) - len(miss)} of {len(cfgs) * len(seeds) * len(cells)} runs")
        if miss:
            fails.append(f"part {part}: {len(miss)} runs missing, e.g. {miss[:3]}")
    # identity: one code identity per host
    idents = collections.defaultdict(set)
    for r in F.values():
        idents[r["host"]].add(json.dumps(r["code"], sort_keys=True))
    for h, v in sorted(idents.items()):
        print(f"- code identity on {h}: {len(v)}")
        if len(v) != 1:
            fails.append(f"{len(v)} code identities on {h}")
    # equality with round 3 (R, and the PR's runs)
    for label, cfgs in (("R (ours without the early stop)", PARTS["R"][2]), ("the PR's rule (F2, F5)", ["pr_head79440bc"])):
        n = same = 0
        for key, r in F.items():
            if key[0] not in cfgs:
                continue
            q = R.get((R3[key[0]],) + key[1:])
            n += 1
            if q is not None and (r["test_mse"], r["train_mse"], r["n_splits"]) == (q["test_mse"], q["train_mse"], q["n_splits"]):
                same += 1
        print(f"- {label}: {same} of {n} runs identical to round 3 (test and training error, splits)")
        if same != n:
            fails.append(f"{label}: {n - same} runs differ from round 3")
    if fails:
        print("\n**CHECKS FAILED**: " + "; ".join(fails))
        sys.exit(1)
    print("\nAll checks passed.\n")
    for part in ("F2", "F5"):
        T, seeds, _ = PARTS[part]
        seeds = list(seeds)
        print(f"## {T} epochs ({len(seeds)} seeds x {len(cells)} cells)\n")
        print("| comparison | test error | seeds lower | training error | seeds lower | time (median ratio) |")
        print("|---|---|---|---|---|---|")
        comps = [("rel79440bc_nofreeze_hl", F, "rel79440bc_freeze_hl", F),
                 ("rel79440bc_nofreeze_hl", F, "pr_head79440bc", F),
                 ("rel79440bc_freeze_hl", F, "pr_head79440bc", F),
                 ("rel79440bc_nofreeze_hl", F, "ngh79440bc_medg_pm_nofreeze_guard", R),
                 ("rel79440bc_freeze_hl", F, "ngh79440bc_medg_pm_freeze_guard", R)]
        for ca, A, cb, B in comps:
            te, lo_te, n, p_te = est(A, ca, B, cb, T, seeds, cells, "test_mse")
            tr, lo_tr, _, p_tr = est(A, ca, B, cb, T, seeds, cells, "train_mse")
            # Against round 3 the time is not comparable (its runs used the slower research implementation).
            tm = "not comparable" if B is R else f"{time_ratio(A, ca, B, cb, T, seeds, cells):.3f}"
            print(f"| {NAMES[ca]} vs {NAMES[cb]} | {te} (p {p_te:.2g}) | {lo_te}/{n} | {tr} (p {p_tr:.2g}) | {lo_tr}/{n} | "
                  f"{tm} |")
        print("\nWilcoxon p: two-sided, unadjusted, descriptive.\n")
        print("| configuration | early stop fired | its step / total steps (median) | checks (median) | splits (median) | "
              "moves in the last pass | runs with > k/4 empty centres |")
        print("|---|---|---|---|---|---|---|")
        for c in PARTS[part][2]:
            rs = [F[(c, ds, k, s, T)] for s in seeds for ds, k in cells]
            fired = [r for r in rs if r.get("es_stop_step") is not None]
            frac = st.median(r["es_stop_step"] / r["total_steps"] for r in fired) if fired else float("nan")
            print(f"| {NAMES[c]} | {len(fired)}/{len(rs)} | {frac:.2f} | {st.median(r['n_checks'] for r in rs):.0f} | "
                  f"{st.median(r['n_splits'] for r in rs):.0f} | {sum(r.get('n_moves_last_pass', 0) for r in rs)} | "
                  f"{sum(r['empty'] > r['k'] / 4 for r in rs)} |")
        print("\nPer dataset (geometric-mean ratio over its cells and the seeds; test / training error):\n")
        print("| dataset | no freeze vs freeze | no freeze vs the PR's rule | freeze vs the PR's rule |")
        print("|---|---|---|---|")
        for ds in sorted({c[0] for c in cells}):
            cs = [c for c in cells if c[0] == ds]
            row = []
            for ca, cb in (("rel79440bc_nofreeze_hl", "rel79440bc_freeze_hl"), ("rel79440bc_nofreeze_hl", "pr_head79440bc"),
                           ("rel79440bc_freeze_hl", "pr_head79440bc")):
                g = lambda m: 100 * (math.exp(np.mean([math.log(F[(ca, d, k, s, T)][m] / F[(cb, d, k, s, T)][m])
                                                        for d, k in cs for s in seeds])) - 1)
                row.append(f"{g('test_mse'):+.2f}% / {g('train_mse'):+.2f}%")
            print(f"| {ds} | " + " | ".join(row) + " |")
        print()


if __name__ == "__main__":
    main()
