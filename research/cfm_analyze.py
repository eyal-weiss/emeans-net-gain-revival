"""Analysis of the confirmatory study exactly as fixed in arxiv/protocol_ngp.md, Section 6 (and the EQ gate, Section 2).

usage: python -I cfm_analyze.py RESULTS_ROOT > report.md
RESULTS_ROOT holds <host>/cfm_{grid,eq,c7,timing}/runs/*.json (as copied from the servers).
Tests are computed only for complete parts; incomplete parts are shown as INTERIM (descriptive, no tests).
"""
import glob, json, math, os, sys
from collections import defaultdict

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("EXPERIMENTS_DIR", os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
sys.argv = sys.argv[:2]
import confirm_grid as CG  # noqa: E402

ROOT = sys.argv[1]
RUNS = {}                                  # stage -> key -> record
for stage in ("grid", "eq", "c7", "timing"):
    RUNS[stage] = {}
    for f in glob.glob(os.path.join(ROOT, "*", f"cfm_{stage}", "runs", "*.json")):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        RUNS[stage][r["key"]] = r
G = RUNS["grid"]
ALL = dict(G)
ALL.update(RUNS["c7"])
key = CG.run_key


def rec(cfg, ds, k, s, st, T, src=ALL):
    return src.get(key(cfg, ds, k, s, st, T))


def wilcoxon(x):
    """Two-sided signed-rank test as fixed in the protocol: zeros dropped (counted), exact without ties and n <= 50."""
    x = np.asarray(x, dtype=float)
    nz = x[x != 0]
    zeros = len(x) - len(nz)
    if len(nz) == 0:
        return float("nan"), zeros, "none"
    exact = len(np.unique(np.abs(nz))) == len(nz) and len(nz) <= 50
    res = stats.wilcoxon(nz, zero_method="wilcox", alternative="two-sided",
                         method="exact" if exact else "approx", correction=not exact)
    return float(res.pvalue), zeros, "exact" if exact else "approx"


def holm(ps):
    ps = list(ps)
    order = sorted(range(len(ps)), key=lambda i: (math.isnan(ps[i]), ps[i]))
    adj, running = [float("nan")] * len(ps), 0.0
    m = sum(not math.isnan(p) for p in ps)
    for rank, i in enumerate(order):
        if math.isnan(ps[i]):
            continue
        running = max(running, min(1.0, (m - rank) * ps[i]))
        adj[i] = running
    return adj


def pct(r):
    return f"{100 * (r - 1):+.1f}%"


def part_spec(part):
    return CG.PARTS[part]


def expected(part):
    n = 0
    for cells, seeds, st, T, cfgs in part_spec(part):
        n += len(cells) * len(seeds) * len(cfgs)
    return n


def finished(part, src=ALL):
    n = 0
    for cells, seeds, st, T, cfgs in part_spec(part):
        n += sum(1 for ds, k in cells for s in seeds for c in cfgs if rec(c, ds, k, s, st, T, src))
    return n


def seed_level(cells, seeds, st, T, a, b):
    """log R_s per seed (geo-mean over cells of test(a)/test(b)); seeds with any missing pair are dropped."""
    out, missing = {}, 0
    for s in seeds:
        logs = []
        for ds, k in cells:
            ra, rb = rec(a, ds, k, s, st, T), rec(b, ds, k, s, st, T)
            if ra is None or rb is None:
                logs = None
                break
            logs.append(math.log(ra["test_mse"] / rb["test_mse"]))
        if logs is None:
            missing += 1
        else:
            out[s] = float(np.mean(logs))
    return out, missing


def cell_table(cells, seeds, st, T, a, comps, tests):
    lines = ["| cell | seeds | " + " | ".join(f"vs {c}: median ratio" + (" (Holm p)" if tests else "") for c in comps)
             + " |", "|---|---|" + "---|" * len(comps)]
    meds = {c: {} for c in comps}
    ps = {c: {} for c in comps}
    for c in comps:
        for ds, k in cells:
            x = [math.log(rec(a, ds, k, s, st, T)["test_mse"] / rec(c, ds, k, s, st, T)["test_mse"])
                 for s in seeds if rec(a, ds, k, s, st, T) and rec(c, ds, k, s, st, T)]
            if x:
                meds[c][(ds, k)] = math.exp(float(np.median(x)))
                ps[c][(ds, k)] = wilcoxon(x)[0] if tests else float("nan")
        if tests:
            adj = holm([ps[c][cell] for cell in meds[c]])
            ps[c] = dict(zip(meds[c], adj))
    for ds, k in cells:
        n = sum(1 for s in seeds if rec(a, ds, k, s, st, T))
        row = []
        for c in comps:
            if (ds, k) in meds[c]:
                txt = pct(meds[c][(ds, k)])
                if tests:
                    p = ps[c][(ds, k)]
                    flag = " **worse**" if (p < 0.05 and meds[c][(ds, k)] > 1) else (" better" if p < 0.05 else "")
                    txt += f" ({p:.3g}){flag}"
                row.append(txt)
            else:
                row.append("–")
        lines.append(f"| {ds} {k} | {n} | " + " | ".join(row) + " |")
    summ = []
    for c in comps:
        v = np.array(list(meds[c].values()))
        if len(v):
            summ.append(f"vs {c}: geo-mean of cell medians {pct(math.exp(np.mean(np.log(v))))}, worst cell "
                        f"{pct(v.max())}, cells better/worse by >1%: {int((v < 0.99).sum())}/{int((v > 1.01).sum())}")
    return "\n".join(lines), summ


def main():
    print("## Progress\n")
    print("| part | finished runs | expected | complete |")
    print("|---|---|---|---|")
    complete = {}
    for part in ("C1", "C2", "C3", "C4", "C5", "C6", "C7", "EQ", "TIMING"):
        src = RUNS["eq"] if part == "EQ" else RUNS["timing"] if part == "TIMING" else ALL
        f, e = finished(part, src), expected(part)
        complete[part] = f == e
        print(f"| {part} | {f} | {e} | {'yes' if f == e else 'no'} |")
    errs = sum(len(open(f).readlines()) for f in glob.glob(os.path.join(ROOT, "*", "cfm_*", "errors.jsonl")))
    nonfinite = [k for st in RUNS.values() for k, r in st.items()
                 if not all(math.isfinite(r.get(x, float("nan"))) for x in ("test_mse", "train_mse"))]
    print(f"\nRun errors logged: {errs}. Records with non-finite errors: {len(nonfinite)}.")

    # ---------------- EQ gate
    print("\n## Equivalence gate (EQ: reference implementation vs C++)\n")
    same = diff = 0
    worst = 0.0
    bad = []
    for k_, r in RUNS["eq"].items():
        t = G.get(k_)
        if t is None:
            continue
        rel = abs(r["test_mse"] - t["test_mse"]) / abs(r["test_mse"])
        worst = max(worst, rel)
        if r["n_splits"] == t["n_splits"] and rel <= 1e-9:
            same += 1
        else:
            diff += 1
            bad.append(k_)
    print(f"Pairs compared: {same + diff}; passing (same splits, relative test difference <= 1e-9): {same}; "
          f"failing: {diff}; largest relative test difference: {worst:.2e}.")
    if bad:
        print("Failing pairs: " + ", ".join(bad[:20]))
    ok = diff == 0 and not nonfinite
    print("Gate: " + (("**PASS**" if complete["EQ"] else "**PASS so far** (EQ incomplete)") if ok else "**FAIL**"))

    # ---------------- primary
    cells, seeds, st, T, _ = part_spec("C1")[0]
    print("\n## Primary analysis (C1: ngp2_full vs pr, test MSE)\n")
    R, missing = seed_level(cells, seeds, st, T, "ngp2_full", "pr")
    if R:
        est = math.exp(np.mean(list(R.values())))
        print(f"Seeds with all 16 cells for both configs: {len(R)} of {len(seeds)}. "
              f"Estimate (geometric mean of R_s): **{pct(est)}**.")
        if complete["C1"]:
            p, zeros, how = wilcoxon(list(R.values()))
            print(f"Wilcoxon signed-rank on log R_s ({how}; zeros dropped: {zeros}): p = {p:.3g}.")
            print(f"**H1 {'CONFIRMED' if est < 1 and p < 0.05 else 'NOT CONFIRMED'}** (rule: estimate < 1 and p < 0.05).")
        else:
            print("**INTERIM**: C1 is incomplete; no test is computed (protocol Section 6).")
        print("\nPer-seed R_s: " + ", ".join(f"s{s} {pct(math.exp(v))}" for s, v in sorted(R.items())))
    tab, summ = cell_table(cells, seeds, st, T, "ngp2_full", ["pr", "lib", "ours", "h21"], complete["C1"])
    print("\nPer-cell medians over seeds of test(ngp2_full) / test(comparator)"
          + (" with Holm-adjusted per-cell Wilcoxon p" if complete["C1"] else " (INTERIM, no tests)") + ":\n")
    print(tab)
    print("\n" + "\n".join(f"- {x}" for x in summ))

    # ---------------- secondary family
    print("\n## Secondary tests (one Holm family of 8)\n")
    fam = [("C1 vs lib", "C1", 0, "lib"), ("C1 vs ours", "C1", 0, "ours"), ("C1 vs h21", "C1", 0, "h21"),
           ("C2 vs pr", "C2", 0, "pr"), ("C3 random vs pr", "C3", 0, "pr"), ("C3 k-means++ vs pr", "C3", 1, "pr"),
           ("C4 vs pr", "C4", 0, "pr"), ("C4 vs lib", "C4", 0, "lib")]
    rows, ps = [], []
    for name, part, idx, comp in fam:
        cells_, seeds_, st_, T_, _ = part_spec(part)[idx]
        R_, miss = seed_level(cells_, seeds_, st_, T_, "ngp2_full", comp)
        comp_done = len(R_) == len(seeds_)
        est = math.exp(np.mean(list(R_.values()))) if R_ else float("nan")
        p = wilcoxon(list(R_.values()))[0] if comp_done else float("nan")
        rows.append((name, len(R_), len(seeds_), est, p, comp_done))
        ps.append(p)
    adj = holm(ps)
    print("| test | seeds complete | estimate | p | Holm p | status |")
    print("|---|---|---|---|---|---|")
    for (name, n, ns, est, p, done), pa in zip(rows, adj):
        print(f"| {name} | {n}/{ns} | {pct(est) if not math.isnan(est) else '–'} | "
              f"{'' if math.isnan(p) else f'{p:.3g}'} | {'' if math.isnan(pa) else f'{pa:.3g}'} | "
              f"{'complete' if done else 'INTERIM (no test)'} |")
    print("\n(Holm over the tests that are complete; final Holm adjustment when all 8 are complete.)")

    # ---------------- per-part descriptive tables
    for part, idx, title in (("C2", 0, "C2: k-means++ starts"), ("C3", 0, "C3: Covertype, random starts"),
                             ("C3", 1, "C3: Covertype, k-means++ starts"), ("C4", 0, "C4: embeddings"),
                             ("C6", 0, "C6: T = 5 (descriptive)")):
        cells_, seeds_, st_, T_, _ = part_spec(part)[idx]
        tab, summ = cell_table(cells_, seeds_, st_, T_, "ngp2_full", ["pr", "lib", "ours", "h21"], False)
        print(f"\n## {title}{'' if complete[part] else ' (INTERIM)'}\n\nPer-cell medians of test(ngp2_full) / test(comparator):\n")
        print(tab)
        print("\n" + "\n".join(f"- {x}" for x in summ))

    # ---------------- C5 strides
    print(f"\n## C5: sampling 1 row in s for the two-nearest search{'' if complete['C5'] else ' (INTERIM)'}\n")
    for cells_, seeds_, st_, T_, cfgs in part_spec("C5"):
        print("| cell | " + " | ".join(f"s={c.replace('ngp2_full', '1').replace('ngp2_s', '').replace('ngp2', '8')}"
                                     for c in cfgs[1:]) + " | fit time s=1 vs s=8 |")
        print("|---|" + "---|" * len(cfgs))
        allm = defaultdict(list)
        for ds, k in cells_:
            row = []
            for c in cfgs[1:]:
                x = [math.log(rec(c, ds, k, s, st_, T_)["test_mse"] / rec("ngp2_full", ds, k, s, st_, T_)["test_mse"])
                     for s in seeds_ if rec(c, ds, k, s, st_, T_) and rec("ngp2_full", ds, k, s, st_, T_)]
                if x:
                    allm[c].append(float(np.median(x)))
                row.append(pct(math.exp(np.median(x))) if x else "–")
            tt = [rec("ngp2_full", ds, k, s, st_, T_)["seconds"] / rec("ngp2", ds, k, s, st_, T_)["seconds"]
                  for s in seeds_ if rec("ngp2_full", ds, k, s, st_, T_) and rec("ngp2", ds, k, s, st_, T_)]
            row.append(f"{np.median(tt):.2f}x" if tt else "–")
            print(f"| {ds} {k} | " + " | ".join(row) + " |")
        print("| geo-mean | " + " | ".join(pct(math.exp(np.mean(allm[c]))) if allm[c] else "–" for c in cfgs[1:])
              + " | |\n")
    print("(entries: median over seeds of test(ngp2, 1 row in s) / test(ngp2_full); fit times from shared hosts)")

    # ---------------- C7
    print("\n## C7: DEEP 10M, k = 65,536\n")
    print("| config | seed | test MSE | train MSE | splits | fit time (s) |")
    print("|---|---|---|---|---|---|")
    for r in sorted(RUNS["c7"].values(), key=lambda r: (r["seed"], r["config"])):
        print(f"| {r['config']} | {r['seed']} | {r['test_mse']:.6g} | {r['train_mse']:.6g} | {r['n_splits']} | "
              f"{r['seconds']:.0f} |")
    capped = glob.glob(os.path.join(ROOT, "*", "cfm_c7_capped.txt"))
    for f in capped:
        for l in open(f):
            print(f"| capped at 90 min: {l.strip()} | | | | | |")

    # ---------------- TIMING
    print(f"\n## TIMING (one job at a time, 8 threads){'' if complete['TIMING'] else ' (INTERIM)'}\n")
    T_ = RUNS["timing"]
    if T_:
        cells_, seeds_, st_, TT, cfgs = part_spec("TIMING")[0]
        print("| cell | " + " | ".join(f"{c} vs lib" for c in cfgs if c != "lib") + " | ngp2_full vs pr |")
        print("|---|" + "---|" * len(cfgs))
        for ds, k in cells_:
            row = []
            for c in cfgs:
                if c == "lib":
                    continue
                x = [T_[key(c, ds, k, s, st_, TT)]["seconds"] / T_[key("lib", ds, k, s, st_, TT)]["seconds"]
                     for s in seeds_ if key(c, ds, k, s, st_, TT) in T_ and key("lib", ds, k, s, st_, TT) in T_]
                row.append(f"{np.median(x):.2f}x" if x else "–")
            x = [T_[key("ngp2_full", ds, k, s, st_, TT)]["seconds"] / T_[key("pr", ds, k, s, st_, TT)]["seconds"]
                 for s in seeds_ if key("ngp2_full", ds, k, s, st_, TT) in T_ and key("pr", ds, k, s, st_, TT) in T_]
            row.append(f"{np.median(x):.2f}x" if x else "–")
            print(f"| {ds} {k} | " + " | ".join(row) + " |")
        loads = [r["load_before"] for r in T_.values()]
        print(f"\nLoad average before the timed runs: median {np.median(loads):.1f}, max {max(loads):.1f}.")
    else:
        print("Not run yet.")

    # ---------------- splits
    print("\n## Splits per run (C1, median over seeds)\n")
    cells, seeds, st, T, cfgs = part_spec("C1")[0]
    print("| config | " + " | ".join(f"{ds.replace('_z', '')} {k}" for ds, k in cells) + " |")
    print("|---|" + "---|" * len(cells))
    for c in cfgs:
        row = []
        for ds, k in cells:
            v = [rec(c, ds, k, s, st, T)["n_splits"] for s in seeds if rec(c, ds, k, s, st, T)]
            row.append(f"{np.median(v):.0f}" if v else "–")
        print(f"| {c} | " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
