"""Pre-registered analysis of the second confirmatory round (arxiv/confirm2/PROTOCOL.md, Section 6). Written before any
run of the round; its sha256 prefix is recorded in the protocol.

usage: python confirm2_analyze.py RESULTS_ROOT [--interim] > analysis.md
  RESULTS_ROOT/<host>/cfm2_grid/runs/*.json (parts P, S5, SK) and RESULTS_ROOT/<host>/cfm2_eq/runs/*.json (EQ2, EQ5)

The analysis refuses to run (exit 2) if the records' provenance is not clean (Section 6.5): duplicate keys, more than one
code/environment identity per implementation and host, a record made on the wrong host or with the wrong implementation, more
than one data identity per dataset, or more than one k-means++ start per (dataset, k, seed).
A part's estimates are printed only when the part is complete, or with --interim (labelled INTERIM, no decision).
No confirmatory decision is issued unless the equivalence gate of every variant it involves has passed (Section 6.4).
"""
import glob, json, math, os, sys
from collections import Counter, defaultdict

import numpy as np
from scipy import stats

ROOT = sys.argv[1]
INTERIM = "--interim" in sys.argv[2:]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("EXPERIMENTS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from confirm2_grid import CELLS16, NEW18, CELLS34, CONFIGS7, VARIANTS, PARTS, HOST, run_key  # noqa: E402

PR, PM, FH, FROZEN = "pr", "ngp2_full_medg_pm", "ngp2_full_medg_fh", "ngp2_full"
SHORT = {"lib": "lib", "pr": "pr", "ours": "ours", "h21": "h21", "ngp2_full": "frozen", PM: "medg_pm", FH: "medg_fh"}
METRICS = (("test_mse", "test"), ("train_mse", "train"))
ALPHA = 0.05
GRID_PARTS, EQ_PARTS = ("P", "S5", "SK"), ("EQ2", "EQ5")


def refuse(msg):
    print(f"# ANALYSIS REFUSED\n\n{msg}\n")
    sys.exit(2)


def load(sub):
    out, dup = {}, []
    for f in sorted(glob.glob(os.path.join(ROOT, "*", sub, "runs", "*.json"))):
        try:
            r = json.load(open(f))
        except Exception:
            refuse(f"unreadable record {f}")
        if r["key"] in out:
            dup.append(r["key"])
        out[r["key"]] = r
    return out, dup


G, DUP = load("cfm2_grid")
EQ, DUP_EQ = load("cfm2_eq")

# ------------------------------------------------------------------------------------------ provenance (Section 6.5)
def registered_part(r, name):
    """The part a record must belong to, from its key fields (None if it belongs to no registered part)."""
    for p in (GRID_PARTS if name == "grid" else EQ_PARTS):
        for cells, seeds, st, T, cfgs in PARTS[p]:
            if ((r["dataset"], r["k"]) in cells and r["seed"] in seeds and r["init"] == st and r["T"] == T
                    and r["config"] in cfgs):
                return p
    return None


problems = []
if DUP or DUP_EQ:
    problems.append(f"duplicate keys: {(DUP + DUP_EQ)[:5]} ({len(DUP) + len(DUP_EQ)} in all)")
for name, recs, impl in (("grid", G, "cpp"), ("EQ", EQ, "py")):
    for h in sorted(set(HOST.values())):        # each host has its own compiled libraries: one identity per host
        codes = Counter(json.dumps(r.get("code"), sort_keys=True) for r in recs.values() if HOST[r["dataset"]] == h)
        if len(codes) > 1:
            problems.append(f"{name} records on {h} come from {len(codes)} code/environment identities")
    wrong_impl = [k for k, r in recs.items() if r.get("impl") != impl]
    if wrong_impl:
        problems.append(f"{name} records with implementation other than {impl}: {wrong_impl[:3]}")
    wrong_host = [k for k, r in recs.items() if not str(r.get("host", "")).startswith(HOST[r["dataset"]])]
    if wrong_host:
        problems.append(f"{name} records made on the wrong host: {wrong_host[:3]}")
    wrong_key = [k for k, r in recs.items()
                 if run_key(r["config"], r["dataset"], r["k"], r["seed"], r["init"], r["T"]) != k]
    if wrong_key:
        problems.append(f"{name} records whose fields do not match their key: {wrong_key[:3]}")
    wrong_part = [k for k, r in recs.items()
                  if registered_part(r, name) is None or r.get("part") != registered_part(r, name)]
    if wrong_part:
        problems.append(f"{name} records outside their registered part: {wrong_part[:3]}")
dsha, ssha = defaultdict(set), defaultdict(set)
for r in list(G.values()) + list(EQ.values()):
    dsha[r["dataset"]].add((r.get("data_sha"), r.get("data_test_sha")))
    if r["init"] == "kmpp":
        ssha[(r["dataset"], r["k"], r["seed"])].add(r.get("start_sha"))
problems += [f"dataset {d} has {len(s)} data identities" for d, s in dsha.items() if len(s) != 1]
problems += [f"k-means++ start {key} has {len(s)} identities" for key, s in ssha.items() if len(s) != 1]
if problems:
    refuse("Provenance check failed:\n\n" + "\n".join(f"- {p}" for p in problems))


def rec(cfg, ds, k, s, st, T):
    return G.get(run_key(cfg, ds, k, s, st, T))


def part_spec(p):
    (cells, seeds, st, T, cfgs), = PARTS[p]
    return cells, seeds, st, T, cfgs


def completeness(p):
    cells, seeds, st, T, cfgs = part_spec(p)
    exp = [(c, ds, k, s) for c in cfgs for ds, k in cells for s in seeds]
    have = sum(1 for c, ds, k, s in exp if rec(c, ds, k, s, st, T) is not None)
    return have, len(exp)


def wilcoxon_p(x):
    """Two-sided Wilcoxon signed-rank p of x against 0: zeros dropped (counted by the caller); exact without ties and
    n <= 50, otherwise the normal approximation with continuity correction; p = 1 for a single nonzero value."""
    x = np.asarray(x, float)
    x = x[x != 0]
    if len(x) == 0:
        return float("nan")
    if len(x) == 1:
        return 1.0
    exact = len(np.unique(np.abs(x))) == len(x) and len(x) <= 50
    return float(stats.wilcoxon(x, zero_method="wilcox", method="exact" if exact else "approx",
                                correction=not exact).pvalue)


def hodges_lehmann(x):
    """Pseudomedian (median of the Walsh averages), the location estimate that goes with the signed-rank test."""
    x = np.asarray(x, float)
    i, j = np.triu_indices(len(x))
    return float(np.median((x[i] + x[j]) / 2.0))


def holm(ps):
    """Holm-adjusted p-values (same order as given)."""
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    adj, run = np.empty_like(ps), 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (len(ps) - i) * ps[j]))
        adj[j] = run
    return adj


def seedwise(p, a, b, m, cells=None):
    """{seed: mean over the cells of log(m_a / m_b)}, the pairs used, and the pairs missing."""
    pc, seeds, st, T, _ = part_spec(p)
    cells = pc if cells is None else cells
    out, missing, used = {}, 0, 0
    for s in seeds:
        v = []
        for ds, k in cells:
            x, y = rec(a, ds, k, s, st, T), rec(b, ds, k, s, st, T)
            if x is None or y is None:
                missing += 1
                continue
            v.append(math.log(x[m] / y[m]))
        if v:
            out[s] = float(np.mean(v))
            used += len(v)
    return out, used, missing


def summary(d):
    v = np.array(list(d.values()))
    if len(v) == 0:
        return None
    return dict(est=math.exp(v.mean()) - 1.0, hl=math.exp(hodges_lehmann(v)) - 1.0, p=wilcoxon_p(v), n=len(v),
                better=int((v < 0).sum()), zeros=int((v == 0).sum()))


def decide(sm, p_adj):
    """Section 6.1: significant only with adjusted p < alpha AND the mean and the pseudomedian on the same side of 0."""
    if not p_adj < ALPHA:
        return "not significant"
    if sm["est"] < 0 and sm["hl"] < 0:
        return "lower"
    if sm["est"] > 0 and sm["hl"] > 0:
        return "higher"
    return "direction ambiguous (mean and pseudomedian disagree)"


def fmt(sm, test=True):
    if sm is None:
        return "–"
    s = f"{100 * sm['est']:+.2f}%"
    if test:
        s += f" (p {sm['p']:.2g}, {sm['better']}/{sm['n']} seeds better" + (f", {sm['zeros']} ties" if sm["zeros"] else "") + ")"
    return s


def cell_ratios(p, a, b, m, ds, k):
    _, seeds, st, T, _ = part_spec(p)
    r = []
    for s in seeds:
        x, y = rec(a, ds, k, s, st, T), rec(b, ds, k, s, st, T)
        if x is not None and y is not None:
            r.append(x[m] / y[m])
    return np.array(r)


# ------------------------------------------------------------------------------------- equivalence gate (Section 6.4)
def gate(variant, eqpart):
    """PASS / FAIL / INCOMPLETE for one variant and one EQ part, over the registered keys."""
    (cells, seeds, st, T, _), = PARTS[eqpart]
    same, diff, missing = 0, [], 0
    for ds, k in cells:
        for s in seeds:
            key = run_key(variant, ds, k, s, st, T)
            e, g = EQ.get(key), G.get(key)
            if e is None or g is None:
                missing += 1
                continue
            finite = all(np.isfinite(x[m]) for x in (e, g) for m in ("test_mse", "train_mse"))
            close = finite and all(abs(e[m] - g[m]) <= 1e-9 * abs(g[m]) for m in ("test_mse", "train_mse"))
            if e["n_splits"] == g["n_splits"] and close:
                same += 1
            else:
                diff.append(f"{key} (splits {e['n_splits']} vs {g['n_splits']}, test {e['test_mse']:.10g} vs "
                            f"{g['test_mse']:.10g}, train {e['train_mse']:.10g} vs {g['train_mse']:.10g})")
    status = "FAIL" if diff else ("INCOMPLETE" if missing else "PASS")
    return status, same, diff, missing


GATES = {(v, e): gate(v, e) for v in VARIANTS for e in EQ_PARTS}


def gate_ok(needs):
    """(ok, reason) for a list of (variant, EQ part) gates."""
    bad = [(v, e, GATES[(v, e)][0]) for v, e in needs if GATES[(v, e)][0] != "PASS"]
    if not bad:
        return True, ""
    return False, "; ".join(f"{SHORT[v]} {e} {st}" for v, e, st in bad)


print("# Second confirmatory round: analysis (pre-registered; arxiv/confirm2/PROTOCOL.md, Section 6)\n")
print(f"Records: {len(G)} grid runs, {len(EQ)} EQ runs. Provenance check passed (no duplicate key; one code/environment "
      f"identity per implementation and host; hosts, parts, implementations, data and k-means++ starts consistent).\n")
errs = sum(1 for f in glob.glob(os.path.join(ROOT, "*", "cfm2_*", "errors.jsonl")) for _ in open(f))
print(f"Error records: {errs}.\n")

print("## Completeness and equivalence gate\n")
print("| part | runs present | expected | complete |")
print("|---|---|---|---|")
COMPLETE = {}
for p in GRID_PARTS:
    have, exp = completeness(p)
    COMPLETE[p] = have == exp
    print(f"| {p} | {have} | {exp} | {'yes' if have == exp else 'NO'} |")
print()
print("| gate | identical | different | missing | status |")
print("|---|---|---|---|---|")
for (v, e), (st_, same, diff, missing) in GATES.items():
    print(f"| {SHORT[v]} {e} | {same} | {len(diff)} | {missing} | **{st_}** |")
for (v, e), (st_, same, diff, missing) in GATES.items():
    if diff:
        print(f"\nDifferences, {SHORT[v]} {e}: " + "; ".join(diff[:5]))
print()
SHOW = {p: COMPLETE[p] or INTERIM for p in COMPLETE}
if INTERIM:
    print("**INTERIM: some numbers below come from incomplete parts. They carry no decision.**\n")

# ---------------------------------------------------------------------------------------------- primary (Section 6.1)
print("## 1. Primary: medg_pm vs pr on part P (34 cells, seeds 30-49), co-primary test and training MSE\n")
if not SHOW["P"]:
    print("Part P is incomplete: no estimate, no test.\n")
else:
    ok, why = gate_ok([(PM, "EQ2")])
    rows, ps = [], []
    for m, name in METRICS:
        d, used, miss = seedwise("P", PM, PR, m)
        sm = summary(d)
        rows.append((name, sm, used, miss))
        ps.append(sm["p"])
    adj = holm(ps)
    print("| endpoint | estimate (geometric mean ratio - 1) | pseudomedian | Wilcoxon p | Holm-adjusted p | seeds better / ties | pairs used / missing | decision |")
    print("|---|---|---|---|---|---|---|---|")
    for (name, sm, used, miss), pa in zip(rows, adj):
        r = decide(sm, pa)
        if not COMPLETE["P"]:
            dec = "interim: none"
        elif not ok:
            dec = f"withheld: equivalence gate not passed ({why})"
        else:
            dec = {"lower": "**confirmed** (medg_pm lower)", "higher": "**refuted**: medg_pm significantly higher",
                   "not significant": "not confirmed"}.get(r, "not confirmed: " + r)
        print(f"| {name} | {100 * sm['est']:+.2f}% | {100 * sm['hl']:+.2f}% | {sm['p']:.3g} | {pa:.3g} | "
              f"{sm['better']}/{sm['n']} / {sm['zeros']} | {used} / {miss} | {dec} |")
    print()
    print("Adverse cells (per-cell two-sided Wilcoxon over the 20 seeds, Holm over the 34 cells; a cell is significantly "
          "worse when the adjusted p < 0.05 and both the median ratio and the pseudomedian of the log ratios are above "
          "1 and 0; better likewise below):\n")
    if not COMPLETE["P"] or not ok:
        print("Withheld: " + ("part P is incomplete." if not COMPLETE["P"] else f"equivalence gate not passed ({why}).") + "\n")
    else:
        for m, name in METRICS:
            pc, mc, hc = [], [], []
            for ds, k in CELLS34:
                r = cell_ratios("P", PM, PR, m, ds, k)
                pc.append(wilcoxon_p(np.log(r)) if len(r) else float("nan"))
                mc.append(float(np.median(r)) if len(r) else float("nan"))
                hc.append(hodges_lehmann(np.log(r)) if len(r) else float("nan"))
            a = holm(np.nan_to_num(pc, nan=1.0))
            cells = list(zip(CELLS34, mc, hc, a))
            worse = [f"{ds} {k} (median {100 * (mm - 1):+.1f}%, adj. p {aa:.2g})" for (ds, k), mm, hh, aa in cells
                     if aa < ALPHA and mm > 1 and hh > 0]
            better = [f"{ds} {k}" for (ds, k), mm, hh, aa in cells if aa < ALPHA and mm < 1 and hh < 0]
            ambig = [f"{ds} {k}" for (ds, k), mm, hh, aa in cells
                     if aa < ALPHA and not ((mm > 1 and hh > 0) or (mm < 1 and hh < 0))]
            print(f"- {name}: significantly worse: {'; '.join(worse) or 'none'}. Significantly better: "
                  f"{len(better)} cells{(' (' + ', '.join(better) + ')') if better else ''}."
                  + (f" Direction ambiguous: {', '.join(ambig)}." if ambig else ""))
        print()

# -------------------------------------------------------------------------------------------- secondary (Section 6.2)
SEC = [("P", FH, PR, "test_mse"), ("P", FH, PR, "train_mse"), ("P", PM, FROZEN, "test_mse"),
       ("P", PM, "lib", "test_mse"), ("P", PM, "ours", "test_mse"), ("P", PM, "h21", "test_mse"),
       ("S5", PM, PR, "test_mse"), ("S5", PM, PR, "train_mse"), ("SK", PM, PR, "test_mse"), ("SK", PM, PR, "train_mse")]
SEC_GATES = [(PM, "EQ2"), (FH, "EQ2"), (PM, "EQ5")]
print("## 2. Secondary tests (one Holm family of 10 at alpha = 0.05)\n")
full = all(COMPLETE[p] for p in GRID_PARTS)
ok, why = gate_ok(SEC_GATES)
if not all(SHOW[p] for p in GRID_PARTS):
    print("Not all of P, S5 and SK are complete: the family is evaluated when all three are complete. Available parts "
          "are described in Section 3.\n")
else:
    res = []
    for p, a, b, m in SEC:
        d, used, miss = seedwise(p, a, b, m)
        res.append((p, a, b, m, summary(d), used, miss))
    adj = holm([r[4]["p"] for r in res])
    if full and not ok:
        print(f"**Results withheld: equivalence gate not passed ({why}).**\n")
    print("| # | part | comparison | endpoint | estimate | pseudomedian | p | Holm-adjusted p | seeds better / ties | result |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for i, ((p, a, b, m, sm, used, miss), pa) in enumerate(zip(res, adj), 1):
        r = "interim" if not full else ("withheld" if not ok else decide(sm, pa))
        print(f"| {i} | {p} | {SHORT[a]} vs {SHORT[b]} | {m.split('_')[0]} | {100 * sm['est']:+.2f}% | "
              f"{100 * sm['hl']:+.2f}% | {sm['p']:.3g} | {pa:.3g} | {sm['better']}/{sm['n']} / {sm['zeros']} | {r} |")
    print()

# ------------------------------------------------------------------------------------------ descriptive (Section 6.3)
print("## 3. Descriptive results (no tests; p-values unadjusted)\n")
OTHERS = [c for c in CONFIGS7 if c != PR]
for p in GRID_PARTS:
    cells, seeds, st, T, cfgs = part_spec(p)
    have, exp = completeness(p)
    print(f"### Part {p} ({st} starts, T = {T}, seeds {seeds[0]}-{seeds[-1]}; {have} of {exp} runs)\n")
    if not SHOW[p]:
        print("Incomplete: not shown.\n")
        continue
    print("| config | test vs pr | train vs pr | test vs medg_pm | train vs medg_pm |")
    print("|---|---|---|---|---|")
    for c in CONFIGS7:
        row = [fmt(summary(seedwise(p, c, PR, m)[0])) if c != PR else "" for m, _ in METRICS]
        row += [fmt(summary(seedwise(p, c, PM, m)[0])) if c != PM else "" for m, _ in METRICS]
        print(f"| {SHORT[c]} | " + " | ".join(row) + " |")
    print()
    print("Original 16 cells and additional 18 cells separately (vs pr, test / train):\n")
    print("| config | 16 original | 18 additional |")
    print("|---|---|---|")
    for c in OTHERS:
        cols = []
        for cs in (CELLS16, NEW18):
            t = summary(seedwise(p, c, PR, "test_mse", cs)[0])
            r = summary(seedwise(p, c, PR, "train_mse", cs)[0])
            cols.append(f"{fmt(t, False)} / {fmt(r, False)}")
        print(f"| {SHORT[c]} | " + " | ".join(cols) + " |")
    print()
    print("Per dataset: geometric mean ratio to pr over its cells and seeds (test / train), and the contribution of "
          "medg_pm to the 34-cell log estimate (test):\n")
    dsets = list(dict.fromkeys(ds for ds, _ in cells))
    print("| dataset | cells | " + " | ".join(SHORT[c] for c in OTHERS) + " | contribution of medg_pm (test) |")
    print("|---|---|" + "---|" * (len(OTHERS) + 1))
    for dname in dsets:
        cc = [(ds, k) for ds, k in cells if ds == dname]
        cols = []
        for c in OTHERS:
            t = summary(seedwise(p, c, PR, "test_mse", cc)[0])
            r = summary(seedwise(p, c, PR, "train_mse", cc)[0])
            cols.append(f"{fmt(t, False)} / {fmt(r, False)}")
        dd = seedwise(p, PM, PR, "test_mse", cc)[0]
        contrib = 100 * np.mean(list(dd.values())) * len(cc) / len(cells) if dd else float("nan")
        print(f"| {dname} | {len(cc)} | " + " | ".join(cols) + f" | {contrib:+.2f}% |")
    print()
    print("Leave one dataset out (medg_pm vs pr; test / train; Wilcoxon p unadjusted):\n")
    print("| dataset left out | test | train |")
    print("|---|---|---|")
    for dname in dsets:
        cc = [(ds, k) for ds, k in cells if ds != dname]
        print(f"| {dname} | {fmt(summary(seedwise(p, PM, PR, 'test_mse', cc)[0]))} | "
              f"{fmt(summary(seedwise(p, PM, PR, 'train_mse', cc)[0]))} |")
    print()
    print("Per cell: median ratio to pr over seeds (test / train):\n")
    print("| cell | " + " | ".join(SHORT[c] for c in OTHERS) + " |")
    print("|---|" + "---|" * len(OTHERS))
    for ds, k in cells:
        vals = []
        for c in OTHERS:
            t, r = cell_ratios(p, c, PR, "test_mse", ds, k), cell_ratios(p, c, PR, "train_mse", ds, k)
            vals.append(f"{100 * (np.median(t) - 1):+.1f}% / {100 * (np.median(r) - 1):+.1f}%" if len(t) else "–")
        print(f"| {ds} {k} | " + " | ".join(vals) + " |")
    print()
    run_away = defaultdict(list)
    for c in CONFIGS7:
        for ds, k in cells:
            for s in seeds:
                r = rec(c, ds, k, s, st, T)
                if r is not None and r["empty"] > 0.25 * k:
                    run_away[c].append(f"{ds} {k} s{s} ({r['empty']} empty)")
    print("Stability: runs with more than a quarter of the centres empty at the end (on Shuttle common for every rule): " +
          ("; ".join(f"{SHORT[c]}: {len(v)} ({', '.join(v[:4])}{', ...' if len(v) > 4 else ''})"
                     for c, v in run_away.items() if v) or "none") + ".\n")
    tm = []
    for c in CONFIGS7:
        r = [rec(c, ds, k, s, st, T)["seconds"] / rec(PR, ds, k, s, st, T)["seconds"] for ds, k in cells for s in seeds
             if rec(c, ds, k, s, st, T) is not None and rec(PR, ds, k, s, st, T) is not None]
        tm.append(f"{SHORT[c]} {np.median(r):.2f}x" if r else f"{SHORT[c]} –")
    print("Fit time, median ratio to pr (shared hosts; not a timing study): " + ", ".join(tm) + ".\n")
