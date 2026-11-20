"""Analysis of the early-stop runs (stop_grid.py records), as pre-specified in stop_grid.py's docstring.
usage: python stop_analyze.py <arxiv/results>  > report.md
"""
import glob, json, math, sys
from collections import Counter, defaultdict

import numpy as np
from scipy import stats

ROOT = sys.argv[1]
DB = {}
for f in glob.glob(f"{ROOT}/pr5915/side/*/side_stop/runs/*.json"):
    r = json.load(open(f))
    DB[(r["config"], r["dataset"], r["k"], r["seed"], r["T"])] = r
CFG = ("pr", "medg_pm", "medg_hl", "medg_e4")
NAME = {"pr": "PR", "medg_pm": "medg_pm", "medg_hl": "medg_pm + stop (half-life, no move)",
        "medg_e4": "medg_pm + stop (net gain < 0.01%)"}


def wilcoxon_p(x):
    x = np.asarray(x, float)
    x = x[x != 0]
    if len(x) == 0:
        return float("nan")
    if len(x) == 1:
        return 1.0
    exact = len(np.unique(np.abs(x))) == len(x) and len(x) <= 50
    return float(stats.wilcoxon(x, zero_method="wilcox", method="exact" if exact else "approx",
                                correction=not exact).pvalue)


def hl(x):
    x = np.asarray(x, float)
    i, j = np.triu_indices(len(x))
    return float(np.median((x[i] + x[j]) / 2.0))


def complete_keys(T):
    """(cells, seeds) for which all four configs exist at T."""
    keys = defaultdict(set)
    for (c, ds, k, s, t) in DB:
        if t == T:
            keys[(ds, k, s)].add(c)
    full = {key for key, cs in keys.items() if set(CFG) <= cs}
    cells = sorted({(ds, k) for ds, k, s in full})
    seeds = sorted({s for ds, k, s in full})
    seeds = [s for s in seeds if all((ds, k, s) in full for ds, k in cells)]   # seeds complete over all cells
    return cells, seeds


def seedwise(a, b, m, T, cells, seeds):
    return {s: float(np.mean([math.log(DB[(a, ds, k, s, T)][m] / DB[(b, ds, k, s, T)][m]) for ds, k in cells]))
            for s in seeds}


def summ(d):
    v = np.array(list(d.values()))
    if len(v) == 0:
        return "n/a"
    zero = int((v == 0).sum())
    return (f"{100 * (math.exp(v.mean()) - 1):+.2f}% (HL {100 * (math.exp(hl(v)) - 1):+.2f}%, p {wilcoxon_p(v):.2g}, "
            f"{int((v < 0).sum())}/{len(v)}" + (f", {zero} ties" if zero else "") + ")")


def time_ratio(a, b, T, cells, seeds):
    v = [math.log(DB[(a, ds, k, s, T)]["seconds"] / DB[(b, ds, k, s, T)]["seconds"]) for s in seeds for ds, k in cells]
    return 100 * (math.exp(np.median(v)) - 1), 100 * (math.exp(np.mean(v)) - 1)


P = print
P("# Automatic early end of the move phase: real runs")
P()
P(f"{len(DB)} runs; configs: " + "; ".join(f"`{c}` = {NAME[c]}" for c in CFG) + ". Fresh seeds (50+). EXPLORATORY, "
  "analysis pre-specified in stop_grid.py. Each entry: exp(mean over seeds of L_s) - 1, L_s = mean over cells of "
  "log(a/b); HL = Hodges-Lehmann; p = Wilcoxon over seeds (descriptive); k/n = seeds where a is lower. Negative = a "
  "lower. Time: paired wall-clock ratio, median (mean) over runs; configs of a key ran in one process.")
P()
for T in (2, 5, 10):
    cells, seeds = complete_keys(T)
    if not seeds:
        P(f"## T = {T}: not complete yet")
        P()
        continue
    ns = [c for c in cells if c[0] != "shuttle_z"]
    P(f"## T = {T} ({len(cells)} cells x seeds {seeds[0]}-{seeds[-1]}, {len(seeds)} seeds)")
    P()
    P("| comparison | test | train | test w/o Shuttle | train w/o Shuttle | time median (mean) |")
    P("|---|---|---|---|---|---|")
    for a, b in (("medg_hl", "medg_pm"), ("medg_e4", "medg_pm"), ("medg_pm", "pr"), ("medg_hl", "pr"),
                 ("medg_e4", "pr")):
        tm, tmn = time_ratio(a, b, T, cells, seeds)
        P(f"| {a} vs {b} | {summ(seedwise(a, b, 'test_mse', T, cells, seeds))} | "
          f"{summ(seedwise(a, b, 'train_mse', T, cells, seeds))} | {summ(seedwise(a, b, 'test_mse', T, ns, seeds))} | "
          f"{summ(seedwise(a, b, 'train_mse', T, ns, seeds))} | {tm:+.1f}% ({tmn:+.1f}%) |")
    P()
    for c in ("medg_hl", "medg_e4"):
        fired = [DB[(c, ds, k, s, T)] for s in seeds for ds, k in cells]
        fr = [r["es_stopped_frac"] for r in fired if r["es_stop_step"] is not None]
        P(f"- `{c}`: fired in {len(fr)}/{len(fired)} runs; share of the move phase completed when it fired: median "
          f"{100*np.median(fr) if fr else float('nan'):.0f}% (min {100*min(fr) if fr else float('nan'):.0f}%)")
    emp = {c: Counter(f"{ds} {k}" for s in seeds for ds, k in cells if DB[(c, ds, k, s, T)]["empty"] > k / 4)
           for c in CFG}
    P("- runs with more than k/4 empty centres: " + "; ".join(f"`{c}` {dict(v) or 'none'}" for c, v in emp.items()))
    P()
    P("Per dataset (geometric means over cells and seeds; time = median paired ratio):")
    P()
    P("| dataset | hl: fired / phase done | hl vs medg_pm: test / train / time | e4: fired / phase done | "
      "e4 vs medg_pm: test / train / time | medg_pm vs PR: test / train / time |")
    P("|---|---|---|---|---|---|")
    for dsn in sorted({ds for ds, k in cells}):
        cl = [(ds, k) for ds, k in cells if ds == dsn]

        def ratio(a, b, m):
            v = [math.log(DB[(a, ds, k, s, T)][m] / DB[(b, ds, k, s, T)][m]) for s in seeds for ds, k in cl]
            return 100 * (math.exp(np.mean(v)) - 1)

        def tr(a, b):
            v = [math.log(DB[(a, ds, k, s, T)]["seconds"] / DB[(b, ds, k, s, T)]["seconds"]) for s in seeds for ds, k in cl]
            return 100 * (math.exp(np.median(v)) - 1)

        def fire(c):
            rr = [DB[(c, ds, k, s, T)] for s in seeds for ds, k in cl]
            fr = [r["es_stopped_frac"] for r in rr if r["es_stop_step"] is not None]
            return f"{len(fr)}/{len(rr)} / {100*np.median(fr):.0f}%" if fr else f"0/{len(rr)} / -"
        P(f"| {dsn} | {fire('medg_hl')} | {ratio('medg_hl','medg_pm','test_mse'):+.2f}% / "
          f"{ratio('medg_hl','medg_pm','train_mse'):+.2f}% / {tr('medg_hl','medg_pm'):+.1f}% | {fire('medg_e4')} | "
          f"{ratio('medg_e4','medg_pm','test_mse'):+.2f}% / {ratio('medg_e4','medg_pm','train_mse'):+.2f}% / "
          f"{tr('medg_e4','medg_pm'):+.1f}% | {ratio('medg_pm','pr','test_mse'):+.2f}% / "
          f"{ratio('medg_pm','pr','train_mse'):+.2f}% / {tr('medg_pm','pr'):+.1f}% |")
    P()
