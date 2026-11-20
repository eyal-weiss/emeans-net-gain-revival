"""EXPLORATORY analysis of the side jobs run after round 2 (2026-11-19 evening; not pre-registered; descriptive only).

Inputs (fetched records):
  results/pr5915/confirm2/<host>/cfm2_grid/runs   round 2 (comparators: pr = the PR's rule at 98deaca, medg_pm, ...)
  results/pr5915/side/<host>/side_prhead98/runs   PV: the PR's plain class at 98deaca (must equal round-2 pr)
  results/pr5915/side/<host>/side_prhead/runs     PH/PH5: the PR head 79440bc, and our rule on its driver (ngh...)
  results/pr5915/side/<host>/side_interval/runs   IV: check interval c = 1, 2, 5, 10, 20 and the event trigger
  results/gpu_netgain/{smoke,egrid,ogrid}/runs    GPU e-means (PR's torch class) and o-means, with and without our rule
Statistics: per seed, L_s = mean over the cells of log(a / b); reported as exp(mean L_s) - 1, the Hodges-Lehmann
pseudomedian, the two-sided Wilcoxon signed-rank p over seeds (descriptive: no family, no correction), and the number
of seeds on which a is lower. Per-cell ratios are geometric means over seeds.
usage: python side_analyze.py <arxiv/results dir>  > report.md
"""
import glob, json, math, os, sys
from collections import Counter, defaultdict

import numpy as np
from scipy import stats

ROOT = sys.argv[1]
SHUTTLE = ("shuttle_z",)


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


def hodges_lehmann(x):
    x = np.asarray(x, float)
    i, j = np.triu_indices(len(x))
    return float(np.median((x[i] + x[j]) / 2.0))


# ------------------------------------------------------------------ records
DB = {}            # (cfg, ds, k, seed, init, T) -> record
SRC = Counter()


def add(pattern, prefix=""):
    for f in glob.glob(os.path.join(ROOT, pattern)):
        r = json.load(open(f))
        key = (prefix + r["config"], r["dataset"], int(r["k"]), int(r["seed"]), r.get("init", "random"), int(r["T"]))
        if key in DB:
            raise SystemExit(f"duplicate record {key} ({f})")
        DB[key] = r
        SRC[pattern.split("/runs")[0].replace("/*/", "/")] += 1


add("pr5915/confirm2/*/cfm2_grid/runs/*.json")
add("pr5915/side/*/side_prhead98/runs/*.json")
add("pr5915/side/*/side_prhead/runs/*.json")
add("pr5915/side/*/side_interval/runs/*.json")
for d in ("egrid", "ogrid"):
    add(f"gpu_netgain/{d}/runs/*.json", prefix="gpu:")


def get(cfg, ds, k, s, T, init="random"):
    return DB.get((cfg, ds, k, s, init, T))


def cells_of(cfg, T, init="random"):
    return sorted({(ds, k) for (c, ds, k, s, i, t) in DB if c == cfg and t == T and i == init})


def seeds_of(cfg, T, init="random"):
    return sorted({s for (c, ds, k, s, i, t) in DB if c == cfg and t == T and i == init})


def seedwise(a, b, m, T, cells, seeds, init="random"):
    out, miss = {}, 0
    for s in seeds:
        v = []
        for ds, k in cells:
            x, y = get(a, ds, k, s, T, init), get(b, ds, k, s, T, init)
            if x is None or y is None:
                miss += 1
                continue
            v.append(math.log(x[m] / y[m]))
        if v:
            out[s] = float(np.mean(v))
    return out, miss


def summ(d):
    v = np.array(list(d.values()))
    if len(v) == 0:
        return "n/a"
    return (f"{100 * (math.exp(v.mean()) - 1):+.2f}% (HL {100 * (math.exp(hodges_lehmann(v)) - 1):+.2f}%, "
            f"p {wilcoxon_p(v):.2g}, {int((v < 0).sum())}/{len(v)})")


def row(label, a, b, T, cells, seeds, init="random", with_noshuttle=True):
    t, mt = seedwise(a, b, "test_mse", T, cells, seeds, init)
    r, mr = seedwise(a, b, "train_mse", T, cells, seeds, init)
    cols = [label, summ(t), summ(r)]
    if with_noshuttle:
        ns = [c for c in cells if c[0] not in SHUTTLE]
        cols += [summ(seedwise(a, b, "test_mse", T, ns, seeds, init)[0]),
                 summ(seedwise(a, b, "train_mse", T, ns, seeds, init)[0])]
    if mt:
        cols[0] += f" ({mt} pairs missing)"
    return "| " + " | ".join(cols) + " |"


HDR = ("| comparison (a vs b) | test | train | test w/o Shuttle | train w/o Shuttle |\n"
       "|---|---|---|---|---|")
HDR3 = "| comparison (a vs b) | test | train |\n|---|---|---|"


def cell_table(pairs, T, cells, seeds, m="test_mse", init="random", top=None):
    """per-cell geometric-mean ratio a/b over seeds, for each (label, a, b)."""
    lines = ["| cell | " + " | ".join(p[0] for p in pairs) + " |", "|---|" + "---|" * len(pairs)]
    for ds, k in cells:
        vals = []
        for _, a, b in pairs:
            v = [math.log(get(a, ds, k, s, T, init)[m] / get(b, ds, k, s, T, init)[m]) for s in seeds
                 if get(a, ds, k, s, T, init) and get(b, ds, k, s, T, init)]
            vals.append(f"{100 * (math.exp(np.mean(v)) - 1):+.1f}%" if v else "n/a")
        lines.append(f"| {ds} {k} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def exact_check(a, b, T, cells, seeds, init="random"):
    n = same = 0
    worst = 0.0
    for s in seeds:
        for ds, k in cells:
            x, y = get(a, ds, k, s, T, init), get(b, ds, k, s, T, init)
            if x is None or y is None:
                continue
            n += 1
            d = max(abs(x[m] - y[m]) / abs(y[m]) for m in ("test_mse", "train_mse"))
            same += d <= 1e-9
            worst = max(worst, d)
    return n, same, worst


def runaways(cfg, T, cells, seeds, init="random"):
    out = Counter()
    for s in seeds:
        for ds, k in cells:
            r = get(cfg, ds, k, s, T, init)
            if r is not None and r.get("empty", 0) > k / 4:
                out[f"{ds} {k}"] += 1
    return dict(out)


def time_ratio(a, b, T, cells, seeds, init="random", field="seconds"):
    v = [math.log(get(a, ds, k, s, T, init)[field] / get(b, ds, k, s, T, init)[field]) for s in seeds
         for ds, k in cells if get(a, ds, k, s, T, init) and get(b, ds, k, s, T, init)]
    return f"{100 * (math.exp(np.median(v)) - 1):+.1f}% (median of {len(v)} paired runs)" if v else "n/a"


P = print
P("# Side jobs after round 2: PR head, our rule on the head's driver, check interval, GPU and o-means")
P()
P("EXPLORATORY (not pre-registered; descriptive p-values, no correction). Generated by "
  "`code/experiments/pr5915/side_analyze.py`.")
P()
P("Records read: " + ", ".join(f"{k} {v}" for k, v in sorted(SRC.items())) + f"; {len(DB)} in all.")
P()
P("Each entry: exp(mean over seeds of L_s) - 1, where L_s = mean over cells of log(a/b); HL = Hodges-Lehmann "
  "pseudomedian; p = two-sided Wilcoxon signed-rank over seeds; k/n = seeds on which a is lower. Negative = a better.")
P()

# ------------------------------------------------------------------ provenance
P("## 0. Provenance")
P()
for name, pred, fields in (
        ("PR head records", lambda c: c.startswith("pr_head") or c.startswith("ngh"), ("e_means", "runner", "ngp_head", "rule", "faiss_file")),
        ("PR 98deaca (PV) records", lambda c: c == "pr_plain98deaca", ("e_means", "runner", "faiss_file"))):
    for f in fields:
        vals = Counter((r["host"], json.dumps(r.get(f), sort_keys=True)) for kk, r in DB.items() if pred(kk[0]))
        P(f"- {name}, `{f}`: " + "; ".join(f"{h}: {v} ({n})" for (h, v), n in sorted(vals.items())))
P()

# ------------------------------------------------------------------ 1. reproduction checks
P("## 1. Reproduction checks")
P()
c34 = cells_of("pr_plain98deaca", 2)
n, same, worst = exact_check("pr_plain98deaca", "pr", 2, c34, seeds_of("pr_plain98deaca", 2))
P(f"- PV, the PR's plain class at 98deaca vs round-2 `pr` (our subclass of it): {same}/{n} runs identical "
  f"(max relative difference {worst:.2g}).")
for T, seeds in ((2, list(range(30, 50))), (5, list(range(30, 40)))):
    cl = cells_of("ngh79440bc_medg_pm_freeze_guard", T)
    n, same, worst = exact_check("ngh79440bc_medg_pm_freeze_guard", "ngp2_full_medg_pm", T, cl, seeds)
    P(f"- T = {T}: our rule with the freeze and the guard on the head's driver vs round-2 `medg_pm` (98deaca driver): "
      f"{same}/{n} runs identical (max relative difference {worst:.2g}).")
P()

# ------------------------------------------------------------------ 2. PR head vs 98deaca
NGH = {"nofreeze_guard": "ngh79440bc_medg_pm_nofreeze_guard", "freeze_guard": "ngh79440bc_medg_pm_freeze_guard",
       "nofreeze_noguard": "ngh79440bc_medg_pm_nofreeze_noguard", "freeze_noguard": "ngh79440bc_medg_pm_freeze_noguard"}
HEAD = "pr_head79440bc"
for T, seeds, label in ((2, list(range(30, 50)), "T = 2, 34 cells, seeds 30-49 (round-2 part P keys)"),
                        (5, list(range(30, 40)), "T = 5, 34 cells, seeds 30-39 (round-2 part S5 keys)")):
    cl = cells_of(HEAD, T)
    P(f"## {'2' if T == 2 else '3'}. {label}")
    P()
    P(HDR)
    P(row("PR head (79440bc) vs PR 98deaca", HEAD, "pr", T, cl, seeds))
    P(row("round-2 medg_pm (98deaca driver) vs PR head", "ngp2_full_medg_pm", HEAD, T, cl, seeds))
    for nm, cfg in NGH.items():
        if cells_of(cfg, T):
            P(row(f"ours on head, {nm} vs PR head", cfg, HEAD, T, cl, seeds))
    for nm, cfg in NGH.items():
        if cells_of(cfg, T):
            P(row(f"ours on head, {nm} vs round-2 medg_pm", cfg, "ngp2_full_medg_pm", T, cl, seeds))
    P(row("ours on head: nofreeze vs freeze (guard)", NGH["nofreeze_guard"], NGH["freeze_guard"], T, cl, seeds))
    if cells_of(NGH["nofreeze_noguard"], T):
        P(row("ours on head: noguard vs guard (nofreeze)", NGH["nofreeze_noguard"], NGH["nofreeze_guard"], T, cl, seeds))
        P(row("ours on head: noguard vs guard (freeze)", NGH["freeze_noguard"], NGH["freeze_guard"], T, cl, seeds))
    P(row("round-2 h21 vs PR head", "h21", HEAD, T, cl, seeds))
    P()
    P("Runs with more than k/4 empty centres:")
    P()
    for cfg in ["pr", HEAD, "ngp2_full", "ngp2_full_medg_pm", "h21"] + list(NGH.values()):
        if cells_of(cfg, T):
            P(f"- `{cfg}`: {runaways(cfg, T, cl, seeds) or 'none'}")
    P()
    P("Per-cell test-error ratios (geometric mean over seeds):")
    P()
    pairs = [("head/98deaca", HEAD, "pr"), ("ours nofreeze/head", NGH["nofreeze_guard"], HEAD),
             ("ours freeze/head", NGH["freeze_guard"], HEAD), ("medg_pm r2/head", "ngp2_full_medg_pm", HEAD),
             ("nofreeze/freeze", NGH["nofreeze_guard"], NGH["freeze_guard"])]
    P(cell_table(pairs, T, cl, seeds))
    P()
    P("Training-error ratios per cell:")
    P()
    P(cell_table(pairs, T, cl, seeds, m="train_mse"))
    P()
    P("Wall-clock (median paired ratio; same host, same load):")
    P()
    P(f"- PR head vs PR 98deaca: {time_ratio(HEAD, 'pr', T, cl, seeds)}")
    P(f"- ours on head nofreeze vs PR head: {time_ratio(NGH['nofreeze_guard'], HEAD, T, cl, seeds)}")
    P(f"- ours on head freeze vs PR head: {time_ratio(NGH['freeze_guard'], HEAD, T, cl, seeds)}")
    mv = [get(NGH["nofreeze_guard"], ds, k, s, T)["n_moves_last_pass"] for s in seeds for ds, k in cl
          if get(NGH["nofreeze_guard"], ds, k, s, T)]
    sp = [get(NGH["nofreeze_guard"], ds, k, s, T)["n_splits"] for s in seeds for ds, k in cl
          if get(NGH["nofreeze_guard"], ds, k, s, T)]
    P(f"- ours nofreeze: moves in the last pass {sum(mv)} of {sum(sp)} splits in all ({100 * sum(mv) / max(sum(sp), 1):.1f}%); "
      f"runs with any last-pass move: {sum(1 for x in mv if x)} of {len(mv)}")
    P()

# ------------------------------------------------------------------ 4. interval ablation
P("## 4. Check interval (part IV: 34 cells, seeds 0-4, T = 2)")
P()
iv_cells, iv_seeds = cells_of("pr@5", 2), seeds_of("pr@5", 2)
P(HDR)
for c in (1, 2, 5, 10, 20):
    P(row(f"medg_pm@{c} vs pr@{c}", f"medg_pm@{c}", f"pr@{c}", 2, iv_cells, iv_seeds))
for c in (1, 2, 5, 10, 20):
    P(row(f"medg_pm@{c} vs pr@5 (the PR's default)", f"medg_pm@{c}", "pr@5", 2, iv_cells, iv_seeds))
for c in (2, 5, 10, 20):
    P(row(f"medg_pm@{c} vs medg_pm@1", f"medg_pm@{c}", "medg_pm@1", 2, iv_cells, iv_seeds))
for c in (1, 2, 10, 20):
    P(row(f"pr@{c} vs pr@5", f"pr@{c}", "pr@5", 2, iv_cells, iv_seeds))
P(row("medg_pm@ev vs medg_pm@1", "medg_pm@ev", "medg_pm@1", 2, iv_cells, iv_seeds))
P()
n, same, worst = exact_check("medg_pm@ev", "medg_pm@1", 2, iv_cells, iv_seeds)
P(f"- event trigger vs every batch: {same}/{n} runs identical (max relative difference {worst:.2g})")
fc = [(r["n_full_checks"], r["steps"]) for kk, r in DB.items() if kk[0] == "medg_pm@ev"]
fr = np.array([a / b for a, b in fc])
P(f"- event trigger: full checks / batches = {fr.mean():.3f} on average (median {np.median(fr):.3f}, "
  f"max {fr.max():.3f}); a periodic check at c = 5 is 0.200")
P()
P("Runs with more than k/4 empty centres:")
P()
for c in ("1", "2", "5", "10", "20", "ev"):
    for b in ("pr", "medg_pm"):
        if b == "pr" and c == "ev":
            continue
        P(f"- `{b}@{c}`: {runaways(f'{b}@{c}', 2, iv_cells, iv_seeds) or 'none'}")
P()
P("Wall-clock (median paired ratio):")
P()
for c in ("1", "2", "10", "20", "ev"):
    P(f"- medg_pm@{c} vs medg_pm@5: {time_ratio(f'medg_pm@{c}', 'medg_pm@5', 2, iv_cells, iv_seeds)}")
for c in ("1", "2", "5", "10", "20", "ev"):
    P(f"- medg_pm@{c} vs pr@5: {time_ratio(f'medg_pm@{c}', 'pr@5', 2, iv_cells, iv_seeds)}")
P()
P("Per-cell test-error ratios:")
P()
P(cell_table([(f"m@{c}/pr@{c}", f"medg_pm@{c}", f"pr@{c}") for c in (1, 5, 20)] +
             [("m@5/m@1", "medg_pm@5", "medg_pm@1"), ("m@20/m@1", "medg_pm@20", "medg_pm@1")], 2, iv_cells, iv_seeds))
P()

# ------------------------------------------------------------------ 5. GPU
P("## 5. GPU e-means (the PR's torch class at 98deaca) with our rule")
P()
eg_cells = [c for c in cells_of("gpu:pr", 2) if c[1] < 16384]
eg_seeds = seeds_of("gpu:pr", 2)
eg_seeds = [s for s in eg_seeds if s >= 30]
P(f"Grid: {len(eg_cells)} cells x seeds {eg_seeds[0]}-{eg_seeds[-1]}, T = 2, round-2 random starts; "
  f"GPU {Counter(r['gpu'] for kk, r in DB.items() if kk[0].startswith('gpu:'))}.")
P()
P(HDR)
P(row("GPU medg_pm vs GPU pr", "gpu:medg_pm", "gpu:pr", 2, eg_cells, eg_seeds))
P(row("GPU medg_fh vs GPU pr", "gpu:medg_fh", "gpu:pr", 2, eg_cells, eg_seeds))
P(row("GPU pr_eager vs GPU pr", "gpu:pr_eager", "gpu:pr", 2, eg_cells, eg_seeds))
P(row("CPU medg_pm vs CPU pr (round 2, same keys)", "ngp2_full_medg_pm", "pr", 2, eg_cells, eg_seeds))
P(row("GPU pr vs CPU pr (same keys)", "gpu:pr", "pr", 2, eg_cells, eg_seeds))
P(row("GPU medg_pm vs CPU medg_pm (same keys)", "gpu:medg_pm", "ngp2_full_medg_pm", 2, eg_cells, eg_seeds))
P()
P("Per-cell test-error ratios:")
P()
P(cell_table([("GPU m/pr", "gpu:medg_pm", "gpu:pr"), ("CPU m/pr", "ngp2_full_medg_pm", "pr"),
              ("GPU fh/pr", "gpu:medg_fh", "gpu:pr")], 2, eg_cells, eg_seeds))
P()
P("Per-cell training-error ratios:")
P()
P(cell_table([("GPU m/pr", "gpu:medg_pm", "gpu:pr"), ("CPU m/pr", "ngp2_full_medg_pm", "pr"),
              ("GPU fh/pr", "gpu:medg_fh", "gpu:pr")], 2, eg_cells, eg_seeds, m="train_mse"))
P()
P("Runs with more than k/4 empty centres: " + "; ".join(
    f"`{c}` {runaways(c, 2, eg_cells, eg_seeds) or 'none'}" for c in ("gpu:pr", "gpu:pr_eager", "gpu:medg_pm", "gpu:medg_fh")))
P()
rp = Counter((kk[0], r.get("replay")) for kk, r in DB.items() if kk[0] in ("gpu:pr", "gpu:pr_eager", "gpu:medg_pm"))
P(f"CUDA-graph replay chosen (config, replay): {dict(rp)}")
P()
P("Wall-clock on the GPU (median paired ratio):")
P()
P(f"- medg_pm vs pr_eager (same eager driver): {time_ratio('gpu:medg_pm', 'gpu:pr_eager', 2, eg_cells, eg_seeds)}")
P(f"- medg_pm vs pr (graph replay where eligible): {time_ratio('gpu:medg_pm', 'gpu:pr', 2, eg_cells, eg_seeds)}")
P(f"- pr_eager vs pr: {time_ratio('gpu:pr_eager', 'gpu:pr', 2, eg_cells, eg_seeds)}")
P(f"- medg_fh vs medg_pm: {time_ratio('gpu:medg_fh', 'gpu:medg_pm', 2, eg_cells, eg_seeds)}")
rs = [r["rule_seconds"] / r["seconds"] for kk, r in DB.items() if kk[0] == "gpu:medg_pm" and r["seconds"] > 0]
P(f"- medg_pm: host-side rule time / run time = {np.median(rs):.3f} median, {np.max(rs):.3f} max")
P()
P("Per-cell median seconds (pr / pr_eager / medg_pm):")
P()
P("| cell | pr | pr_eager | medg_pm |\n|---|---|---|---|")
for ds, k in eg_cells:
    t = [np.median([get(c, ds, k, s, 2)["seconds"] for s in eg_seeds if get(c, ds, k, s, 2)])
         for c in ("gpu:pr", "gpu:pr_eager", "gpu:medg_pm")]
    P(f"| {ds} {k} | " + " | ".join(f"{x:.2f}" for x in t) + " |")
P()
P("### 5b. Embeddings, k = 16384 (DEEP and SPACEV 1M, seeds 10-12)")
P()
emb = [c for c in cells_of("gpu:pr", 2) if c[1] == 16384]
P("| cell | seed | pr test / train | pr_eager test / train | medg_pm test / train | medg_pm/pr test | train | "
  "s pr / eager / medg_pm | empty pr / medg_pm |")
P("|---|---|---|---|---|---|---|---|---|")
for ds, k in emb:
    for s in seeds_of("gpu:pr", 2):
        a, e, m = get("gpu:pr", ds, k, s, 2), get("gpu:pr_eager", ds, k, s, 2), get("gpu:medg_pm", ds, k, s, 2)
        if not (a and e and m):
            continue
        P(f"| {ds} {k} | {s} | {a['test_mse']:.5g} / {a['train_mse']:.5g} | {e['test_mse']:.5g} / {e['train_mse']:.5g} | "
          f"{m['test_mse']:.5g} / {m['train_mse']:.5g} | {100 * (m['test_mse'] / a['test_mse'] - 1):+.2f}% | "
          f"{100 * (m['train_mse'] / a['train_mse'] - 1):+.2f}% | {a['seconds']:.1f} / {e['seconds']:.1f} / {m['seconds']:.1f} | "
          f"{a['empty']} / {m['empty']} |")
P()

# ------------------------------------------------------------------ 6. o-means
P("## 6. o-means (owlgebra-ai, its count rule) vs o-means with our rule")
P()
og_cells = [c for c in cells_of("gpu:om", 2) if c[1] < 16384]
og_seeds = [s for s in seeds_of("gpu:om", 2) if s < 10]
P(HDR)
P(row("o-means + medg_pm vs o-means", "gpu:om_medg_pm", "gpu:om", 2, og_cells, og_seeds))
P()
P("Per-cell ratios (geometric mean over seeds 0-4):")
P()
P(cell_table([("test", "gpu:om_medg_pm", "gpu:om")], 2, og_cells, og_seeds))
P()
P(cell_table([("train", "gpu:om_medg_pm", "gpu:om")], 2, og_cells, og_seeds, m="train_mse"))
P()
P("Runs with more than k/4 empty centres: " + "; ".join(
    f"`{c}` {runaways(c, 2, og_cells, og_seeds) or 'none'}" for c in ("gpu:om", "gpu:om_medg_pm")))
P()
P("Mean empty centres per cell (om / om_medg_pm): " + "; ".join(
    f"{ds} {k}: {np.mean([get('gpu:om', ds, k, s, 2)['empty'] for s in og_seeds]):.1f} / "
    f"{np.mean([get('gpu:om_medg_pm', ds, k, s, 2)['empty'] for s in og_seeds]):.1f}" for ds, k in og_cells))
P()
P(f"Wall-clock: o-means + medg_pm vs o-means {time_ratio('gpu:om_medg_pm', 'gpu:om', 2, og_cells, og_seeds)}")
P()
P("DEEP 1M, k = 16384:")
P()
P("| seed | om test / train | om+medg_pm test / train | ratio test | train | s om / om+medg_pm | empty om / om+medg_pm |")
P("|---|---|---|---|---|---|---|")
for ds, k in [c for c in cells_of("gpu:om", 2) if c[1] == 16384]:
    for s in seeds_of("gpu:om", 2):
        a, m = get("gpu:om", ds, k, s, 2), get("gpu:om_medg_pm", ds, k, s, 2)
        if a and m:
            P(f"| {s} | {a['test_mse']:.5g} / {a['train_mse']:.5g} | {m['test_mse']:.5g} / {m['train_mse']:.5g} | "
              f"{100 * (m['test_mse'] / a['test_mse'] - 1):+.2f}% | {100 * (m['train_mse'] / a['train_mse'] - 1):+.2f}% | "
              f"{a['seconds']:.1f} / {m['seconds']:.1f} | {a['empty']} / {m['empty']} |")
