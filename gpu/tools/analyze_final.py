"""Analysis of the GPU experiments (gpu_final/): timing (D), accuracy and time on 34 cells (A: e-means, B: o-means),
large scale (C). Exploratory (not pre-registered): estimates with 95% t intervals over seeds and two-sided Wilcoxon p
over seeds; no multiplicity correction.
usage: python analyze_final.py DIR > report.md
"""
import collections, json, math, os, statistics as st, sys

import numpy as np
from scipy import stats

D = sys.argv[1]


def load(name):
    p = os.path.join(D, name)
    if not os.path.exists(p):
        return []
    out = []
    for line in open(p):
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


gm = lambda v: math.exp(sum(map(math.log, v)) / len(v))
pct = lambda x: f"{100 * (x - 1):+.1f}%"
P = print


def timing(name, base, others, title):
    R = load(name)
    if not R:
        return
    by = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    for r in R:
        by[(r["dataset"], r["k"])][r["config"]][r["seed"]].append(r["seconds"])
    P(f"### {title}\n")
    P("Seconds per fit: minimum over 3 fits per seed (contention only adds time), median over the seeds after the "
      "first (which carries compilation).\n")
    P("| cell | " + " | ".join([base] + others) + " | " + " | ".join(f"{o} / {base}" for o in others) + " |")
    P("|---|" + "---|" * (1 + 2 * len(others)))
    ratios = collections.defaultdict(list)
    for cell in sorted(by, key=lambda c: (c[1], c[0])):
        v = by[cell]
        seeds = sorted(v[base])[1:] or sorted(v[base])
        med = {c: st.median(min(v[c][s]) for s in seeds if s in v[c]) for c in [base] + others if v.get(c)}
        if base not in med:
            continue
        for o in others:
            if o in med:
                ratios[o].append(med[o] / med[base])
        P(f"| {cell[0]} {cell[1]} | " + " | ".join(f"{med[c]:.3f}" if c in med else "-" for c in [base] + others) +
          " | " + " | ".join(f"{med[o] / med[base]:.2f}" if o in med else "-" for o in others) + " |")
    for o in others:
        if ratios[o]:
            P(f"\n{o} / {base}: geometric mean over cells {gm(ratios[o]):.2f} (range {min(ratios[o]):.2f}-{max(ratios[o]):.2f})")
    P()


def accuracy(name, base, ours, title):
    R = load(name)
    if not R:
        return
    by = collections.defaultdict(dict)
    for r in R:
        by[(r["dataset"], r["k"], r["seed"])][r["config"]] = r
    cells = sorted({(d, k) for d, k, _ in by})
    seeds = sorted({s for _, _, s in by})
    P(f"### {title}\n")
    complete = [c for c in cells if all(base in by.get((c[0], c[1], s), {}) and ours in by.get((c[0], c[1], s), {})
                                       for s in seeds)]
    P(f"{len(complete)} complete cells of {len(cells)}, seeds {seeds[0]}-{seeds[-1]} ({len(seeds)}).\n")
    for m, lab in (("test_mse", "test error"), ("train_mse", "training error")):
        L = []
        for s in seeds:
            v = [math.log(by[(c[0], c[1], s)][ours][m] / by[(c[0], c[1], s)][base][m]) for c in complete]
            L.append(np.mean(v))
        L = np.array(L)
        mean = L.mean()
        h = stats.t.ppf(0.975, len(L) - 1) * L.std(ddof=1) / math.sqrt(len(L)) if len(L) > 1 else float("nan")
        p = stats.wilcoxon(L).pvalue if len(L) > 1 and np.any(L != 0) else float("nan")
        P(f"- {lab}, {ours} vs {base}: {pct(math.exp(mean))} [{pct(math.exp(mean - h))}, {pct(math.exp(mean + h))}], "
          f"lower on {int((L < 0).sum())}/{len(L)} seeds, Wilcoxon p = {p:.2g}")
    T = []
    for c in complete:
        T.append(st.median(by[(c[0], c[1], s)][ours]["seconds"] / by[(c[0], c[1], s)][base]["seconds"]
                           for s in seeds[1:] or seeds))
    P(f"- time {ours} / {base} (median over seeds per cell, geometric mean over cells): {gm(T):.2f} "
      f"(range {min(T):.2f}-{max(T):.2f}; scored runs, a shared GPU: see part D for clean timing)\n")
    P(f"| cell | test {ours}/{base} | train {ours}/{base} | time {ours}/{base} | empty {base} / {ours} |")
    P("|---|---|---|---|---|")
    for c in complete:
        lt = gm([by[(c[0], c[1], s)][ours]["test_mse"] / by[(c[0], c[1], s)][base]["test_mse"] for s in seeds])
        lr = gm([by[(c[0], c[1], s)][ours]["train_mse"] / by[(c[0], c[1], s)][base]["train_mse"] for s in seeds])
        tt = st.median(by[(c[0], c[1], s)][ours]["seconds"] / by[(c[0], c[1], s)][base]["seconds"] for s in seeds[1:] or seeds)
        e = lambda cfg: st.median(by[(c[0], c[1], s)][cfg].get("empty", 0) for s in seeds)
        P(f"| {c[0]} {c[1]} | {pct(lt)} | {pct(lr)} | {tt:.2f} | {e(base):.0f} / {e(ours):.0f} |")
    P()


def contention():
    p = os.path.join(D, "contention.log")
    sp = os.path.join(D, "stage.txt")
    if not os.path.exists(p):
        return
    rows = [l.split() for l in open(p) if l.strip()]
    rows = [(int(t), int(u.split(",")[0]), int(u.split(",")[1]), int(n)) for t, u, n in rows if "," in u]
    P("### GPU sharing during the runs\n")
    util = [r[1] for r in rows]
    others = [r[3] for r in rows]
    P(f"{len(rows)} samples (every 5 s): GPU utilization median {st.median(util)}%, mean {np.mean(util):.0f}%; "
      f"samples with more than one process on the GPU (ours plus another user's): "
      f"{100 * np.mean([o > 1 for o in others]):.0f}%.\n")
    if os.path.exists(sp):
        P("Stages:\n\n```\n" + open(sp).read().strip() + "\n```\n")


P("# GPU experiments: the optimized net-gain rule in GPU e-means and in o-means\n")
P("Exploratory (not pre-registered). One RTX 3090 Ti (host_a), shared with another user's training job during the "
  "runs (see the last section). Seeds with random starts as in round 2; 2 epochs.\n")
P("## D. Timing\n")
timing("D_timing_em.jsonl", "pr", ["ng", "poc"], "GPU e-means (pr: the PR's GPU rule with CUDA graphs; ng: ours, "
       "optimized; poc: our first, unoptimized port)")
timing("D_timing_om.jsonl", "om", ["om_ng", "om_poc"], "o-means (om: o-means' own rule; om_ng: ours, optimized; "
       "om_poc: unoptimized port)")
P("## A. GPU e-means: 34 cells, 10 seeds\n")
accuracy("A_emeans.jsonl", "pr", "ng", "ours (ng) vs the PR's GPU rule (pr)")
P("## B. o-means: 34 cells, 10 seeds\n")
accuracy("B_omeans.jsonl", "om", "om_ng", "ours (om_ng) vs o-means' rule (om)")
P("## C. Large scale (k = 16,384)\n")
accuracy("C_large_em.jsonl", "pr", "ng", "GPU e-means")
accuracy("C_large_om.jsonl", "om", "om_ng", "o-means")
contention()
