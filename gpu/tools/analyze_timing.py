"""Timing table (minimum of the fits per seed, median over the seeds after the first, ratios to a base config).
usage: python analyze_timing.py FILE.jsonl BASE OTHER[,OTHER...] TITLE"""
import collections, json, math, statistics as st, sys
f, base, others, title = sys.argv[1], sys.argv[2], sys.argv[3].split(","), sys.argv[4]
by = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
for line in open(f):
    if line.startswith("{"):
        r = json.loads(line)
        by[(r["dataset"], r["k"])][r["config"]][r["seed"]].append(r["seconds"])
gm = lambda v: math.exp(sum(map(math.log, v)) / len(v))
print(f"### {title}\n")
print("| cell | " + " | ".join([base] + others) + " | " + " | ".join(f"{o} / {base}" for o in others) + " |")
print("|---|" + "---|" * (1 + 2 * len(others)))
ratios = collections.defaultdict(list)
for cell in sorted(by, key=lambda c: (c[1], c[0])):
    v = by[cell]
    seeds = sorted(v[base])[1:] or sorted(v[base])
    med = {c: st.median(min(v[c][s]) for s in seeds if s in v[c]) for c in [base] + others if v.get(c)}
    for o in others:
        if o in med:
            ratios[o].append(med[o] / med[base])
    print(f"| {cell[0]} {cell[1]} | " + " | ".join(f"{med[c]:.3f}" if c in med else "-" for c in [base] + others) +
          " | " + " | ".join(f"{med[o] / med[base]:.2f}" if o in med else "-" for o in others) + " |")
print()
for o in others:
    small = [r for (c, r) in zip(sorted(by, key=lambda c: (c[1], c[0])), ratios[o]) if c[1] < 16384]
    big = [r for (c, r) in zip(sorted(by, key=lambda c: (c[1], c[0])), ratios[o]) if c[1] >= 16384]
    print(f"{o} / {base}: geometric mean {gm(ratios[o]):.2f} over {len(ratios[o])} cells (range {min(ratios[o]):.2f}-"
          f"{max(ratios[o]):.2f}); small cells {gm(small):.2f}" + (f", 1M points {gm(big):.2f}" if big else ""))
print()
