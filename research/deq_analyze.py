"""Summary of decide_equiv.py records: equivalence of the C++ decision and its runtime effect (descriptive).
usage: python deq_analyze.py <arxiv/results>  > report.md
"""
import glob, json, math, sys
from collections import defaultdict

import numpy as np

ROOT = sys.argv[1]
SUB = sys.argv[2] if len(sys.argv) > 2 else "side_deq"
R = [json.load(open(f)) for f in glob.glob(f"{ROOT}/pr5915/side/*/{SUB}/runs/*.json")]
P = print
P("# C++ decision: equivalence with the round-2 implementation and runtime")
P()
P(f"{len(R)} paired jobs (each: round-2 implementation, then the C++ decision, same key, same process). "
  "Hosts: host_a (no AVX-512), host_b (AVX-512).")
P()
P("## Equivalence")
P()
P("| stat | host | T | jobs | decisions identical at every check | final test/train/splits identical | checks compared | "
  "moves | checks with gains not bit-identical | max relative gain difference | checks with costs not bit-identical |")
P("|---|---|---|---|---|---|---|---|---|---|---|")
g = defaultdict(list)
for r in R:
    g[(r["stat"], r["host"].split(".")[0], r["T"])].append(r)
for key in sorted(g):
    L = g[key]
    P(f"| {key[0]} | {key[1]} | {key[2]} | {len(L)} | {sum(r['decisions_identical'] for r in L)} | "
      f"{sum(r['final_identical'] for r in L)} | {sum(r['checks_compared'] for r in L)} | {sum(r['moves'] for r in L)} | "
      f"{sum(r['checks_gain_not_bitwise'] for r in L)} | {max(r['gain_max_rel'] for r in L):.1e} | "
      f"{sum(r['checks_cost_not_bitwise'] for r in L)} |")
if all("arrays" in r for r in R):
    P()
    P("Per estimate (checks compared, all jobs of a row pooled): checks not bit-identical / largest relative difference "
      "over finite entries / non-finite mismatches; rho and candidate counts.")
    P()
    P("| stat | host | rate | delta | cost | gain | rho not bit-identical | candidate-count mismatches |")
    P("|---|---|---|---|---|---|---|---|")
    g2 = defaultdict(list)
    for r in R:
        g2[(r["stat"], r["host"].split(".")[0])].append(r)
    for key in sorted(g2):
        L = g2[key]
        cell = lambda kk: (f"{sum(r['arrays'][kk]['not_bitwise'] for r in L)} / "
                           f"{max(r['arrays'][kk]['max_rel'] for r in L):.0e} / {sum(r['arrays'][kk]['nonfinite'] for r in L)}")
        P(f"| {key[0]} | {key[1]} | {cell('rate')} | {cell('delta')} | {cell('cost')} | {cell('gain')} | "
          f"{sum(r['rho_not_bitwise'] for r in L)} | {sum(r['ncand_mismatch'] for r in L)} |")
bad = [r["key"] for r in R if not (r["decisions_identical"] and r["final_identical"])]
P()
P(f"Jobs with any difference in decisions or final results: {len(bad)}" + (f": {bad[:10]}" if bad else ""))
P()
P("## Runtime (paired, same process; servers shared with other users)")
P()
P("Decision time = the 'decide' timer minus the nearest-other-centre search, i.e. the per-check estimates and the "
  "matching, for both implementations (records of side_deq, the first grid, subtracted the reference's matching only). "
  "Per dataset, medians over cells, seeds and T.")
P()
P("| dataset | stat | decision ms/check: round 2 -> C++ | speed-up | decision share of run: round 2 -> C++ | "
  "total run time C++ vs round 2 |")
P("|---|---|---|---|---|---|")
h = defaultdict(list)
for r in R:
    h[(r["dataset"], r["stat"])].append(r)
tot = defaultdict(list)
for key in sorted(h):
    L = h[key]
    a = [r["ref"]["decide_own"] / max(r["checks_ref"], 1) * 1e3 for r in L]
    b = [r["cpp"]["decide_own"] / max(r["checks_cpp"], 1) * 1e3 for r in L]
    sa = [r["ref"]["decide_own"] / r["ref"]["secs"] for r in L]
    sb = [r["cpp"]["decide_own"] / r["cpp"]["secs"] for r in L]
    tr = [math.log(r["cpp"]["secs"] / r["ref"]["secs"]) for r in L]
    tot[key[1]] += tr
    P(f"| {key[0]} | {key[1]} | {np.median(a):.2f} -> {np.median(b):.2f} | {np.median(a) / max(np.median(b), 1e-9):.1f}x | "
      f"{100*np.median(sa):.1f}% -> {100*np.median(sb):.1f}% | {100*(math.exp(np.median(tr)) - 1):+.1f}% |")
P()
for st, v in sorted(tot.items()):
    sh = [(r["ref"]["decide_own"] / r["ref"]["secs"], r["cpp"]["decide_own"] / r["cpp"]["secs"]) for r in R if r["stat"] == st]
    P(f"- {st}: total run time, C++ decision vs round 2: median over all jobs {100*(math.exp(np.median(v)) - 1):+.1f}% "
      f"(noisy: shared servers); decision share of run time: median {100*np.median([a for a, _ in sh]):.1f}% -> "
      f"{100*np.median([b for _, b in sh]):.1f}%, max {100*max(a for a, _ in sh):.1f}% -> {100*max(b for _, b in sh):.1f}%")
P()
