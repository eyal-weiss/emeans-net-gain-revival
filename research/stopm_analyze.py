"""Analysis of the move-phase measurement (stopmeasure_grid.py records; exploratory, descriptive).

For each run: the move phase = the checks before the freeze. Per check, the predicted net gain of the accepted moves
(sum of w*gain - cost, in squared-error units per batch) relative to the last batch's squared error (obj).
Offline stop rules (exact up to the stop point, since the run is identical until then; after it the run would make
no more moves):
  HL      : stop once one half-life of the EMA (prod(1 - alpha) since the last accepted move <= 1/2) passes with no move
  V(eps)  : stop at the first check from which the predicted net gain stays below eps * obj for one half-life
For each rule: the share of the move phase left (in batches), the moves and predicted net gain after the stop (what
would be missed), and the time saved = move-phase machinery after the stop (two-nearest search beyond a plain search,
margin and axis statistics, removal statistics, decision incl. delta search) as a share of the run's total time.
usage: python stopm_analyze.py <arxiv/results>  > report.md
"""
import glob, json, math, sys
from collections import defaultdict

import numpy as np

ROOT = sys.argv[1]
recs = [json.load(open(f)) for f in glob.glob(f"{ROOT}/pr5915/side/*/side_stopm/runs/*.json")]
for r in recs:     # alpha0 from the round-2 record of the same key (the decay product underflows in long runs)
    o = glob.glob(f"{ROOT}/pr5915/confirm2/*/cfm2_grid/runs/cfm2__{r['config']}__{r['dataset']}__k{r['k']}__s{r['seed']}"
                  f"__random__T{r['T']}.json")
    r["alpha0"] = json.load(open(o[0]))["alpha0"]


# C++ decision: per (dataset, k, T), median ratio of the decision's own time, C++ / round-2 (decide_equiv records)
_F = {}
for f in glob.glob(f"{ROOT}/pr5915/side/*/side_deq/runs/*.json"):
    q = json.load(open(f))
    if q["stat"] == "medg_pm" and q["ref"]["decide_own"] > 0:
        _F.setdefault((q["dataset"], q["k"], q["T"]), []).append(q["cpp"]["decide_own"] / q["ref"]["decide_own"])
FRATIO = {kk: float(np.median(v)) for kk, v in _F.items()}


def log_decay(r, s0, s1):
    """log prod_{s0 < t <= s1} (1 - alpha_t), alpha_t = the PR's cosine schedule (reproduces the logged decay_prod);
    -inf if a step has alpha >= 1 (the statistics are replaced)."""
    T, a0 = r["total_steps"], r["alpha0"]
    al = lambda t: a0 * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(t, T) / max(1, T))))
    out = 0.0
    for t in range(s0 + 1, s1 + 1):
        a = al(t)
        if a >= 1.0:
            return -math.inf
        out += math.log(1.0 - a)
    return out
MOVE_KEYS = ("search_sampled_top2", "margin_stats", "accumulate_removal_stats", "update_removal_axes", "decide")


def analyse(r):
    ch = [c for c in r["checks"] if not c["frozen"]]
    if not ch:
        return None
    tot_steps = r["total_steps"]
    first_frozen_step = next((c["step"] for c in r["checks"] if c["frozen"]), tot_steps)
    move_steps = first_frozen_step                      # batches with moves allowed (approx.: up to the first frozen check)
    final_t = r["times"]
    # plain-search cost per row, from the frozen part (search_top1 is only used there)
    frozen_rows = r["n"] * r["T"] - ch[-1]["rows"]
    top1_per_row = final_t.get("search_top1", 0.0) / max(frozen_rows, 1)

    fr = FRATIO.get((r["dataset"], r["k"], r["T"]))

    def machinery(c, cpp=False):                        # cumulative move-phase extra time up to check c
        t = c["times"]
        m = sum(t.get(kk, 0.0) for kk in MOVE_KEYS) - top1_per_row * c["rows"]
        if cpp:                                         # the decision's own time scaled to the C++ decision
            own = t.get("decide", 0.0) - t.get("delta_search", 0.0) - t.get("match", 0.0)
            m -= (1.0 - fr) * own
        return m

    last = ch[-1]
    mach_end = machinery(last)
    net = np.array([c.get("net_sum", 0.0) for c in ch])
    rel = np.array([(c.get("net_sum", 0.0) / c["obj"]) if c["obj"] else 0.0 for c in ch])
    moves = np.array([c["moves"] for c in ch])
    steps_ = [c["step"] for c in ch]

    def stop_at(cond_quiet):
        """first check index i such that the window from the start of the current quiet stretch has lasted one
        half-life; cond_quiet(i) says check i is 'quiet'."""
        start = None
        for i in range(len(ch)):
            if cond_quiet(i):
                if start is None:
                    start = i
                s0 = steps_[start - 1] if start > 0 else -1
                if log_decay(r, s0, steps_[i]) <= math.log(0.5):
                    return i
            else:
                start = None
        return None

    own_end = final_t.get("decide", 0.0) - final_t.get("delta_search", 0.0) - final_t.get("match", 0.0)
    secs_cpp = r["seconds"] - (1.0 - fr) * own_end if fr is not None else float("nan")
    mach_end_cpp = machinery(last, True) if fr is not None else float("nan")
    out = dict(checks=len(ch), moves=int(moves.sum()), net=float(net.sum()), mach_share=mach_end / r["seconds"],
               mach_share_cpp=mach_end_cpp / secs_cpp,
               last_move=(int(np.flatnonzero(moves)[-1]) + 1) / len(ch) if moves.any() else 0.0)
    rules = {"HL": lambda i: moves[i] == 0}
    for eps in (1e-2, 1e-3, 1e-4):
        rules[f"V{eps:g}"] = (lambda e: (lambda i: rel[i] < e))(eps)
    for name, cond in rules.items():
        i = stop_at(cond)
        if i is None:
            out[name] = dict(stopped=False, phase_left=0.0, moves_missed=0.0, net_missed=0.0, saved=0.0, saved_cpp=0.0)
            continue
        left = (move_steps - ch[i]["step"]) / max(move_steps, 1)
        out[name] = dict(stopped=True, phase_left=left,
                         moves_missed=float(moves[i + 1:].sum()) / max(moves.sum(), 1),
                         net_missed=float(net[i + 1:].sum()) / max(net.sum(), 1e-300),
                         saved=(mach_end - machinery(ch[i])) / r["seconds"],
                         saved_cpp=((mach_end_cpp - machinery(ch[i], True)) / secs_cpp) if fr is not None else float("nan"))
    # where the moves happen: share of all moves made in the first 10/25/50% of the move phase
    steps = np.array([c["step"] for c in ch])
    for q in (0.1, 0.25, 0.5):
        out[f"moves_by_{q}"] = float(moves[steps <= q * move_steps].sum()) / max(moves.sum(), 1)
        out[f"net_by_{q}"] = float(net[steps <= q * move_steps].sum()) / max(net.sum(), 1e-300)
    return out


by = defaultdict(list)
for r in recs:
    a = analyse(r)
    if a is not None:
        by[(r["T"], r["dataset"])].append(a)

P = print
P("# Move-phase measurement: when do the moves stop paying, and what would an early stop save?")
P()
P(f"{len(recs)} runs (medg_pm, round-2 keys; every final result identical to its round-2 record). EXPLORATORY.")
P()
for T in (2, 5):
    P(f"## T = {T}")
    P()
    P("| dataset | checks | move-phase machinery / run time | moves in first 10% / 25% / 50% of the move phase | "
      "last move at (share of phase) | HL: stops? phase left / moves missed / net missed / time saved | "
      "V(1e-3): stops? phase left / moves missed / net missed / time saved |")
    P("|---|---|---|---|---|---|---|")
    allr = []
    for (t, ds), L in sorted(by.items()):
        if t != T:
            continue
        allr += L
        f = lambda key: np.median([x[key] for x in L])

        def rule(nm):
            st = sum(x[nm]["stopped"] for x in L)
            g = lambda kk: np.median([x[nm][kk] for x in L])
            return (f"{st}/{len(L)}; {100*g('phase_left'):.0f}% / {100*g('moves_missed'):.1f}% / "
                    f"{100*g('net_missed'):.1f}% / {100*g('saved'):.1f}%")
        P(f"| {ds} | {f('checks'):.0f} | {100*f('mach_share'):.0f}% | {100*f('moves_by_0.1'):.0f}% / "
          f"{100*f('moves_by_0.25'):.0f}% / {100*f('moves_by_0.5'):.0f}% | {100*f('last_move'):.0f}% | {rule('HL')} | "
          f"{rule('V0.001')} |")
    P()
    P("All cells (medians over runs):")
    P()
    for nm in ("HL", "V0.01", "V0.001", "V0.0001"):
        st = sum(x[nm]["stopped"] for x in allr)
        g = lambda kk: np.median([x[nm][kk] for x in allr])
        m = lambda kk: np.mean([x[nm][kk] for x in allr])
        P(f"- {nm}: stops in {st}/{len(allr)} runs; median phase left {100*g('phase_left'):.0f}%, moves missed "
          f"{100*g('moves_missed'):.1f}% (mean {100*m('moves_missed'):.1f}%), predicted net gain missed "
          f"{100*g('net_missed'):.1f}% (mean {100*m('net_missed'):.1f}%), time saved {100*g('saved'):.1f}% "
          f"(mean {100*m('saved'):.1f}%); with the C++ decision: time saved median {100*g('saved_cpp'):.1f}% "
          f"(mean {100*m('saved_cpp'):.1f}%)")
    P(f"- move-phase machinery share of run time: median {100*np.median([x['mach_share'] for x in allr]):.0f}% "
      f"(with the C++ decision: {100*np.median([x['mach_share_cpp'] for x in allr]):.0f}%)")
    P()
