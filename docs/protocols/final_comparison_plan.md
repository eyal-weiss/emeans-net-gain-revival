# Final comparison on the released implementation: analysis plan

**Written on 2026-11-21, after the launch on the first host and before any result of parts F2 or F5 was read.**
Requested by the author: rerun the freeze comparison on the final implementations, with the early stop on (decision
of 2026-11-20, item 3), after choosing no freeze as the default (2026-11-21). Not a new confirmatory round: no
new hypotheses are tested. It documents the released class (`net_gain_revival` version 1.1, the class for the PR's head
driver 79440bc, with the C++ kernels) and estimates the effects with confidence intervals.

## Design

Same 34 cells, random starts, host placement, threads (8 workers × 4 threads, nice 10) and scoring as round 3's Part F,
on the same seeds, so that every run pairs with a round-3 run of the research code:

| Part | Configurations | Seeds | T |
|---|---|---|---|
| R | ours, freeze and no freeze, **without** the early stop | 200–201 | 2 |
| F2 | the PR's rule; ours with the freeze; ours without the freeze (both with the early stop "hl") | 200–259 | 2 |
| F5 | as F2 | 200–239 | 5 |

Runner `final_grid.py`, launcher `final_launch.sh`; code checked before launch with `verify_head.py` (release class,
NumPy and C++, against round 3's research class `ngp_head.NetGainHead` without the early stop and against an early-stop
reference built on it: identical final centres in 128 of 128 comparisons on eight datasets; a second run on eight other
datasets on the other host).

## Checks (fail closed)

- R: every run equals round 3's record of `ngh79440bc_medg_pm_{freeze,nofreeze}_guard` at the same key (test and
  training error exactly).
- F2, F5: every run of the PR's rule equals round 3's record of `pr_head79440bc` at the same key.
- All runs present; one code identity per host; data hashes as in round 3.

## Estimates (as in round 3)

Unit: the seed. L_s = mean over the 34 cells of log(error_a / error_b); estimate exp(mean L_s) − 1 with a 95% t
interval; the number of seeds with L_s < 0; the two-sided Wilcoxon signed-rank p (descriptive, unadjusted). For T = 2
and T = 5, test and training error:

1. ours without the freeze vs ours with the freeze (both with the early stop);
2. ours without the freeze vs the PR's rule; 3. ours with the freeze vs the PR's rule;
4. the early stop's effect: ours with the early stop vs round 3's runs without it (same seeds, freeze and no freeze).

Running time: medians of the paired ratios (same host and session for 1–3; for 4, across sessions, indicative only).
Descriptive: runs where the early stop fired and its step, checks, splits, moves in the last pass, runs with more than a
quarter of the centres empty, per-dataset ratios.

## Note added after the two-epoch part ended (before the five-epoch part did)

The running-time ratio of comparison 4 (the early stop's effect against round 3's runs) is not interpretable: the round-3
runs used the research implementation (statistics in NumPy, decision in Python), about 1.2 to 1.5 times slower than the
released class with its C++ parts, so the ratio mixes the implementation with the early stop. Only its error effects are
reported; the early stop's time saving is not estimated by this design.
