# Third confirmatory round: the check interval, and dropping the freeze on the newer driver

**Written on 2026-11-20, before any run at seeds 200–299.** Order of events: second confirmatory round (rule confirmed,
with checks every 5 batches and the freeze of PR #5915 at 98deaca) → exploratory runs (check interval on development
seeds 0–4; the newer driver of PR #5915, commit 79440bc, on seeds 30–49 at 2 epochs and 30–39 at 5 epochs) → the author
asked for stronger evidence before reporting (a) every 5 batches as an empirically validated value for the check interval
and (b) the effect of dropping the freeze on the newer driver → this protocol was drafted with seeds 60–79 → independent
review and a power analysis on the exploratory runs (Section 5) → the design was revised before any run (seeds 200–299,
more seeds, an equivalence test on the mean, fail-closed checks; Section 8) → launch.

Seeds 200–299 have not been used by any run of the project (used: development 0–29, round 1 10–29, round 2 30–49, early
stop 50–59, smoke tests 100–101; checked on both hosts before launch). Our rule keeps the freeze of PR #5915 at
98deaca; it does not drop it. Part F concerns an observation reported in the paper and in the comment on the pull
request, not the rule itself.

## 1. Rules and code

- **The rule**: `ngp2_full_medg_pm` (round 2), unchanged. Part I runs it with a check after every c batches
  (`ngp_interval.py`; c = 5 is the default, identical to round 2), with the C++ per-point sums and matching
  (`NGP_IMPL=cpp`) and the Python decision, on the driver of PR #5915 at 98deaca, as in round 2.
- Part F runs on the driver of PR #5915 at commit 79440bc (FAISS overlay `faiss_head_79440bc`): the PR's own rule
  (`pr_head79440bc`) and our rule on that driver (`ngp_head.NetGainHead`, decision `netgain_rule.decide`) with the pair
  guard, with and without the freeze (no move once less than one pass remains).
- Runners: `interval_grid.py` (Parts G and I; changed for this round only by the flag `--round3`, which admits seeds
  30–31 and 200–299, by a fuller code identity and by the test-data hash) and `prhead_grid.py` through
  `round3_prhead.py` (Part F; `prhead_grid.py` changed only by the fuller code identity, the test-data hash and the
  resume check on it). On-host gate check: `r3_gatecheck.py`. Launcher: `launch_r3.sh`. Analysis: `confirm3_analyze.py`.
  Hashes in Section 7.
- Every record stores the full code identity of round 2 (the rule's modules, the kernels, `grid.py`, `datasets.py`, the
  FAISS build in use with its `e_means` module and compiled libraries, and the Python, NumPy and FAISS versions) plus
  the round-3 files, and the hashes of the training and test data. Before scheduling, a runner hashes the data files of
  every dataset it will run and refuses to start if they differ from their recorded hashes; it refuses to resume over
  records made with other code or other data.

## 2. Data and placement

The 34 cells of round 2 (same data files; both data hashes are checked against the round-2 records), random starts
(sorted `RandomState(seed).permutation(n)[:k]`), the placement of round 2 (Host A: KDD, GeoNames, Power, Shuttle;
Host B: the other ten datasets); compared runs share their host. 8 workers × 4 threads per host, `nice 10`.

## 3. Parts

| Part | Driver | Configurations | Seeds | T | Runs |
|---|---|---|---|---|---|
| G (gate) | 98deaca | medg_pm, check every 5 batches | 30–31 | 2 | 68 |
| I | 98deaca | medg_pm, check every c = 1, 2, 5, 10, 20 batches | 200–299 | 2 | 17,000 |
| F2 | 79440bc | PR rule; ours with freeze; ours without freeze (both with guard) | 200–259 | 2 | 6,120 |
| F5 | 79440bc | as F2 | 200–239 | 5 | 4,080 |

Order on each host: G, then the gate check, then I, F2, F5; each starts only when the previous one ended without
error. If a gate run differs from its round-2 record (test and training error exactly, same host, same data hashes),
the host stops after G. Runs are written atomically, one file each; a stop loses only the runs in progress and a
restart skips finished runs. Runs are not stopped or changed because of results.

## 4. Analysis (`confirm3_analyze.py`)

**Checks before any inference (fail closed).** Gate: all 68 expected gate keys exist and equal the round-2 records of
medg_pm at the same key (same host, test and training error exactly). Every record: key = file name = its fields;
expected configuration, seed and T; host = round-2 placement = the directory it was fetched into; both data hashes =
round 2; finite positive errors. One code identity per part and host, and the same for G and I on a host; this identity
is complete and registered: for G and I it contains round 2's full identity on that host plus the registered hashes of
`interval_grid.py` and `ngp_interval.py`; for F it contains round 2's entries for `grid.py`, `datasets.py`, the compiled
FAISS libraries, the versions and the shared rule modules, the `e_means` module of 79440bc, and the registered hashes of
`prhead_grid.py`, `round3_prhead.py`, `ngp_head.py` and `netgain_rule.py` (Section 7). No duplicate keys. If any check fails the analysis prints the failures and stops, with no estimate or test.

**Unit and estimates.** Unit: the seed. For two configurations and an endpoint (test or training error), L_s = mean over
the 34 cells of log(error_a / error_b) at seed s. Estimand: the mean of L_s (the log of the geometric-mean ratio);
estimate exp(mean L_s) − 1 with a 95% t confidence interval. Also reported: the Hodges–Lehmann pseudomedian with its
exact 95% confidence interval (symmetric location model) and the number of seeds with L_s < 0.

**Difference tests** (as in rounds 1 and 2): two-sided Wilcoxon signed-rank test of L_s (exact without ties for n ≤ 50,
otherwise the normal approximation with continuity correction). This is a test of a shift of the seed-level
distribution (symmetric location model); hence the direction rule: a difference counts only if the Holm-adjusted
p < 0.05 and the mean and the pseudomedian have the same sign (the sign stated in the hypothesis, where one is stated).

**Equivalence tests** (on the estimand above): two one-sided one-sample t-tests of L_s against log(0.99) (alternative:
greater) and log(1.01) (alternative: less); p = the larger of the two; equivalence if the Holm-adjusted p < 0.05, that
is, the confidence interval for the mean at the Holm-adjusted level lies within ±1% (geometric-mean ratio between 0.99
and 1.01). The 90% intervals printed next to these tests are unadjusted; the decisions use the Holm-adjusted p-values.
The t-test targets the mean without assuming symmetry, but its calibration is approximate when L_s is skewed (L_s is a
mean over 34 cells, some heavy-tailed, and the sample has 100 or 40 seeds). A simulation on the exploratory runs
(`confirm3/power/`, Section 5) estimates the rejection rate at the margin at Holm's smallest level: between 0.5 and 1.8
times the nominal level for the interval comparisons, and between 0.7 and 1.5 times for F5–F6. Reporting rule, fixed
before any hypothesis data were analysed: an equivalence with a Holm-adjusted p below 0.025 is reported without
qualification (it would hold under the largest simulated inflation); one with an adjusted p between 0.025 and 0.05 is
reported as equivalent at the 5% level, with this calibration stated next to it. The decisions themselves are unchanged. The same test with
Wilcoxon signed-rank tests is reported as a sensitivity check only. The margin of 1% is a third of the smallest confirmed advantage of the rule over cost-based
revival (2.9% on test error) and was fixed before any run. Degenerate inputs are conservative (no nonzero value or a
non-finite p gives p = 1).

**Family I (one Holm family of 12 hypotheses, α = 0.05; Part I):**
- I1, I2 (test, train): checking every 10 batches gives a **higher** error than every 5.
- I3, I4 (test, train): checking every 20 batches gives a higher error than every 5.
- I5–I8 (test, train for c = 1, then c = 2): checking every c batches is **equivalent** to every 5 within ±1%.
- I9–I12 (test, train for c = 1, then c = 2): checking every c batches gives a **different** error from every 5 (no
  direction stated; the direction rule applies with the observed sign).
- **Reading, stated now:** "every 5 batches is an empirically validated value" if I1–I4 confirm the higher errors at 10
  and 20 batches and I5–I8 confirm equivalence at 1 and 2 batches. If an equivalence is not shown, the matching
  difference test and the confidence interval say whether more frequent checks are better, and by how much; this is
  reported as it is, per hypothesis, together with the running time.
- Descriptive: running time of each c relative to c = 5 (median of the paired ratios), numbers of checks and splits,
  runs with more than a quarter of the centres empty, per-dataset ratios.

**Family F (one Holm family of 8 hypotheses, α = 0.05; Parts F2 and F5):**
- F1, F2 (test, train; 2 epochs): our rule **without** the freeze has a **lower** error than with the freeze.
- F3, F4 (test, train; 2 epochs): our rule without the freeze has a lower error than the PR's own rule.
- F5, F6 (test, train; 5 epochs): our rule without the freeze is **equivalent** to it with the freeze within ±1%.
- F7, F8 (test, train; 5 epochs): our rule without the freeze has a lower error than the PR's own rule.
- Descriptive: running times; moves made in the last pass; runs with more than a quarter of the centres empty;
  per-dataset ratios; for 5 epochs also the unadjusted two-sided difference test of F5–F6's comparisons.

Missing runs: a family is tested only when all its parts are complete; otherwise the analysis reports the missing runs,
withholds the family's tests and exits with code 1.

## 5. Sample size and expectations (from the exploratory runs, stated before running)

The first draft used 20 seeds. A power analysis on the exploratory runs (resampling each cell's observed log ratios, so
that the heavy-tailed cells are kept; Holm worst case α = 0.05/12 or 0.05/8) showed that 20 seeds are too few for the
equivalence tests: the seed-level spread of the test error is 1.6–2.4% (mostly Shuttle with k = 256, where one test
point far from the training data dominates, and KDD Cup 99 with k = 256 for the training error).

| Comparison | Exploratory estimate | Power at the planned n |
|---|---|---|
| I1–I4: every 10 and 20 vs every 5 | +2.7% / +2.1%; +9.3% / +11.6% (test / train) | ≈ 1 (n = 100) |
| I5, I7: every 1 and 2 vs every 5, test, equivalence | −0.66%, −0.70% | 0.94–0.99 if the true difference is 0; 0.13–0.15 if it is the exploratory value |
| I6, I8: same, train | +0.34%, −0.16% | ≈ 1 |
| F1–F4 (2 epochs) | −1.3% / −2.1%; −5.1% / −16.3% | ≈ 1 (n = 60; 0.95–1 already at 20) |
| F5, F6 (5 epochs, equivalence) | +0.05% / −0.34% | ≈ 1 (n = 40) |
| F7, F8 (5 epochs) | −2.6% / −10.4% | 0.98 / 1 (n = 40) |

Expectations: I1–I4 confirmed; I6 and I8 equivalent; I5 and I7 uncertain (if checking every batch truly lowers the test
error by about 0.7%, equivalence will not be shown and I9 or I11 may show the difference); F1–F4 and F7–F8 confirmed;
F5–F6 equivalent.

Calibration of the t-based equivalence tests (`power/power_calibration.txt`; 20,000 resamples; rejection rate when the
true mean of L_s lies exactly on the margin, at Holm's smallest level; Monte Carlo standard error about 0.0005):

| Comparison | Skewness of L_s | Level | Lower margin | Upper margin |
|---|---|---|---|---|
| every 1 vs 5, test | +0.82 | 0.00417 | 0.00195 | 0.00765 |
| every 1 vs 5, train | +0.02 | 0.00417 | 0.00410 | 0.00430 |
| every 2 vs 5, test | +0.84 | 0.00417 | 0.00310 | 0.00655 |
| every 2 vs 5, train | +0.71 | 0.00417 | 0.00205 | 0.00655 |
| F5, without vs with the freeze, test (n = 40) | +0.42 | 0.00625 | 0.00425 | 0.00960 |
| F6, same, train | −0.32 | 0.00625 | 0.00770 | 0.00495 |

The test is too liberal at one margin and too conservative at the other, as expected under skew; the largest excess
is 1.8 times nominal (Section 4 gives the reporting rule that follows from it).

The whole round takes about 3–4 hours per host (estimated from the exploratory runs).

## 6. Review

Independent review of the protocol, the runners' changes, the wrapper, the gate check and the analysis script before
launch (Section 8 records the outcomes).

## 7. Code identity at launch

sha256 prefixes (16 hex digits) of the files as deployed on both hosts (identical on both). The compiled kernels, the FAISS build (98deaca installed; 79440bc overlay) and the Python, NumPy and FAISS versions are recorded in every run record and checked by the analysis.

| file | sha256 |
|---|---|
| `interval_grid.py` | `d8661261e5bb586a` |
| `ngp_interval.py` | `d5d0a9d114bcbcd3` |
| `ngp_variants.py` | `2206ca5608fd6c87` |
| `ngp_variants2.py` | `15999ad7b2e45d9d` |
| `ablation.py` | `bd0f15eb4d0a6641` |
| `ablation_cpp.py` | `bf6af35b14df3026` |
| `ngp_kernels.cpp` | `be826b95e29abbd7` |
| `confirm2_grid.py` | `43c4d1907957f214` |
| `prhead_grid.py` | `bab5ed3e080c41d3` |
| `round3_prhead.py` | `31ae07126f2abf6a` |
| `ngp_head.py` | `69dbabd4318f2ce0` |
| `netgain_rule.py` | `bdb2e0e9f0ee0f14` |
| `r3_gatecheck.py` | `2a1fc09a960615ec` |
| `launch_r3.sh` | `acb918fa4d7b4630` |
| `confirm3_analyze.py` | `b71360c2f93f2fa0` |
| `grid.py` (= round 2) | `4c0db0d4bb71ec55` |
| `datasets.py` (= round 2) | `568dd5ca2acf359d` |

## 8. Events

1. First draft (seeds 60–79, 20 seeds; Wilcoxon equivalence tests).
2. Independent review, round 1 (verdict: fix first). Findings and changes: the interval runner refused seeds 30–99
   (blocker; flag `--round3` added); the Wilcoxon equivalence test concerns the pseudomedian, not the mean that is
   reported (replaced by t-based tests on the mean; Wilcoxon kept as a sensitivity check); a non-finite p could pass an
   equivalence test (degenerate inputs made conservative); the gate did not enumerate its expected keys or check hosts
   (now exact keys, hosts, data hashes and an on-host check that stops the host); provenance failures did not stop the
   inference and the checks were incomplete (now fail closed: duplicates, key and field agreement, placement, both
   data hashes, one code identity per part and host, the same for G and I); the code identity of the runners was
   narrower than in round 2 (now round 2's full identity plus the round-3 files); an incomplete F5 did not fail; the
   descriptive output missed check counts and per-dataset ratios for F, and timing medians were taken on log ratios
   (fixed).
3. Power analysis (Section 5) → seeds 200–299: 100 for I, 60 for F2, 40 for F5; F5 made confirmatory (F5–F8); I9–I12
   added. Before any run.
4. Independent review, round 2 (round-1 findings resolved except provenance). Findings and changes: the recorded data
   hashes were copied from the datasets' metadata (the runners now hash the data files before scheduling and on
   resume); a uniform but incomplete code identity would have passed (the analysis now requires the complete,
   registered identity of Section 7); the standalone gate check accepted an unknown host and missing hashes (fixed);
   the printed 90% intervals are now labelled unadjusted; the robustness claim for the t-test was qualified and the
   simulation archived (`confirm3/power/`).
5. Smoke tests on both hosts in scratch directories (seed 100; and two exploratory keys of the newer driver, which
   reproduced their exploratory records exactly); the scratch directories were deleted. Code deployed (Section 7).
6. Calibration simulation finished (Section 5); the reporting rule for equivalences near the threshold was added to
   Section 4. No change to the analysis code or its hash.
7. Launch on both hosts, with the deployed hashes rechecked against Section 7. The launch was several hours later than
   planned: the script that waited for the calibration simulation matched its own command line and never returned.
   Nothing ran in between. The gate passed on both hosts (all 68 runs identical to round 2 in both errors), and Part I
   started on each host after its own check.
8. Host A stopped responding during Part I, after 4,114 of its 5,000 interval runs, and was rebooted several hours
   later. Its system statistics show memory use rising from 48% to 95% of 188 GB within about 40 minutes, the number
   of processes doubling and the load average reaching about 2,900, which points to another workload; round 3 ran
   the same 8 workers throughout. All records on the host were checked after the reboot (every
   one readable, with positive errors; two interrupted writes had left only temporary files, which the runners ignore,
   so those two runs were redone). The launcher was restarted unchanged (same hashes as Section 7); it re-ran the gate
   check (passed) and resumed Part I with the 886 missing runs, re-verifying the code and data identity of every
   existing record. Host B completed all its parts without interruption.
