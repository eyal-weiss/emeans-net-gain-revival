# Confirmatory protocol for the posterior net-gain rule `ngp2_full`

*Published version: in the original, decision times were given to the minute, the author's decisions were quoted, and the machines were named. Here the times are replaced by the order of events, the quotes by short summaries and the machine names by Host A and Host B; nothing else was changed.*

**Written on 2026-11-19, before any run of the grid below, after the author had approved its outline.**

## 1. The frozen rule

`ngp2_full` is the config `CONFIGS["ngp2_full"]` in `arxiv/code/experiments/pr5915/ablation.py`. It runs on the e-means implementation of FAISS PR #5915 (commit `98deaca`, built from source in the conda env `emeans_pr5915`), and uses the PR's driver, batch schedule, donor statistics, Oja split axis, Gaussian half-mean split and timing rules.

The revival check runs every 5 batches, as in the library. It removes centre j and splits donor d when the estimated removal cost is below the estimated split gain (w = 1).

- **Cost of removing j:** E[λ_j] · r̃_j · δ_j.
  - E[λ_j] is the posterior mean of j's batch rate. The model is Gamma–Poisson on the counts since j's birth, with a moment-fitted prior whose Poisson part is mean(N/T²).
  - δ_j is the squared distance to the nearest other centre, from an exact search.
  - r̃_j is the normal empirical-Bayes posterior of the ratio between j's mean second-nearest margin and δ_j. The margins come from a two-nearest search on **every row** of the batches in which moves are possible.
- **Gain of splitting d:** the PR's gain (2/π)·|U_d|, with an empirical-Bayes correction.
  - The correction is a log-normal empirical-Bayes posterior.
  - Its noise variance is (κ − 1)/n_eff, with κ the pooled kurtosis of the projections on the split axis.
- **Matching:** greedy, cheapest removal paired with the largest-gain donor (matcher v2), with the PR's pair guard.
- **Cool-down:** a centre whose inherited weight Π(1 − α) since its last split is above ½ is neither removed nor split.

The math note (`netgain_note/note.tex`, rev 2b) documents the same estimators and their approximations. It describes the rule as `ngp4`, which adds three options that `ngp2_full` does **not** use: moderated variances, compound-Poisson gain noise and the no-fallback-donor rule.

The rule is not changed after this protocol. Any later change is new development and needs new seeds.

## 2. Implementations

- **Reference (Python/NumPy on the PR's driver)** = `ablation.py`. It defines the rule. As first planned it was used for every quality result in C1–C6; it is now used in part EQ (amended before launch, Section 8.1).
- **C++** (`ablation_cpp.py` plus a compiled kernel library). Written after this protocol and before any run that uses it, for representative running times.
  - **In C++:** the per-row margin and projection statistics, and the matching loop.
  - **Unchanged:** the search itself (FAISS), the driver and the O(k)-per-check vectorized estimate arithmetic. These are shared with, or of the same kind as, the comparators' implementations.
  - **Used for:** first planned for C7 and TIMING only; amended before launch to every part (Section 8.1).
- **Equivalence check, pre-specified, and fail-closed** (revised in Sections 8.1–8.2): part EQ reruns every `ngp2*` config of the study with the reference implementation on the 16 cells, seeds 10–11, and compares each run with its C++ twin. The configs are `ngp2_full` and the strides s = 2, 4, 8, 16, 32.
  - **The gate passes only if** every pair has the same number of splits, the relative difference in test MSE is ≤ 1e-9, and every C++ record has finite errors.
  - If it fails for a config, that config's C++ results are not used for any confirmatory claim. The config is rerun with the reference implementation in every affected part before analysis, logged as an amendment.
  - DEEP 10M (C7) has no reference twin (cost). This is stated as a limitation; C7 uses the same kernels.
- **Runtime claims:** these rest only on the TIMING part and on C7 (C++, timed per run). The fit times recorded in C1–C6 come from runs that share a host with other jobs and use the reference implementation, so they are not used for runtime claims.

## 3. Comparators

| Config | What it is |
|---|---|
| `lib` | FAISS 1.16 count rule, as reproduced on the PR's driver |
| `pr` | PR #5915 default: cost-based removal, gain donors, axis split, PR timing |
| `ours` | arXiv v1 distance-aware revival, τ = 4, W = 2 |
| `h21` | the PR plus v1's distance test as a safeguard |

All other settings are the library defaults (scaling laws, `batch_law="auto"`).

## 4. Data, seeds and what is new

**Development data for the new rule.** All development runs used random starts and T = 2.
- The 16 cells below, seeds 0–4.
- DEEP/SIFT/SPACEV 1M at k = 16,384, seeds 0–2.

**Never used for the new rule:**
- seeds 10–29 (random starts) and k-means++ starts with seeds 0–9;
- T = 5;
- Covertype. It was the development set of arXiv v1 and is labelled as such.
- DEEP 10M.

**The 16 development cells:**

| Dataset | k |
|---|---|
| KDD | 256, 1024, 4096 |
| Census | 256, 1024, 4096 |
| MSD | 256, 1024, 4096 |
| GeoNames | 256, 1024, 4096 |
| Power | 256, 1024 |
| Shuttle | 256, 1024 |

**k-means++ starts:**
- From the v1 cache: `runs/main/kmpp`, scikit-learn 1.9.1 `kmeans_plusplus` with `random_state=seed`.
- Power and Shuttle are not in that cache. Their starts were computed on 2026-11-19 with the same function (`grid.kmpp_job`) before any run.

Data preprocessing is exactly as in the v1 protocol: deduplicated rows, 20,000 rows held out with seed 1234, test = one half of them, and scaling with training statistics.

## 5. Parts

The priority order is C1, C4, C5, C2, C3, C6. Each run is assigned to the first part in this order that needs it. A run shared by two parts (for example `ngp2_full` in C1 and C5) is run once.

| Part | Cells | Seeds | Start | T | Configs |
|---|---|---|---|---|---|
| **C1, primary** | 16 development cells | 10–29 | random | 2 | lib, pr, ours, h21, ngp2_full |
| C2 | 16 development cells | 0–9 | k-means++ | 2 | as C1 |
| C3 | Covertype z-scored and min-max, k = 256/1024/4096 | 10–29, and 0–9 | random, and k-means++ | 2 | as C1 |
| C4 | DEEP / SIFT / SPACEV 1M, k = 16,384 | 10–19 | random | 2 | as C1 |
| C5, ablation | 16 cells (seeds 10–19), and the 3 embeddings (seeds 10–12) | as listed | random | 2 | 1 row in s for the two-nearest search: s = 1 (`ngp2_full`), 2, 4, 8 (`ngp2`), 16, 32 |
| C6 | 16 development cells | 10–14 | random | 5 | as C1 |
| C7 | DEEP 10M, k = 65,536 | 10–11 | random | 2 | lib, pr, ngp2_full (C++) |
| TIMING | KDD, MSD, GeoNames at k = 4096; Power at k = 1024; DEEP and SPACEV at k = 16,384 | 10–12 | random | 2 | C1's configs plus s = 2, 4, 8, 16, 32 (C++) |

**Execution:**
- Host A runs the datasets stored there; Host B runs Census, Covertype and SIFT, the datasets stored only there. Host B is heavily loaded by other users.
- Host A: 6 workers × 4 threads. Host B: 4 workers × 4 threads. All runs at nice 10.
- C7: a separate process group on Host A, 2 workers × 4 threads, one job per run.
- TIMING: one job at a time, 8 threads, on Host A only, when none of our other jobs runs there. The 1-minute load average is recorded before and after each run.

**Stopping:**
- A hard stop at a fixed time on 2026-11-19, set before launch. Unfinished parts resume later with the same code; the runner skips finished runs. No part is changed because of partial results.
- C7 cap: 90 minutes per run, and a fixed end time for the whole part, before the hard stop. A run that hits the cap is reported as such.
- Runs are not stopped because results look unfavourable.
- Runs are stopped for crashes, bugs or server problems. A fix that does not change the rule is logged as a dated amendment, and the affected runs are rerun.

## 6. Metrics and analysis

**Metrics:**
- **Test MSE** on the held-out rows, the primary metric.
- Train MSE: all training rows for tabular data, the fixed 200k subset for embeddings.
- Cluster-size imbalance, tail error, number of splits.
- Fit time.

**Primary analysis and decision rule (C1)** (revised after the review, Section 8.2):
- **Unit of inference: the seed.** For each seed s = 10…29, R_s is the geometric mean over the 16 C1 cells of the ratio test MSE(`ngp2_full`) / test MSE(`pr`). Runs are paired by seed: same data, same start.
- **Estimate:** the geometric mean of R_s over the 20 seeds (= the geometric mean over all 320 cell-seed pairs).
- **Test:** a two-sided Wilcoxon signed-rank test of log R_s against 0, at α = 0.05.
- **H1 is confirmed if and only if** the estimate is below 1 and p < 0.05.
- **Completion:** H1 is evaluated only when C1 is complete (all 16 cells × 20 seeds × the two configs). Numbers shown before completion are labelled interim, carry no test, and change nothing in the runs.
- **Adverse cells:** per-cell two-sided Wilcoxon tests on the log ratios over the 20 seeds, Holm-corrected over the 16 cells.
  - A cell is "significantly worse" if its Holm-adjusted p < 0.05 and its median ratio > 1.
  - Such cells do not change the H1 decision. They are reported, and any claim in the paper is qualified by them. No "no worse than" claim is made.
- **Descriptive, for comparison with development:** the geometric mean over cells of the per-cell median ratio, the worst cell, and the number of cells better or worse by more than 1%.
- **Wilcoxon details, all tests:**
  - log ratios, `scipy.stats.wilcoxon` with `zero_method="wilcox"` (zero differences, i.e. identical results, are dropped and counted);
  - exact p-values when there are no ties and n ≤ 50, otherwise the normal approximation with continuity correction;
  - a pair with a missing run is dropped and counted.

**Expectation from development, stated before running:**
- On the 8 smoke cells, seeds 0–2: `ngp2_full` −11.5% geo-mean vs `pr`, and `ngp2` −14.2%.
- On the 16 cells, seeds 0–4: `ngp2` −7.5%.
- KDD is the weakest dataset: `ngp2_full` +11–14% vs `pr` at k = 256 and 1024.

**Secondary tests.** These use the same seed-level construction over each part's cells. Together they form one Holm family at α = 0.05, 8 tests:
1. C1 vs lib
2. C1 vs ours
3. C1 vs h21
4. C2 vs pr (10 seeds)
5. C3 random starts vs pr (20 seeds)
6. C3 k-means++ starts vs pr (10 seeds)
7. C4 vs pr (10 seeds)
8. C4 vs lib (10 seeds)

**Descriptive only (no tests):**
- C5: test MSE and fit time as a function of s.
- C6, with 5 seeds: the smallest possible exact two-sided p is 0.0625.
- C7: test MSE and fit time per run.
- TIMING: the median fit-time ratio vs lib and vs pr per cell, with a per-component breakdown.
- Per-cell tables, train MSE, imbalance, tail error and splits for every part.

**Reporting:** every config of every part is reported, including unfavourable ones and incomplete parts (labelled as incomplete).

## 7. Code identity at launch

The sha256 prefixes of `ablation.py` and `confirm_grid.py` at launch are recorded in Section 9 below and in every result record (`code`).

## 8. Changes before any run

### 8.1 C++ implementation for every part (before the review result and before any run)

The author asked that everything relevant to running time be implemented in C++, so that the times are representative. The C++ version was written and tested before launch.

**What it consists of:**
- `ngp_kernels.cpp`, compiled with `g++ -O3 -ffp-contract=off -fopenmp`:
  - fused per-row two-nearest statistics: labels, margins, projections on the split axes, per-cluster sums. Rows are processed in parallel and the sums are accumulated in row order, so results do not depend on the thread count. With stride 1 there is no batch copy.
  - the greedy matching.
- `ablation_cpp.py`: a subclass of the reference class that calls these kernels. It changes only the code paths of the `ng_post2` configs; lib, pr, ours and h21 run the identical reference code.

**Tests before launch:**
- Kernel vs the reference NumPy code on 20 random batches with distance ties: labels, distances and margin sums are identical; projection sums differ by at most 1.8e-14 (relative), which is the summation order.
- Matching vs the reference on 3,000 random instances with exact ties and duplicate centres: identical.
- Whole runs, reference vs C++, seed 10. Configs `ngp2_full`, `ngp2` and `ngp2_s32` on Shuttle 256/1024, Power 256, KDD 256, MSD 1024 and GeoNames 1024: **18 of 18 identical** in test MSE and splits. Plus Census and Covertype on Host B (Section 9).

**Consequences:**
- Section 2 is amended: **every part (C1–C7, TIMING) uses the C++ implementation** (`NGP_IMPL=cpp`). The reference remains the definition of the rule.
- A new part **EQ**, run with the reference implementation after C1–C6 on each host: `ngp2_full` and `ngp2` on the 16 cells (each on the host that runs that cell), seeds 10–11, random starts, T = 2. Every EQ run is compared exactly with its C++ twin in C1/C5.
  - Reported: identical / different counts and the largest relative difference.
  - The 1e-6 criterion of Section 2 applies.
- The fit times recorded in C1–C6 now come from the C++ implementation. They still come from shared hosts, so runtime claims still rest on TIMING and C7.

### 8.2 After the Codex review (`codex_review_confirm/REVIEW.md`, gpt-6.1-sol, high; verdict "fix blockers first")

The author had delegated design decisions arising from the review, to be reported in the morning. All of these changes were made before any confirmatory run.

1. **BLOCKER, C7 escaping the hard stop.** C7 runs now use `timeout --foreground`, so they stay in the stage's process group, and the lane PIDs are waited on explicitly.
   - Tested: a 5-second cap stopped the run with exit code 124 and left no process behind.
   - A cap is recorded only if exit code 124/137 comes after at least 89.8 minutes. Any other failure stops the lane and is logged.
   - The runner now exits non-zero when a run fails, so a failed run is not retried forever.
2. **BLOCKER, no decision rule.** Fixed in Section 6:
   - the seed-level aggregate test, with H1 decided by estimate < 1 and p < 0.05;
   - the treatment of adverse cells;
   - the completion requirement;
   - interim numbers labelled as such.
3. **Statistics.** Fixed in Section 6: log ratios, exact or approximate Wilcoxon, zeros and missing pairs, one Holm family of 8 secondary tests. C6 is descriptive only.
4. **Equivalence gate.** Fail-closed, and extended to every `ngp2*` config (Section 2). The compiled kernel library, its source and the wrapper are hashed in every record.
5. **Disclosure of pre-launch runs at confirmatory keys.** These runs were made only to check correctness. Their values were seen, but no decision depended on them. The grid reruns them with the frozen code; the computation is deterministic, so the values will be the same. They are reported normally.
   - **Smoke test**:
     - Shuttle k = 256, seed 10, T = 5, random start (C6): `ngp2_full` 0.238, `h21` 0.783, `ours` 1.030.
     - Power k = 256, seed 0, k-means++ start (C2): `lib` 0.281, `ngp2_full` 0.234; also `ngp2_s32`, which is in no part.
   - **Equivalence test**, seed 10, random start, T = 2. Only `ngp2*` configs; no comparator was run at these keys.
     - Host A: `ngp2_full`, `ngp2` and `ngp2_s32` on Shuttle 256/1024, Power 256, KDD 256, MSD 1024 and GeoNames 1024 (C1/C5).
     - Host B: `ngp2_full` and `ngp2` on Census 256, Covertype-z 1024 and Covertype-mm 256 (C1, C3, C5).
   - Any later test uses seeds ≥ 100.
6. **C7 failure handling.** As in item 1.
7. **Durable resumption.** Every finished run is written immediately and atomically to `runs/<key>.json`; resumption reads those files. A stop loses at most the runs in progress.
8. **Duplicate launches.** Prevented by one `flock` per stage. TIMING refuses to start while the grid or C7 runs on the host.
9. **Resume identity and time zone.** Resuming refuses if a finished run was made with other code (hashes in each record), unless an amendment allows it explicitly. Stop times carry an explicit time zone.

Not changed: the fixed stop time and later resumption, which the author requested; and C6's 5 seeds, now descriptive.

## 9. Launch record

Launched on 2026-11-19: the grid stage on Host A and Host B (C1, C4, C5, C2, C3, C6 in this priority order, then EQ), and C7 on Host A. TIMING starts on Host A after the grid and C7 stages there have finished, if that is before the hard stop; otherwise later.

| File | sha256 prefix |
|---|---|
| `ablation.py` | bd0f15eb4d0a6641 |
| `confirm_grid.py` | 495df544e1b8b426 |
| `ablation_cpp.py` | bf6af35b14df3026 |
| `ngp_kernels.cpp` | be826b95e29abbd7 |
| `ngp_kernels.so`, Host A (g++ 13.3) | 5bbe78ecb8cdf5e2 |
| `ngp_kernels.so`, Host B (g++ 9.4) | df3e7810f8dc1f43 |
| `launch_confirm.sh` | 98d9a2e2acd8d60b |

Pre-launch equivalence runs on Host B: `ngp2_full` and `ngp2` on Census 256, Covertype-z 1024 and Covertype-mm 256, seed 10 — 6 of 6 identical.

## 10. Amendments during the run

### Amendment 1 — during the run: C7 stopped; TIMING needs a quiet host

**What happened.** Shortly after launch, another user occupied about 66 of Host A's 128 hardware threads at normal priority. Our jobs run at nice 10, so the embedding runs of C4 took about 470 s each instead of the 50–80 s seen in development. Both C7 runs of the frozen rule (DEEP 10M, k = 65,536, seeds 10 and 11) hit the 90-minute cap and were recorded as capped.

**Decision (compute reasons, made without looking at any C7 output; C7 had produced none):**
- C7 was stopped then. The comparator runs (pr, lib) cannot answer C7's question without the frozen rule's runs, would very likely also hit the cap under this load, and would have blocked TIMING for about three more hours.
- C7 is to be rerun later, on a quieter host, with the same code. Its cap is to be decided by the author.

**TIMING.** The automatic start now additionally requires a 1-minute load average below 40 on Host A (`timing_after2.sh`, replacing `timing_after.sh`). Times measured while other users occupy half the host would not be representative. If the condition is not met before the part's deadline, TIMING is postponed to a later session.

### Observation (no change made)

Each worker configured for 4 threads uses up to about 8 cores, because FAISS's OpenMP threads and OpenBLAS's threads both busy-wait. The study's configured threads stay within the 32-thread budget per host, as in all development runs. The actual CPU use was about 48 cores on Host A, including C7. Future runs could set `OMP_WAIT_POLICY=PASSIVE` or `OPENBLAS_NUM_THREADS=1`. They were not changed mid-run, so that the computation of runs within a part stays identical.

### Amendment 2 — later in the run: a finished stage's sleeper blocked TIMING

**What happened.** The grid stage (including EQ) finished with rc = 0. Its hard-stop sleeper (`sleep` until the hard stop) inherited the stage's lock file descriptor and outlived the stage, so `cfm_grid.lock` stayed held. The timing waiter would have waited until its deadline and postponed TIMING, although the host was quiet (load ≈ 21).

**Fix (launcher only; no computation affected).** The orphaned `sleep` (the only process left in the grid stage's process group) was killed. The lock was then free, and TIMING started (load 21.4). For future launches, the sleeper should close fd 9 (`9>&-`) or be killed together with its `sleep` child.

**Verification, not a new result.** Just before that, the five C6 KDD 1024 `ngp2_full` runs (seeds 10–14, T = 5) were rerun with the reference implementation, into a separate directory (`cfm_check_t5`). This was because EQ covers only T = 2 and these runs looked anomalous. All five reproduce the C++ records exactly: test MSE, number of splits and empty centres.
