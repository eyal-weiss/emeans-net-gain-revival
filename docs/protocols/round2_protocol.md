# Second confirmatory round: `medg_pm` (and `medg_fh`) against PR #5915

*Published version: in the original, decision times were given to the minute, the author's decisions were quoted, and the machines were named. Here the times are replaced by the order of events, the quotes by short summaries and the machine names by Host A and Host B; nothing else was changed.*

**Written on 2026-11-19, before any run at seeds 30–49.** Order of events: first confirmatory round (not confirmed) →
diagnosis and exploratory variants on development seeds and eight added datasets → follow-up report with the results
and a recommended evaluation design → the author chose medg_pm as the primary rule, with medg_fh as the reported
alternative, and asked for the tests to start → this protocol was written → launch. The evaluation design below
(co-primary test and training error, Holm over the two, per-dataset tables) is the one recommended in the follow-up
report (not released; its results are summarized in the technical report). The author did not discuss it separately; it
was reported to them at launch and could be changed only before any result of this round had been seen.

## 1. The rules under test

- **`ngp2_full_medg_pm`** (primary) and **`ngp2_full_medg_fh`** (alternative): the frozen rule `ngp2_full` with a
  different estimate of the split gain. Specification: `VARIANTS_MATH.md` in this folder. Code:
  `code/experiments/pr5915/ngp_variants.py` and `ngp_variants2.py` on top of `ablation.py` / `ablation_cpp.py`.
- The variants were designed after the first confirmatory round (seeds 10–29, k-means++ seeds 0–9, T = 5 seeds 10–14)
  and after exploratory runs on the additional datasets (seeds 0–19). They are frozen by this protocol; any later change
  is new development.
- **Implementation:** as in round 1. The per-row statistics and the matching are in C++ (`ablation_cpp.py`,
  `ngp_kernels.cpp`), the rest in Python on the PR's driver. The reference implementation (`NGP_IMPL=py`) defines the
  rule; part EQ checks equality (Section 6.4).

## 2. Comparators

`pr` (PR #5915 default; the reference for every test), `lib` (FAISS 1.16 count rule on the PR's driver), `ours` (arXiv v1
distance-aware revival), `h21` (PR plus v1's distance test), and the frozen `ngp2_full`. Same code as in round 1.

## 3. Data

34 cells: the 16 development cells of round 1 and the 18 additional cells of the exploratory follow-up.

| Dataset | k | Dataset | k |
|---|---|---|---|
| KDD (z) | 256, 1024, 4096 | MNIST (/255) | 256, 1024 |
| Census (z) | 256, 1024, 4096 | Fashion-MNIST (/255) | 256, 1024 |
| MSD (z) | 256, 1024, 4096 | Corel (z) | 256, 1024 |
| GeoNames | 256, 1024, 4096 | 3D Road (z) | 256, 1024, 4096 |
| Power (z) | 256, 1024 | cod-rna (z) | 256, 1024, 4096 |
| Shuttle (z) | 256, 1024 | IJCNN1 (z) | 256, 1024 |
| | | SensIT (z) | 256, 1024 |
| | | HIGGS subsample (z) | 256, 1024 |

- Preparation as in round 1 (`datasets.py` `_finish`): duplicates removed, 20,000 rows held out with seed 1234, the test
  set is one half of them, scaling with training statistics.
- **One change from the exploratory runs:** MNIST and Fashion-MNIST are scaled by 1/255 (`mnist_255`, `fashion_255`)
  instead of per-pixel min-max. Per-pixel min-max stretches rarely lit pixels to [0, 1] and made a few test images
  artificial outliers (10–12% of MNIST's test error). No run of any rule has been made on the /255 versions except the
  smoke tests of Section 10.
- Data hashes (train / test sha256 prefixes), identical on both hosts: `mnist_255` 00de5c7b0edd1805 / a37353e1eae19343;
  `fashion_255` b2f38330dd457bfb / 69881c7d5fc6ee5c. The other datasets are those of round 1 and of the follow-up;
  every record stores its dataset's train hash.

## 4. Seeds

Seeds 30–49 have never been used by any run of this project (random starts or k-means++). Smoke tests use seeds ≥ 100 or
earlier seeds whose results are already known; the runner refuses seeds 30–99 outside the parts below.

## 5. Parts, execution, stopping and resuming

| Part | Cells | Seeds | Start | T | Configs |
|---|---|---|---|---|---|
| **P, primary** | 34 | 30–49 | random | 2 | lib, pr, ours, h21, ngp2_full, medg_pm, medg_fh |
| S5 | 34 | 30–39 | random | 5 | as P |
| SK | 34 | 30–39 | k-means++ | 2 | as P |
| EQ2 | 34 | 30–31 | random | 2 | medg_pm, medg_fh, reference implementation |
| EQ5 | 34 | 30 | random | 5 | medg_pm, medg_fh, reference implementation |

9,520 runs in P, S5 and SK, plus 136 in EQ2 and 68 in EQ5.

- **Starts.** Random: k distinct training rows, the indices `RandomState(seed).permutation(n)[:k]` sorted in increasing
  order (as in round 1 and every earlier run; the order of the centres can matter for ties). k-means++:
  scikit-learn 1.9.1 `kmeans_plusplus(X, k, random_state=seed)` (`grid.kmpp_job`, as in round 1), computed before the
  SK runs in the environment that has scikit-learn and stored in `kmpp/`; every SK record stores the start's hash.
- **Placement** (fixed): Host A runs KDD, GeoNames, Power and Shuttle; Host B runs Census, MSD and the 8 additional
  datasets. Every run of a (cell, seed, start, T) is made in one job on one host, so the compared runs always share a
  host. 8 workers × 4 threads per host (32 threads), nice 10, `OMP_WAIT_POLICY=PASSIVE`.
- **Order.** P, EQ2, S5, EQ5, the missing k-means++ starts, SK; each step starts only if the previous one ended without
  error. EQ2 comes right after P so that the primary decision does not wait for the other parts. Within a part, seed by
  seed (largest cells first within a seed), so a stopped part holds complete seeds except the last few.
- **Stopping and resuming.** The author may stop the round at any time and resume it later (`ctl.sh stop` /
  `ctl.sh start`). Every finished run is written immediately and atomically to its own file; a stop loses only the runs
  in progress, and resuming skips the finished runs and continues in the same order. Resuming refuses if any finished run
  was made with other code or another environment (this project's code, the installed FAISS build's e_means module and
  compiled libraries, the Python and NumPy versions), on another host, from other data (train and test hashes) or from
  another k-means++ start; every (re)start also hashes the actual data files and refuses if they differ from the hashes
  recorded at preparation. Runs are deterministic on a given host, so a stop and resume does not change any result.
- **No early look.** Progress is reported as counts of finished runs only. The analysis script prints a part's
  estimates and tests only when the part is complete; an interim look (`--interim`) happens only if the author asks, is
  logged in Section 11 and carries no decision. Runs are not stopped, and no part is changed, because of results.
- Runs are stopped for crashes, bugs or server problems. A fix that does not change any rule is logged as a dated
  amendment and the affected runs are rerun.

## 6. Analysis (`code/experiments/pr5915/confirm2_analyze.py`, hashed in Section 9)

Unit of inference: the seed. For a part, two configs a and b, and an endpoint (test MSE or training MSE), L_s is the
mean over the part's cells of log(endpoint(a) / endpoint(b)) at seed s (runs paired by cell and seed: same data, same
start). The estimate is exp(mean_s L_s) − 1 (the geometric mean ratio minus 1). The test is a two-sided Wilcoxon signed-rank
test of L_s against 0; its own location estimate, the Hodges–Lehmann pseudomedian of L_s (median of the Walsh
averages), is reported next to the estimate:
`scipy.stats.wilcoxon`, `zero_method="wilcox"` (zeros dropped and counted), exact p without ties and n ≤ 50, otherwise
the normal approximation with continuity correction; p = 1 when a single nonzero L_s remains. A pair with a missing run
is dropped and counted.

**Direction rule (all tests).** A result is significant in a direction only if its (Holm-adjusted) p < 0.05 **and** the
geometric-mean estimate and the pseudomedian are on the same side of 0. If they disagree, the result is "direction
ambiguous" and counts as not significant. (Added after the Codex review: the mean of the L_s and the signed ranks can
point in opposite directions when one seed is extreme.)

### 6.1 Primary (part P; decided only when P is complete and medg_pm's EQ2 gate has passed)

- H1-test: medg_pm has lower test MSE than pr. H1-train: medg_pm has lower training MSE than pr (all training rows).
- The two p-values are Holm-adjusted together (α = 0.05). Each hypothesis is **confirmed** iff its adjusted p < 0.05
  and both the estimate and the pseudomedian are below 0; **refuted** iff its adjusted p < 0.05 and both are above 0;
  otherwise not confirmed.
- **Adverse cells.** Per cell, a two-sided Wilcoxon over the 20 seeds on the log ratios, Holm over the 34 cells, for each
  endpoint. A cell is "significantly worse" if its adjusted p < 0.05, its median ratio > 1 and its pseudomedian of log
  ratios > 0 (better: both below; otherwise direction ambiguous). Reported only when P is complete and medg_pm's EQ2
  gate has passed. Such cells do not change the decision; they are reported, and any claim is qualified by them. No
  "no worse than" claim is made.
- Training MSE is in-sample (the objective k-means minimises); test MSE is on held-out rows and, on Shuttle and KDD, is
  dominated by a few test points far from all training data (follow-up report). This is why both are primary.

### 6.2 Secondary: one Holm family of 10 tests (α = 0.05), evaluated when P, S5 and SK are complete and the gates medg_pm EQ2, medg_fh EQ2 and medg_pm EQ5 have passed

1. P: medg_fh vs pr, test. 2. P: medg_fh vs pr, train. 3. P: medg_pm vs frozen ngp2_full, test.
4. P: medg_pm vs lib, test. 5. P: medg_pm vs ours, test. 6. P: medg_pm vs h21, test.
7. S5: medg_pm vs pr, test. 8. S5: medg_pm vs pr, train. 9. SK: medg_pm vs pr, test. 10. SK: medg_pm vs pr, train.

Each is reported as significantly lower, significantly higher, direction ambiguous, or not significant (two-sided,
Holm-adjusted). SK uses k-means++ starts with the same code as P, so it relies on the T = 2 gate.

### 6.3 Descriptive (no tests; unadjusted p-values where shown)

For each of P, S5 and SK, for every config: against pr and against medg_pm; the 16 original and 18 additional cells
separately; a per-dataset table (geometric mean ratio, and the dataset's contribution to the 34-cell log estimate);
leave-one-dataset-out estimates (including without Shuttle); per-cell median ratios; **stability**: the number of runs
with more than a quarter of the centres empty at the end, per config; fit-time ratios to pr (shared hosts; not a timing
study).

### 6.4 Equivalence gate (EQ2, EQ5)

For each variant and each EQ part, every registered key must have both its reference run (EQ) and its C++ twin (P or
S5). The gate passes only if every pair has the same number of splits, finite test and training MSE in both
implementations, and relative differences ≤ 1e-9 in both. Otherwise it is FAIL (a pair differs) or INCOMPLETE (a pair is
missing). **No decision that involves a variant is issued until its gate has passed**: the primary needs medg_pm EQ2;
the secondary family needs medg_pm EQ2, medg_fh EQ2 and medg_pm EQ5. If a gate fails, that variant's C++ results are
not used for any confirmatory claim until rerun with the reference implementation (logged as an amendment).

### 6.5 Provenance checks (the analysis refuses to run otherwise)

No duplicate key; every record's fields match its key; one code/environment identity per implementation and host (C++ for P, S5,
SK; reference for EQ); every record made on its dataset's registered host with the registered implementation and in the
part its key belongs to; one data identity (train and test
hashes) per dataset; one start per (dataset, k, seed) for SK. The laptop's copy mirrors the servers exactly
(`ctl.sh analyze` fetches with deletions and does not analyse after a failed transfer).

## 7. Expectations from the exploratory runs (stated before running)

Development seeds, so not confirmation:

| Group | medg_pm vs pr, test | medg_pm vs pr, train |
|---|---|---|
| 16 original cells, seeds 10–29 (C1 keys) | −4.8% (p 0.024) | −21.8% |
| 18 additional cells (min-max MNIST/Fashion), seeds 0–19 | −0.1% | −0.3% |
| 34 cells, combined as in Section 6 (equal cell weights) | about −2.3% | about −11% |
| 16 original cells without Shuttle | −0.3% | about −16% |

- The test endpoint rests mostly on Shuttle: in the exploratory runs Shuttle contributed −4.7 of the −4.8 points on the
  original cells. The training endpoint is broad. KDD is the weakest dataset (+0.4 points at T = 2 in exploration).
- T = 5 (C6 keys, 16 cells, 5 seeds): medg_pm −7.3% test, −13.2% train; the frozen rule +8.1% test with runaway runs on
  KDD 1024. k-means++ starts (C2 keys): medg_pm +8.9% test (one isolated Shuttle test point), −36% train.
- Stability (runs ending with more than a quarter of the centres empty, over all round-1 and exploratory records):
  medg_pm 0 of 135 on KDD and 0 of 1,400 on the other datasets except Shuttle; medg_fh 0 of 135 and 0 of 1,335. On
  Shuttle such runs are common for every rule (pr 47 of 100, h21 35 of 90, frozen 13 of 100, medg_pm 7 of 90, medg_fh
  10 of 90), so there the criterion reflects the data, not a runaway. The frozen rule: 22 of 150 on KDD.

## 8. Reproducibility

Runs are bit-for-bit deterministic on a given host (round 1: 24 of 24 reruns identical; this round's smoke tests: 67 of
67 earlier records reproduced on the same host). **Across the two hosts, results can differ on KDD** (all 4 rules
compared at seed 13 differed in the last digits or more; Shuttle, Corel and cod-rna reproduced across hosts), presumably
through floating-point differences in the distance kernels on the two CPUs, which KDD's near-duplicate rows amplify. This
does not affect the comparisons, because compared runs always share a host (Section 5), but a dataset's runs must not
move to the other host during the round without an amendment.

## 9. Code identity at launch

sha256 prefixes; the server copies are identical to the laptop's. Every record also stores the full code/environment
identity (these files, `grid.py` 4c0db0d4bb71ec55 and `datasets.py` 568dd5ca2acf359d on the servers, the 23 files of
the installed FAISS build's `e_means` module and compiled libraries, FAISS 1.16.0, NumPy and Python versions).

| File | sha256 prefix |
|---|---|
| `confirm2_grid.py` (runner) | 43c4d1907957f214 |
| `confirm2_analyze.py` (analysis, laptop) | 26d8767d665ee60d |
| `ablation.py` (frozen rule and comparators, unchanged since round 1) | bd0f15eb4d0a6641 |
| `ablation_cpp.py`, `ngp_kernels.cpp` (unchanged since round 1) | bf6af35b14df3026, be826b95e29abbd7 |
| `ngp_kernels.so`, Host A / Host B (unchanged since round 1) | 5bbe78ecb8cdf5e2 / df3e7810f8dc1f43 |
| `ngp_variants.py`, `ngp_variants2.py` (the variants; unchanged since the exploratory runs) | 2206ca5608fd6c87, 15999ad7b2e45d9d |
| `datasets_extra.py` | be609c0fc22f4594 |
| `launch_r2.sh`, `stop_r2.sh`, `status_r2.sh`, `ctl.sh` | 3b40fcdaa603f9db, cdc41a4b96ced147, e10f623d53057c34, 09c1b304a223f1ea |
| `VARIANTS_MATH.md` | 8e266491e19f51d4 |

## 10. Pre-launch review and tests

### 10.1 Codex review (gpt-6.1-sol, high; `codex_review/`)

- **Round 1** (`REVIEW.md`): **SPEC-CODE VERDICT: MATCH** (the code implements `VARIANTS_MATH.md` exactly; Section 1 of
  that file agrees with `note.tex` and the frozen code; 200 PM/FH root comparisons and 96 posterior checks passed).
  LAUNCH: changes required: 1 blocker, 6 major, 4 minor. All fixed or documented:
  1. Bisection bracket (minor) and 2. Fay–Herriot rank policy (minor): documented in `VARIANTS_MATH.md`; code unchanged
     (the variants are frozen).
  3. Direction (major): the direction rule of Section 6.
  4. EQ did not control decisions (blocker): EQ split into EQ2/EQ5, run early; decisions require the gates (6.1, 6.2, 6.4).
  5. EQ could falsely pass (major): registered keys enumerated; both endpoints finite and equal in both implementations.
  6. Provenance (major): Section 6.5; the analysis refuses unclean records.
  7. Reporting (minor): p = 1 for one nonzero value; ties shown; pr vs medg_pm; all configs in every table.
  8. Resume identity (major): FAISS build, NumPy, Python, data hashes, host and start hashes checked on resume (Section 5).
  9. Fetch (major): exact mirror with deletions; no analysis after a failed transfer.
  10. Start order (minor): sorted indices documented. 11. Thread budget (minor): exclusion in both directions.
- **Round 2** (`REVIEW_round2.md`): fixes 1–11 verified except three major points (adverse cells bypassed the gate and
  the direction rule; a record's part was not checked against its key; resume trusted `meta.json` instead of the data
  files) and two minor ones (failure statuses in `ctl.sh`; ties in the secondary table). All fixed.
- **Round 3** (`REVIEW_round3.md`): those fixes verified; one minor point (a record outside every registered part with no
  part field passed the provenance check). Fixed (one line) and tested on Codex's example; not re-reviewed.
- After round 3, found in the smoke tests: the provenance check required one code identity overall, but each host has
  its own compiled libraries. Changed to one identity per host (and implementation); tested (two builds pass, a mixed
  build within a host is refused).

### 10.2 Smoke tests with the launch code (`cfm2_smoke_final/` on each host; seeds 0, 10, 13 = earlier keys, 100 = new)

Per host: 119 C++ runs and 16 reference runs, 0 errors, all with the launch code (`confirm2_grid.py` 43c4d1907957f214).
- **Reproduction of earlier records** (Shuttle seed 10 and KDD seed 13 on Host A; Corel and cod-rna seed 0 on Host B; all
  7 configs): 67 of 67 records made earlier on the same host are reproduced exactly (test and training MSE, splits). The
  3 `medg_fh` KDD records of seed 13 differ from the exploratory ones, which were made on Host B (Section 8).
- **Reference vs C++** (EQ2 and EQ5 with seed 100: Shuttle, Power, Corel and MNIST/255, both variants, T = 2 and 5): 32
  of 32 identical.
- **k-means++ path** (seed 100, same four datasets): starts computed in the scikit-learn environment, then 56 SK runs,
  each storing its start's hash.

Earlier smoke tests (same day, superseded code versions) also passed a stop-and-resume test: the runner's process group
was killed in the middle of runs (4 processes, all gone; 3 records kept, no partial file) and the restart ran exactly the
39 remaining runs (42 of 42 at the end). A runner restarted after a code change refused to resume, as designed.

## 11. Amendments and events during the round

- 2026-11-19, before launch: the k-means++ starts of SK were being computed ahead of time on both hosts (starts stage,
  6 workers); the grid launch stops that stage, and the grid computes the remaining starts itself before SK.
- **Launch** (2026-11-19) on both hosts (`ctl.sh start`; protocol file sha256 prefix 384fc07f23ad63e6 before this
  line was added). Host A: 2,800 runs of P/S5/SK plus 60 EQ runs; Host B: 6,720 plus 144. Six k-means++ starts of
  seeds 30–39 were on disk on Host B at launch, none on Host A.
- **Primary analysis** (Section 6.1), as soon as P (4,760 of 4,760 runs) and EQ2 (both variants
  68 of 68 identical: PASS) were complete on both hosts; no error. `analysis_primary_2026-11-19.md`. Both co-primary
  hypotheses CONFIRMED: test −2.89% (pseudomedian −3.09%, p 6.3e-05, Holm 6.3e-05, 18/20 seeds), training −8.87%
  (pseudomedian −8.66%, p 1.9e-06, Holm 3.8e-06, 20/20). Significantly worse cells (all small): test 7 (MSD 256/1024,
  MNIST 1024, Fashion 1024, SensIT 256/1024, HIGGS 1024; medians +0.1% to +0.9%), training 6 (medians +0.0% to +0.4%).
  S5, SK, EQ5 continue; the secondary family is evaluated when they are complete.
- **Round complete** (P, S5, SK: 9,520 of 9,520 runs; EQ2 and EQ5 PASS for both variants, 204 of
  204 identical; 0 errors; provenance check passed). Full analysis: `analysis_final_2026-11-19.md`. Secondary family
  (Holm, 10 tests): medg_fh vs pr lower on test (−2.93%) and training (−9.76%); medg_pm vs frozen not significant
  (−0.69%); vs lib and ours lower (−38.5%, −28.5%); vs h21 not significant (−0.81%, Holm p 0.097); S5 (T = 5) lower on
  test (−5.01%) and training (−7.13%), 10/10 seeds; SK (k-means++) **higher on test (+2.39%, Holm p 0.012, 1/10 seeds
  better)** and lower on training (−17.78%). The SK test result comes from Shuttle (+55.8% there; without Shuttle −0.26%).
