# Everything we tried on cluster revival (11-19 Nov 2026): one-page index

Seed for the technical report. Each line: idea -> outcome -> where the records/notes are. "vs PR" = FAISS PR #5915's
cost-based revival at 98deaca, test error unless stated. Details: assylbekov_2026-11-18/NOTES.md (18-19 Nov),
night_2026-11-16/REPORT.md, followup_2026-11-19/REPORT.md, confirm_report/REPORT.md, confirm2/, side_2026-11-19/.

## A. Distance-aware revival (arXiv v1, 11-16 Nov)
- Library count rule (usage < 0.1 x mean -> freed; donors by usage; epsilon split): evicts small isolated clusters.
- v1 rule: distance test (keep a rarely used centre that is far from every other centre, squared distance > 4 x MSE)
  + abandonment clause. Large gains on KDD/Shuttle/GeoNames, identical on embeddings, overhead ~1-3%. (paper, results/all)
- Simple alternatives (lower size threshold 0.01/0.03, revival in first epoch only, abandonment clause alone): all fail
  (lose revival's benefit on embeddings or do not protect isolated groups). (night_2026-11-16 §2.2-B)
- Refinements (total cost count x distance^2, donor-aware total cost, local error scale, combined): each fixes its
  target failure, loses elsewhere (KDD k-means++ +70..+109%). Lesson: needs a measured split gain. (§2.2-A)
- KDD with min-max scaling: v1 gain shrinks to 3-10% (standardisation places rare one-hot categories far away).
- scikit-learn MiniBatchKMeans: same flaw; the distance veto removes 1/3-1/2 of the excess on KDD, nothing on GeoNames.
- GPU ports of the v1 rule (torch e-means, o-means): bit-identical, overhead +3..15%. (gpu_port/NOTES.md)

## B. Concurrent work and hybrids (18 Nov)
- PR #5915 cost-based revival (removal cost from second-nearest margins, donors by Gaussian split gain (2/pi)|U|,
  split along the principal axis, freeze in the last pass): beats v1 in 158/160 runs; fails Assylbekov's k = 2
  counterexample (median of 2 costs = mean). (results/pr5915/compare.md)
- Ablation of the PR: cost removal gives most, gain donors next, axis split only matters with gain donors, timing minor.
- h21 = PR + distance test: best on Shuttle test, small losses on KDD/GeoNames/MSD; fixes the k = 2 counterexample.
- h22 = v1 + gain donors + axis split: better than v1, far behind PR. (results/pr5915/ablation)

## C. Net-gain rule development (18-19 Nov; development seeds 0-9)
- Net-gain check cost_j < w gain_d with greedy matching; cost = bound N_j delta_j (no second-nearest search) or the
  PR's estimate: w = 1-1.25 best (-3% vs PR); higher w churns; the estimate is not better than the bound. (netgain/)
- Cool-down: no move while inherited EMA weight > 1/2 (parameter-free half-life): stops churn, -4% vs PR; a fixed
  10-batch cool-down hurts KDD; threshold 1/4 unstable. (cooldown/)
- Uniform discount w = 0.25-0.75: no help -> errors are evidence-specific, not a global bias. (lowerw/)
- Calibration study: the cost bound is 13-100x too low for low-evidence centres; gains calibrated for random centres but
  chosen donors 1.2-2.1x high (winner's curse); train and test truths agree once evidence >= 1. (calib/)
- ngp (posterior net gain: Gamma-Poisson rate posterior in the cost; log-normal empirical-Bayes gain with start-up
  bias removed; matcher v2 with pair guard): tabular -7.6% vs PR, but embeddings worse than the library (the bound is
  too loose in high d -> half the moves). ngp_ivf (approximate delta search by IVF): no quality cost; faster only for
  k >= 1024 with larger d. (ngp/, emb_ngp/)
- ngp2 (cost calibrated by sampled second-nearest margins, 1 row in 8; gain noise from pooled kurtosis): -7.5% vs PR,
  embedding regression fixed. Residual winner's curse on gains. (ngp2/, emb_ngp2/, calib_ngp2/)
- ngp3 (gain measured for the actual split, q = E|p|/s): churns (young donors overvalued); variants ngp3e, ngp3b,
  ngp3_gr, ngp3f (forgetting-factor rates) all fail. Lesson: keep ngp2's log-scale gain shrinkage.
- ngp4 (ngp2 + every row + no fallback donor + moderated ratio variances + compound gain noise): KDD worse; every step
  that increases gain shrinkage costs KDD. (ngp4/, kdd_attr/)
- Propose-then-confirm on the immediate effect (ngp4c), on the one-step-adapted effect (ngp4ca): far worse on Shuttle
  (evidence-starved; moves pay off only after adaptation).
- Winner's-curse corrections: measured gain noise (ngp5m), Tweedie (ngp5t), both (ngp5mt): none beats ngp2. (ngp5/)
- Adaptive confirmation wait (ngp2cw): +46% worse. Every-batch checks with the mean rule (ngp2b): unstable (KDD).
- Event-triggered checks (ngp2e, exact O(k) bound): same decisions as every-batch, but fires on ~all move-phase
  batches -> no saving. (trig/)
- Second-nearest sampling strides 1-32: sparser looked better on KDD; dropped once the C++ kernels made every row cheap.

## D. Confirmatory round 1 (frozen ngp2_full, seeds 10-29; 19 Nov night)
- Primary not confirmed (test -2.2% vs PR, p 0.26); k-means++ starts +10.9% worse; T = 5 KDD 1024 runaway (537-761
  empty centres); train -19.6%. Shuttle test error is ~68% one test point far from all training data. (confirm_report/)

## E. Diagnosis and fixes (19 Nov day; exploratory)
- Runaway: posterior MEAN of a log-normal gain with no evidence (huge prior variance) -> empty centres top donors.
- KDD 256 failure: unweighted moment prior collapses to its floor -> all posterior gains equal, no donor ranking.
- KDD 4096 weak spot: one global prior shrinks the real gains of the sparse far tail.
- Fixes: medg (posterior median gain), med (median also for the rate in the cost: train +7%, rejected), pm (Paule-Mandel
  prior, mean: still runs away), medg_pm (PM prior + median: CHOSEN), medg_fh (Fay-Herriot prior on log rate and
  previous delta + median: alternative; trades KDD 1024 test for KDD 4096 train). (diag_t5/, followup_2026-11-19/)
- 8 new datasets: benign, every net-gain variant within ~1% of PR. Stress tests (T = 3, 10; every-batch checks;
  synthetic far groups; Covertype; embeddings): medg_pm stable everywhere. (ext/, ext2/, ext3/, ext4/)

## F. Confirmatory round 2 (medg_pm, seeds 30-49, 34 cells; 19 Nov)
- Primary confirmed: test -2.9%, train -8.9% vs PR. T = 5: -5.0% / -7.1%. k-means++: test +2.4% (Shuttle), train
  -17.8%. vs h21 and vs the frozen rule: n.s. medg_fh ~ medg_pm. (confirm2/analysis_final_2026-11-19.md)

## G. Side experiments (19 Nov evening; exploratory)
- PR head 79440bc (no freeze, closing handover, no guard): PR's own train +6.7% worse. Our rule on its driver, no
  freeze + guard: -1.25% test / -2.1% train vs freeze at T = 2 (neutral at T = 5), +20 pts time; dropping our
  pair guard hurts. Check interval: medg_pm flat for every 1-5 batches, worse at 10/20; PR worse at 1 and 20; event
  trigger again no saving. GPU e-means + medg_pm: train -31%, test +3.5% vs GPU PR (unoptimized). o-means + medg_pm:
  test -67% vs o-means' count rule. (side_2026-11-19/)

## H. Engineering and evaluation lessons
- Extra time of the net-gain rule = the all-centres nearest-other-centre search, ~k/(5 T B) of the row search (verified);
  KD-tree helps only for d <= 3; Python Paule-Mandel fit ~0.5 ms per check.
- Fused C++ kernels make every-row margin statistics nearly free; the reference Python and C++ are bit-identical.
- Same-host runs are bit-reproducible; KDD differs across CPUs -> pin datasets to hosts.
- Test error on Shuttle (and partly KDD) is dominated by a few test points far from all training data -> report train and
  test as co-primary; MNIST per-pixel min-max creates artificial outliers -> scale by /255.
- Variants designed after seeing confirmatory seeds need fresh seeds (hence round 2 on seeds 30-49).

## I. Runtime work (19 Nov night)
- Regime switch to a cheaper rule: the tail share does not predict where the count rule is good enough (GeoNames,
  Power, Road3D, Corel lose 13-95% with it); it is within ~1% only on diffuse high-d data. (side_2026-11-19/regime_runtime.md)
- Early end of the move phase: moves front-loaded on low-d data; offline estimates 2-12% time saved; not yet run.
  (side_2026-11-19/stop_measure.md)
- Whole decision in C++: bit-identical decisions (306 jobs), 3-6.5x faster decision, medg_pm vs PR +10% -> +7%.
  (side_2026-11-19/cpp_decision.md)
