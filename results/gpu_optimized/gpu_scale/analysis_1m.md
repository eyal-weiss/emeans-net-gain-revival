# No freeze at 1M points, k = 16,384 (head driver, 2 epochs, seeds 10-14, scored runs on a shared GPU)

Run 2026-11-21 22:22-22:25 on host_a (a1m.sh; final reviewed code). Geometric mean over the two datasets per seed,
mean over seeds with a 95% t interval; "lower" = seeds where the first rule has lower error.

| comparison | test error | training error |
|---|---|---|
| no freeze vs freeze | -0.24% [-0.25, -0.22], lower 5/5 | -0.17% [-0.18, -0.16], lower 5/5 |
| no freeze vs the PR's rule | -0.08% [-0.21, +0.05], lower 4/5 | +0.01% [+0.01, +0.02], lower 0/5 |
| freeze vs the PR's rule | +0.15% [+0.03, +0.27], lower 0/5 | +0.18% [+0.17, +0.20], lower 0/5 |

Per dataset: DEEP no freeze vs freeze test -0.26% (5/5), SPACEV -0.21% (5/5); no freeze vs PR test DEEP -0.07%, SPACEV
-0.09% (5/5).

Empty clusters at the end (median): PR 18, freeze 299, no freeze 40. With the freeze, about 300 of 16,384 centroids
(1.8%) end the run empty; without it the checks keep reviving them through the last pass. Checks: freeze 6, no freeze 12.
The freeze runs are bit-identical to the night's 98deaca runs (same final-centroid sha, e.g. DEEP seed 10 92e7c09a).
