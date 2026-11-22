# GPU at 1M and 10M points, k = 16,384 (head driver 79440bc; final code; 2026-11-22 02:12-02:15)

Run by `run.sh` once the GPU had had no other process for 3 minutes (and the CPU comparison had ended). One RTX 3090 Ti.
`contention.log`: the timing parts (02:12:42-02:14:17) ran with no other process on the GPU; another user's job started
at about 02:14:45, during the 10M accuracy runs (accuracy does not depend on timing).

## Timing (NG_TIMING=3: three fits per seed, minimum per seed, median over seeds 31-32; 2 epochs)

| cell | PR's rule | ours, freeze | ours, no freeze | freeze / PR | no freeze / PR |
|---|---|---|---|---|---|
| DEEP 1M | 0.219 s | 0.289 s | 0.346 s | 1.32 | 1.58 |
| SPACEV 1M | 0.223 s | 0.319 s | 0.396 s | 1.43 | 1.78 |
| DEEP 10M | 1.683 s | 1.799 s | 1.889 s | **1.07** | **1.12** |

o-means (its own rule / ours): DEEP 1M 0.044 / 0.139 s (3.16); SPACEV 1M 0.052 / 0.164 s (3.15); DEEP 10M 0.418 / 1.207 s
(2.89). Checks at 10M: freeze 7, no freeze 15 (1M: 6 and 12). The 1M times equal those of the shared-GPU timing of
2026-11-21 (`gpu_v3_timing`) to the millisecond.

## Accuracy at 10M (DEEP, seeds 10-12, 2 epochs)

| comparison | test error | training error |
|---|---|---|
| no freeze vs freeze | +0.005% [-0.031, +0.040], lower 1/3 | -0.004% [-0.012, +0.003], lower 3/3 |
| no freeze vs the PR's rule | +0.020% [-0.050, +0.091], lower 0/3 | -0.030% [-0.094, +0.034], lower 3/3 |
| freeze vs the PR's rule | +0.016% [-0.059, +0.091], lower 1/3 | -0.026% [-0.097, +0.046], lower 2/3 |

Empty centres at the end: PR 12-16, freeze 11-13, no freeze 11-14 (at 1M: 18, about 300, 40). The early stop did not
fire. At this size the three rules are indistinguishable.
