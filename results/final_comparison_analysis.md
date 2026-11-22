# Final comparison on the released implementation: analysis

10336 runs; 34 cells; round-3 records loaded: 10200

- part R: 136 of 136 runs
- part F2: 6120 of 6120 runs
- part F5: 4080 of 4080 runs
- code identity on host_b: 1
- code identity on host_a: 1
- R (ours without the early stop): 136 of 136 runs identical to round 3 (test and training error, splits)
- the PR's rule (F2, F5): 3400 of 3400 runs identical to round 3 (test and training error, splits)

All checks passed.

## 2 epochs (60 seeds x 34 cells)

| comparison | test error | seeds lower | training error | seeds lower | time (median ratio) |
|---|---|---|---|---|---|
| ours, no freeze vs ours, freeze | -1.24% [-1.53%, -0.94%] (p 1.8e-09) | 56/60 | -2.26% [-2.73%, -1.78%] (p 6.9e-10) | 51/60 | 1.061 |
| ours, no freeze vs the PR's rule | -4.40% [-5.30%, -3.49%] (p 2.6e-09) | 50/60 | -16.93% [-17.66%, -16.19%] (p 1.6e-11) | 60/60 | 1.072 |
| ours, freeze vs the PR's rule | -3.20% [-4.16%, -2.23%] (p 5e-07) | 47/60 | -15.01% [-15.85%, -14.16%] (p 1.6e-11) | 60/60 | 1.021 |
| ours, no freeze vs round 3: ours, no freeze, no early stop | +0.06% [+0.05%, +0.08%] (p 5.7e-10) | 4/60 | +0.06% [+0.04%, +0.08%] (p 3.1e-10) | 1/60 | not comparable |
| ours, freeze vs round 3: ours, freeze, no early stop | +0.05% [+0.03%, +0.06%] (p 4.6e-09) | 6/60 | +0.05% [+0.05%, +0.06%] (p 1.6e-11) | 0/60 | not comparable |

Wilcoxon p: two-sided, unadjusted, descriptive.

| configuration | early stop fired | its step / total steps (median) | checks (median) | splits (median) | moves in the last pass | runs with > k/4 empty centres |
|---|---|---|---|---|---|---|
| the PR's rule | 0/2040 | nan | 0 | 405 | 0 | 62 |
| ours, freeze | 769/2040 | 0.24 | 6 | 379 | 0 | 12 |
| ours, no freeze | 894/2040 | 0.29 | 12 | 404 | 174979 | 60 |

Per dataset (geometric-mean ratio over its cells and the seeds; test / training error):

| dataset | no freeze vs freeze | no freeze vs the PR's rule | freeze vs the PR's rule |
|---|---|---|---|
| census_z | -0.05% / -0.04% | -0.68% / -0.63% | -0.63% / -0.58% |
| codrna_z | -0.03% / -0.03% | -0.18% / -0.23% | -0.15% / -0.20% |
| corel_z | -0.16% / -0.07% | -0.50% / -1.37% | -0.35% / -1.30% |
| fashion_255 | -0.11% / -0.10% | +0.08% / -0.01% | +0.19% / +0.09% |
| geonames | -0.16% / -0.01% | -0.64% / -1.33% | -0.48% / -1.33% |
| higgs_z | -0.04% / -0.04% | +0.07% / -0.01% | +0.12% / +0.03% |
| ijcnn1_z | -0.17% / -0.15% | -0.28% / -0.46% | -0.11% / -0.31% |
| kdd_z | -6.07% / -20.55% | -4.45% / -77.44% | +1.73% / -71.61% |
| mnist_255 | -0.08% / -0.07% | +0.02% / -0.03% | +0.10% / +0.05% |
| msd_z | -0.49% / -0.12% | +0.35% / -0.15% | +0.84% / -0.03% |
| power_z | +0.00% / +0.00% | -3.27% / -3.16% | -3.27% / -3.16% |
| road3d_z | -0.04% / -0.04% | +0.10% / +0.00% | +0.13% / +0.05% |
| sensit_z | -0.46% / -0.24% | +0.28% / -0.33% | +0.74% / -0.09% |
| shuttle_z | -9.16% / -3.21% | -47.45% / -56.36% | -42.15% / -54.91% |

## 5 epochs (40 seeds x 34 cells)

| comparison | test error | seeds lower | training error | seeds lower | time (median ratio) |
|---|---|---|---|---|---|
| ours, no freeze vs ours, freeze | +0.10% [+0.05%, +0.14%] (p 0.00023) | 11/40 | -0.45% [-0.65%, -0.26%] (p 4.9e-06) | 35/40 | 1.012 |
| ours, no freeze vs the PR's rule | -3.77% [-4.84%, -2.69%] (p 5.7e-08) | 35/40 | -10.96% [-12.07%, -9.84%] (p 1.8e-12) | 40/40 | 1.043 |
| ours, freeze vs the PR's rule | -3.86% [-4.93%, -2.78%] (p 3.1e-08) | 35/40 | -10.55% [-11.64%, -9.46%] (p 1.8e-12) | 40/40 | 1.036 |
| ours, no freeze vs round 3: ours, no freeze, no early stop | +0.08% [+0.05%, +0.10%] (p 6.5e-08) | 4/40 | +0.30% [+0.13%, +0.48%] (p 7.8e-09) | 1/40 | not comparable |
| ours, freeze vs round 3: ours, freeze, no early stop | +0.08% [+0.05%, +0.11%] (p 3.7e-09) | 3/40 | +0.21% [+0.04%, +0.37%] (p 5.8e-07) | 2/40 | not comparable |

Wilcoxon p: two-sided, unadjusted, descriptive.

| configuration | early stop fired | its step / total steps (median) | checks (median) | splits (median) | moves in the last pass | runs with > k/4 empty centres |
|---|---|---|---|---|---|---|
| the PR's rule | 0/1360 | nan | 0 | 365 | 0 | 25 |
| ours, freeze | 679/1360 | 0.16 | 12 | 324 | 0 | 0 |
| ours, no freeze | 721/1360 | 0.18 | 14 | 325 | 38926 | 0 |

Per dataset (geometric-mean ratio over its cells and the seeds; test / training error):

| dataset | no freeze vs freeze | no freeze vs the PR's rule | freeze vs the PR's rule |
|---|---|---|---|
| census_z | -0.00% / -0.00% | -0.90% / -0.84% | -0.90% / -0.84% |
| codrna_z | -0.00% / -0.00% | -0.24% / -0.24% | -0.23% / -0.24% |
| corel_z | -0.02% / -0.00% | -0.06% / -0.83% | -0.04% / -0.83% |
| fashion_255 | -0.01% / -0.01% | +0.04% / -0.00% | +0.05% / +0.01% |
| geonames | -0.01% / +0.02% | -0.30% / -0.76% | -0.29% / -0.78% |
| higgs_z | -0.00% / -0.01% | +0.14% / +0.06% | +0.14% / +0.07% |
| ijcnn1_z | -0.03% / -0.02% | -0.17% / -0.35% | -0.15% / -0.33% |
| kdd_z | +1.11% / -4.06% | -4.16% / -61.97% | -5.21% / -60.36% |
| mnist_255 | -0.01% / -0.00% | +0.01% / -0.03% | +0.01% / -0.02% |
| msd_z | -0.10% / +0.00% | +0.45% / -0.08% | +0.55% / -0.08% |
| power_z | +0.00% / +0.00% | -3.34% / -3.54% | -3.34% / -3.54% |
| road3d_z | -0.00% / -0.00% | +0.10% / +0.17% | +0.10% / +0.18% |
| sensit_z | -0.09% / -0.09% | +0.47% / -0.37% | +0.55% / -0.28% |
| shuttle_z | +0.29% / -1.41% | -42.09% / -35.97% | -42.26% / -35.06% |

