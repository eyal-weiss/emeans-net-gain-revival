# Timing of the optimized GPU code (2026-11-21 afternoon)

NG_TIMING=3: 3 fits per (cell, seed, config), seeds 30-32, 2 epochs; minimum over the 3 fits, median over seeds 31-32 (seed 30 carries compilation). One RTX 3090 Ti, shared with another user's job (another process in 95% of the 5-second samples; contention.log). Code: gpu_fast_head (head driver) and gpu_fast_v2 (98deaca, o-means) on host_a, identical files.

### Head driver (79440bc): the PR's rule (pr), ours with the freeze (ng), ours without (ngn); early stop on

| cell | pr | ng | ngn | ng / pr | ngn / pr |
|---|---|---|---|---|---|
| kdd_z 256 | 0.059 | 0.078 | 0.088 | 1.31 | 1.49 |
| shuttle_z 256 | 0.063 | 0.074 | 0.099 | 1.17 | 1.57 |
| corel_z 1024 | 0.051 | 0.063 | 0.085 | 1.22 | 1.66 |
| geonames 1024 | 0.076 | 0.094 | 0.094 | 1.24 | 1.23 |
| kdd_z 1024 | 0.065 | 0.085 | 0.100 | 1.31 | 1.54 |
| mnist_255 1024 | 0.023 | 0.034 | 0.045 | 1.47 | 1.96 |
| power_z 1024 | 0.095 | 0.119 | 0.119 | 1.26 | 1.26 |
| shuttle_z 1024 | 0.044 | 0.059 | 0.082 | 1.34 | 1.86 |
| codrna_z 4096 | 0.070 | 0.091 | 0.124 | 1.29 | 1.76 |
| geonames 4096 | 0.114 | 0.182 | 0.264 | 1.59 | 2.32 |
| kdd_z 4096 | 0.099 | 0.138 | 0.174 | 1.39 | 1.75 |
| road3d_z 4096 | 0.090 | 0.118 | 0.164 | 1.31 | 1.82 |
| deep1m 16384 | 0.219 | 0.289 | 0.346 | 1.32 | 1.58 |
| spacev1m 16384 | 0.223 | 0.320 | 0.396 | 1.43 | 1.77 |

ng / pr: geometric mean 1.33 over 14 cells (range 1.17-1.59); small cells 1.32, 1M points 1.37
ngn / pr: geometric mean 1.66 over 14 cells (range 1.23-2.32); small cells 1.66, 1M points 1.67

### 98deaca driver: the PR's rule (pr), ours (ng)

| cell | pr | ng | ng / pr |
|---|---|---|---|
| kdd_z 256 | 0.048 | 0.067 | 1.42 |
| shuttle_z 256 | 0.051 | 0.072 | 1.43 |
| corel_z 1024 | 0.041 | 0.060 | 1.45 |
| geonames 1024 | 0.060 | 0.089 | 1.48 |
| kdd_z 1024 | 0.063 | 0.082 | 1.29 |
| mnist_255 1024 | 0.023 | 0.034 | 1.49 |
| power_z 1024 | 0.079 | 0.114 | 1.45 |
| shuttle_z 1024 | 0.037 | 0.057 | 1.53 |
| codrna_z 4096 | 0.061 | 0.088 | 1.44 |
| geonames 4096 | 0.099 | 0.178 | 1.80 |
| kdd_z 4096 | 0.177 | 0.136 | 0.77 |
| road3d_z 4096 | 0.079 | 0.115 | 1.45 |
| deep1m 16384 | 0.220 | 0.284 | 1.30 |
| spacev1m 16384 | 0.243 | 0.316 | 1.30 |

ng / pr: geometric mean 1.38 over 14 cells (range 0.77-1.80); small cells 1.39, 1M points 1.30

### o-means: its own rule (om), ours (om_ng)

| cell | om | om_ng | om_ng / om |
|---|---|---|---|
| kdd_z 256 | 0.036 | 0.066 | 1.84 |
| shuttle_z 256 | 0.028 | 0.036 | 1.28 |
| corel_z 1024 | 0.020 | 0.028 | 1.41 |
| geonames 1024 | 0.017 | 0.027 | 1.58 |
| kdd_z 1024 | 0.040 | 0.065 | 1.63 |
| mnist_255 1024 | 0.021 | 0.038 | 1.84 |
| power_z 1024 | 0.027 | 0.041 | 1.50 |
| shuttle_z 1024 | 0.016 | 0.024 | 1.49 |
| codrna_z 4096 | 0.016 | 0.025 | 1.52 |
| geonames 4096 | 0.020 | 0.031 | 1.54 |
| kdd_z 4096 | 0.060 | 0.103 | 1.72 |
| road3d_z 4096 | 0.021 | 0.032 | 1.53 |
| deep1m 16384 | 0.043 | 0.137 | 3.17 |
| spacev1m 16384 | 0.051 | 0.163 | 3.18 |

om_ng / om: geometric mean 1.73 over 14 cells (range 1.28-3.18); small cells 1.57, 1M points 3.17

