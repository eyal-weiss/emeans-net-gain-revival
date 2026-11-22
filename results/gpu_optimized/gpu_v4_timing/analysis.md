# Timing of the final optimized GPU code (2026-11-21, after the round-3 review fix)

NG_TIMING=3: 3 fits per (cell, seed, config), seeds 30-32, 2 epochs; minimum over the 3 fits, median over seeds 31-32 (seed 30 carries compilation). One RTX 3090 Ti, shared with another user's job (see contention.log).

### Head driver (79440bc): the PR's rule (pr), ours with the freeze (ng), ours without (ngn); early stop on

| cell | pr | ng | ngn | ng / pr | ngn / pr |
|---|---|---|---|---|---|
| kdd_z 256 | 0.076 | 0.107 | 0.124 | 1.41 | 1.63 |
| shuttle_z 256 | 0.063 | 0.075 | 0.100 | 1.18 | 1.57 |
| corel_z 1024 | 0.086 | 0.115 | 0.162 | 1.34 | 1.88 |
| geonames 1024 | 0.078 | 0.095 | 0.098 | 1.23 | 1.26 |
| kdd_z 1024 | 0.085 | 0.110 | 0.136 | 1.30 | 1.61 |
| mnist_255 1024 | 0.037 | 0.036 | 0.050 | 0.97 | 1.35 |
| power_z 1024 | 0.144 | 0.189 | 0.191 | 1.32 | 1.33 |
| shuttle_z 1024 | 0.044 | 0.059 | 0.083 | 1.34 | 1.87 |
| codrna_z 4096 | 0.079 | 0.104 | 0.143 | 1.30 | 1.80 |
| geonames 4096 | 0.119 | 0.191 | 0.279 | 1.60 | 2.33 |
| kdd_z 4096 | 0.100 | 0.139 | 0.175 | 1.39 | 1.75 |
| road3d_z 4096 | 0.101 | 0.169 | 0.189 | 1.68 | 1.87 |
| deep1m 16384 | 0.221 | 0.291 | 0.348 | 1.32 | 1.58 |
| spacev1m 16384 | 0.238 | 0.343 | 0.417 | 1.45 | 1.75 |

ng / pr: geometric mean 1.33 over 14 cells (range 0.97-1.68); small cells 1.33, 1M points 1.38
ngn / pr: geometric mean 1.66 over 14 cells (range 1.26-2.33); small cells 1.66, 1M points 1.66

### 98deaca driver: the PR's rule (pr), ours (ng)

| cell | pr | ng | ng / pr |
|---|---|---|---|
| kdd_z 256 | 0.090 | 0.136 | 1.51 |
| shuttle_z 256 | 0.057 | 0.094 | 1.65 |
| corel_z 1024 | 0.041 | 0.062 | 1.50 |
| geonames 1024 | 0.060 | 0.089 | 1.49 |
| kdd_z 1024 | 0.068 | 0.090 | 1.31 |
| mnist_255 1024 | 0.023 | 0.034 | 1.50 |
| power_z 1024 | 0.078 | 0.113 | 1.45 |
| shuttle_z 1024 | 0.037 | 0.056 | 1.54 |
| codrna_z 4096 | 0.060 | 0.088 | 1.45 |
| geonames 4096 | 0.098 | 0.178 | 1.81 |
| kdd_z 4096 | 0.218 | 0.190 | 0.87 |
| road3d_z 4096 | 0.078 | 0.114 | 1.46 |
| deep1m 16384 | 0.348 | 0.450 | 1.29 |
| spacev1m 16384 | 0.292 | 0.568 | 1.94 |

ng / pr: geometric mean 1.46 over 14 cells (range 0.87-1.94); small cells 1.44, 1M points 1.58

### o-means: its own rule (om), ours (om_ng)

| cell | om | om_ng | om_ng / om |
|---|---|---|---|
| kdd_z 256 | 0.042 | 0.092 | 2.18 |
| shuttle_z 256 | 0.029 | 0.037 | 1.28 |
| corel_z 1024 | 0.020 | 0.028 | 1.41 |
| geonames 1024 | 0.017 | 0.027 | 1.58 |
| kdd_z 1024 | 0.063 | 0.102 | 1.63 |
| mnist_255 1024 | 0.020 | 0.038 | 1.86 |
| power_z 1024 | 0.027 | 0.040 | 1.48 |
| shuttle_z 1024 | 0.017 | 0.024 | 1.47 |
| codrna_z 4096 | 0.017 | 0.025 | 1.52 |
| geonames 4096 | 0.020 | 0.031 | 1.55 |
| kdd_z 4096 | 0.095 | 0.159 | 1.67 |
| road3d_z 4096 | 0.021 | 0.032 | 1.56 |
| deep1m 16384 | 0.044 | 0.138 | 3.15 |
| spacev1m 16384 | 0.052 | 0.164 | 3.17 |

om_ng / om: geometric mean 1.75 over 14 cells (range 1.28-3.17); small cells 1.58, 1M points 3.16

