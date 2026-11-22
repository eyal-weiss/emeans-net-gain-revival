# D2: timing on a quiet GPU (2026-11-21 10:32, only our process on the GPU in all 11 samples)

Same cells, seeds, configs and method as part D (records D2_timing_em.jsonl, D2_timing_om.jsonl; contention_D2.log).

### GPU e-means (pr: the PR's GPU rule with CUDA graphs; ng: ours, optimized; poc: our first, unoptimized port)

Seconds per fit: minimum over 3 fits per seed (contention only adds time), median over the seeds after the first (which carries compilation).

| cell | pr | ng | poc | ng / pr | poc / pr |
|---|---|---|---|---|---|
| kdd_z 256 | 0.049 | 0.075 | - | 1.54 | - |
| shuttle_z 256 | 0.049 | 0.106 | - | 2.14 | - |
| corel_z 1024 | 0.041 | 0.090 | - | 2.17 | - |
| geonames 1024 | 0.060 | 0.125 | - | 2.09 | - |
| kdd_z 1024 | 0.063 | 0.086 | - | 1.37 | - |
| mnist_255 1024 | 0.023 | 0.041 | - | 1.77 | - |
| power_z 1024 | 0.078 | 0.148 | - | 1.88 | - |
| shuttle_z 1024 | 0.037 | 0.081 | - | 2.21 | - |
| codrna_z 4096 | 0.061 | 0.125 | - | 2.03 | - |
| geonames 4096 | 0.098 | 0.230 | - | 2.35 | - |
| kdd_z 4096 | 0.176 | 0.141 | - | 0.80 | - |
| road3d_z 4096 | 0.078 | 0.158 | - | 2.02 | - |
| deep1m 16384 | 0.220 | 0.305 | - | 1.39 | - |
| spacev1m 16384 | 0.242 | 0.337 | - | 1.39 | - |

ng / pr: geometric mean over cells 1.74 (range 0.80-2.35)

### o-means (om: o-means' own rule; om_ng: ours, optimized; om_poc: unoptimized port)

Seconds per fit: minimum over 3 fits per seed (contention only adds time), median over the seeds after the first (which carries compilation).

| cell | om | om_ng | om_poc | om_ng / om | om_poc / om |
|---|---|---|---|---|---|
| kdd_z 256 | 0.036 | 0.077 | - | 2.10 | - |
| shuttle_z 256 | 0.029 | 0.048 | - | 1.63 | - |
| corel_z 1024 | 0.020 | 0.036 | - | 1.79 | - |
| geonames 1024 | 0.017 | 0.028 | - | 1.64 | - |
| kdd_z 1024 | 0.041 | 0.075 | - | 1.84 | - |
| mnist_255 1024 | 0.022 | 0.048 | - | 2.18 | - |
| power_z 1024 | 0.028 | 0.051 | - | 1.83 | - |
| shuttle_z 1024 | 0.016 | 0.033 | - | 2.03 | - |
| codrna_z 4096 | 0.017 | 0.035 | - | 2.04 | - |
| geonames 4096 | 0.020 | 0.032 | - | 1.59 | - |
| kdd_z 4096 | 0.060 | 0.110 | - | 1.83 | - |
| road3d_z 4096 | 0.021 | 0.042 | - | 2.00 | - |
| deep1m 16384 | 0.044 | 0.164 | - | 3.72 | - |
| spacev1m 16384 | 0.052 | 0.185 | - | 3.58 | - |

om_ng / om: geometric mean over cells 2.05 (range 1.59-3.72)

