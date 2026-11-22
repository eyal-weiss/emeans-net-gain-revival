# GPU experiments: the optimized net-gain rule in GPU e-means and in o-means

Exploratory (not pre-registered). One RTX 3090 Ti (host_a), shared with another user's training job during the runs (see the last section). Seeds with random starts as in round 2; 2 epochs.

## D. Timing

### GPU e-means (pr: the PR's GPU rule with CUDA graphs; ng: ours, optimized; poc: our first, unoptimized port)

Seconds per fit: minimum over 3 fits per seed (contention only adds time), median over the seeds after the first (which carries compilation).

| cell | pr | ng | poc | ng / pr | poc / pr |
|---|---|---|---|---|---|
| kdd_z 256 | 0.048 | 0.075 | 0.082 | 1.54 | 1.69 |
| shuttle_z 256 | 0.050 | 0.107 | 0.608 | 2.14 | 12.10 |
| corel_z 1024 | 0.042 | 0.093 | 0.480 | 2.18 | 11.29 |
| geonames 1024 | 0.060 | 0.125 | 0.576 | 2.08 | 9.62 |
| kdd_z 1024 | 0.064 | 0.088 | 0.114 | 1.38 | 1.78 |
| mnist_255 1024 | 0.024 | 0.043 | 0.104 | 1.82 | 4.38 |
| power_z 1024 | 0.080 | 0.150 | 0.574 | 1.88 | 7.19 |
| shuttle_z 1024 | 0.037 | 0.082 | 0.390 | 2.20 | 10.47 |
| codrna_z 4096 | 0.062 | 0.128 | 0.925 | 2.05 | 14.85 |
| geonames 4096 | 0.099 | 0.231 | 1.146 | 2.35 | 11.64 |
| kdd_z 4096 | 0.176 | 0.142 | 0.396 | 0.80 | 2.24 |
| road3d_z 4096 | 0.080 | 0.161 | 0.996 | 2.01 | 12.47 |
| deep1m 16384 | 0.221 | 0.307 | 4.044 | 1.38 | 18.26 |
| spacev1m 16384 | 0.242 | 0.330 | 4.001 | 1.36 | 16.51 |

ng / pr: geometric mean over cells 1.74 (range 0.80-2.35)

poc / pr: geometric mean over cells 7.53 (range 1.69-18.26)

### o-means (om: o-means' own rule; om_ng: ours, optimized; om_poc: unoptimized port)

Seconds per fit: minimum over 3 fits per seed (contention only adds time), median over the seeds after the first (which carries compilation).

| cell | om | om_ng | om_poc | om_ng / om | om_poc / om |
|---|---|---|---|---|---|
| kdd_z 256 | 0.037 | 0.077 | 0.141 | 2.10 | 3.83 |
| shuttle_z 256 | 0.032 | 0.050 | 0.090 | 1.57 | 2.82 |
| corel_z 1024 | 0.023 | 0.042 | 0.071 | 1.82 | 3.09 |
| geonames 1024 | 0.019 | 0.030 | 0.123 | 1.53 | 6.38 |
| kdd_z 1024 | 0.041 | 0.076 | 0.241 | 1.86 | 5.91 |
| mnist_255 1024 | 0.024 | 0.050 | 0.125 | 2.10 | 5.24 |
| power_z 1024 | 0.028 | 0.052 | 0.245 | 1.84 | 8.63 |
| shuttle_z 1024 | 0.018 | 0.037 | 0.058 | 2.00 | 3.17 |
| codrna_z 4096 | 0.020 | 0.040 | 0.194 | 2.01 | 9.80 |
| geonames 4096 | 0.021 | 0.034 | 0.272 | 1.60 | 12.85 |
| kdd_z 4096 | 0.061 | 0.110 | 0.660 | 1.82 | 10.91 |
| road3d_z 4096 | 0.024 | 0.048 | 0.217 | 1.97 | 8.92 |
| deep1m 16384 | 0.045 | 0.162 | 4.846 | 3.58 | 107.15 |
| spacev1m 16384 | 0.052 | 0.185 | 5.393 | 3.57 | 103.86 |

om_ng / om: geometric mean over cells 2.02 (range 1.53-3.58)

om_poc / om: geometric mean over cells 9.04 (range 2.82-107.15)

## A. GPU e-means: 34 cells, 10 seeds

### ours (ng) vs the PR's GPU rule (pr)

34 complete cells of 34, seeds 30-39 (10).

- test error, ng vs pr: +0.4% [-2.2%, +3.1%], lower on 5/10 seeds, Wilcoxon p = 0.7
- training error, ng vs pr: -11.6% [-13.2%, -10.1%], lower on 10/10 seeds, Wilcoxon p = 0.002
- time ng / pr (median over seeds per cell, geometric mean over cells): 1.82 (range 0.81-2.54; scored runs, a shared GPU: see part D for clean timing)

| cell | test ng/pr | train ng/pr | time ng/pr | empty pr / ng |
|---|---|---|---|---|
| census_z 256 | -0.9% | -0.8% | 1.67 | 0 / 0 |
| census_z 1024 | -1.0% | -0.9% | 1.71 | 0 / 0 |
| census_z 4096 | -0.5% | -0.5% | 1.27 | 0 / 0 |
| codrna_z 256 | -0.2% | -0.4% | 2.50 | 0 / 0 |
| codrna_z 1024 | -0.0% | -0.0% | 2.21 | 0 / 0 |
| codrna_z 4096 | -0.1% | -0.2% | 2.28 | 0 / 0 |
| corel_z 256 | -0.2% | -0.5% | 2.54 | 0 / 0 |
| corel_z 1024 | -0.5% | -1.4% | 2.21 | 7 / 4 |
| fashion_255 256 | +0.0% | -0.0% | 1.79 | 0 / 0 |
| fashion_255 1024 | +0.4% | +0.2% | 1.97 | 0 / 0 |
| geonames 256 | -0.6% | -1.2% | 2.28 | 0 / 0 |
| geonames 1024 | -1.0% | -1.0% | 2.19 | 0 / 0 |
| geonames 4096 | -1.0% | -2.5% | 2.37 | 16 / 5 |
| higgs_z 256 | +0.1% | +0.0% | 1.97 | 0 / 0 |
| higgs_z 1024 | +0.2% | +0.1% | 1.63 | 0 / 0 |
| ijcnn1_z 256 | -0.5% | -0.5% | 1.81 | 0 / 0 |
| ijcnn1_z 1024 | -0.5% | -0.8% | 1.87 | 0 / 0 |
| kdd_z 256 | +2.6% | -54.5% | 1.55 | 2 / 3 |
| kdd_z 1024 | +5.3% | -60.0% | 1.39 | 15 / 10 |
| kdd_z 4096 | +5.7% | -41.1% | 0.81 | 238 / 68 |
| mnist_255 256 | +0.0% | -0.0% | 1.79 | 0 / 0 |
| mnist_255 1024 | +0.2% | +0.1% | 1.74 | 0 / 0 |
| msd_z 256 | +0.9% | +0.5% | 1.35 | 0 / 0 |
| msd_z 1024 | +0.6% | +0.0% | 1.50 | 0 / 0 |
| msd_z 4096 | +0.2% | -0.2% | 0.82 | 10 / 2 |
| power_z 256 | -1.4% | -2.2% | 2.32 | 0 / 0 |
| power_z 1024 | -1.7% | -2.0% | 1.97 | 0 / 0 |
| road3d_z 256 | -0.2% | -0.2% | 2.04 | 0 / 0 |
| road3d_z 1024 | +0.2% | +0.1% | 2.08 | 0 / 0 |
| road3d_z 4096 | -0.2% | -0.4% | 2.29 | 0 / 0 |
| sensit_z 256 | +0.4% | +0.1% | 1.82 | 0 / 0 |
| sensit_z 1024 | +0.6% | -0.2% | 1.60 | 0 / 1 |
| shuttle_z 256 | -28.2% | -58.1% | 2.53 | 52 / 57 |
| shuttle_z 1024 | +50.2% | -61.4% | 2.11 | 422 / 200 |

## B. o-means: 34 cells, 10 seeds

### ours (om_ng) vs o-means' rule (om)

34 complete cells of 34, seeds 30-39 (10).

- test error, om_ng vs om: -32.4% [-33.5%, -31.3%], lower on 10/10 seeds, Wilcoxon p = 0.002
- training error, om_ng vs om: -40.3% [-41.3%, -39.2%], lower on 10/10 seeds, Wilcoxon p = 0.002
- time om_ng / om (median over seeds per cell, geometric mean over cells): 1.88 (range 1.33-3.36; scored runs, a shared GPU: see part D for clean timing)

| cell | test om_ng/om | train om_ng/om | time om_ng/om | empty om / om_ng |
|---|---|---|---|---|
| census_z 256 | -6.1% | -6.0% | 1.53 | 0 / 0 |
| census_z 1024 | -5.4% | -5.0% | 3.36 | 0 / 0 |
| census_z 4096 | -5.1% | -4.9% | 3.33 | 0 / 0 |
| codrna_z 256 | -2.4% | -2.8% | 1.47 | 0 / 0 |
| codrna_z 1024 | -4.1% | -4.1% | 1.61 | 0 / 0 |
| codrna_z 4096 | -4.6% | -5.3% | 2.01 | 0 / 0 |
| corel_z 256 | -12.2% | -11.5% | 1.90 | 0 / 0 |
| corel_z 1024 | -10.5% | -13.8% | 1.84 | 0 / 0 |
| fashion_255 256 | -0.6% | -0.7% | 1.98 | 0 / 0 |
| fashion_255 1024 | -0.3% | -1.0% | 2.59 | 0 / 0 |
| geonames 256 | +0.0% | +0.0% | 1.60 | 0 / 0 |
| geonames 1024 | +0.0% | +0.0% | 1.64 | 0 / 0 |
| geonames 4096 | +0.0% | +0.0% | 1.64 | 0 / 0 |
| higgs_z 256 | -0.2% | -0.3% | 1.51 | 0 / 0 |
| higgs_z 1024 | +0.1% | -0.5% | 1.99 | 0 / 0 |
| ijcnn1_z 256 | -6.0% | -6.5% | 1.62 | 0 / 0 |
| ijcnn1_z 1024 | -9.2% | -11.0% | 1.72 | 0 / 0 |
| kdd_z 256 | -95.2% | -90.6% | 2.72 | 5 / 2 |
| kdd_z 1024 | -96.4% | -94.7% | 1.85 | 11 / 10 |
| kdd_z 4096 | -74.4% | -97.8% | 1.81 | 29 / 81 |
| mnist_255 256 | -0.2% | -0.3% | 2.15 | 0 / 0 |
| mnist_255 1024 | -0.2% | -0.9% | 1.83 | 0 / 0 |
| msd_z 256 | -0.5% | -1.4% | 1.89 | 0 / 0 |
| msd_z 1024 | +0.0% | -2.3% | 2.00 | 0 / 0 |
| msd_z 4096 | +0.2% | -4.4% | 2.72 | 0 / 4 |
| power_z 256 | -26.0% | -27.1% | 1.36 | 0 / 0 |
| power_z 1024 | -30.6% | -30.8% | 1.88 | 0 / 0 |
| road3d_z 256 | -7.9% | -8.3% | 1.33 | 0 / 0 |
| road3d_z 1024 | -14.5% | -14.6% | 1.68 | 0 / 0 |
| road3d_z 4096 | -19.4% | -21.4% | 1.84 | 0 / 0 |
| sensit_z 256 | -0.6% | -1.6% | 1.54 | 0 / 0 |
| sensit_z 1024 | -0.3% | -4.6% | 1.88 | 0 / 1 |
| shuttle_z 256 | -92.3% | -97.1% | 1.87 | 0 / 30 |
| shuttle_z 1024 | -69.4% | -93.8% | 1.92 | 2 / 42 |

## C. Large scale (k = 16,384)

### GPU e-means

2 complete cells of 3, seeds 10-14 (5).

- test error, ng vs pr: +0.1% [+0.0%, +0.2%], lower on 1/5 seeds, Wilcoxon p = 0.12
- training error, ng vs pr: +0.2% [+0.1%, +0.2%], lower on 0/5 seeds, Wilcoxon p = 0.062
- time ng / pr (median over seeds per cell, geometric mean over cells): 1.41 (range 1.40-1.42; scored runs, a shared GPU: see part D for clean timing)

| cell | test ng/pr | train ng/pr | time ng/pr | empty pr / ng |
|---|---|---|---|---|
| deep1m 16384 | +0.2% | +0.2% | 1.42 | 18 / 277 |
| spacev1m 16384 | +0.1% | +0.1% | 1.40 | 31 / 348 |

### o-means

2 complete cells of 3, seeds 10-14 (5).

- test error, om_ng vs om: -0.4% [-0.5%, -0.2%], lower on 5/5 seeds, Wilcoxon p = 0.062
- training error, om_ng vs om: -0.5% [-0.6%, -0.5%], lower on 5/5 seeds, Wilcoxon p = 0.062
- time om_ng / om (median over seeds per cell, geometric mean over cells): 3.65 (range 3.56-3.75; scored runs, a shared GPU: see part D for clean timing)

| cell | test om_ng/om | train om_ng/om | time om_ng/om | empty om / om_ng |
|---|---|---|---|---|
| deep1m 16384 | -0.1% | -0.3% | 3.56 | 36 / 269 |
| spacev1m 16384 | -0.6% | -0.7% | 3.75 | 93 / 375 |

### GPU sharing during the runs

675 samples (every 5 s): GPU utilization median 0%, mean 34%; samples with more than one process on the GPU (ours plus another user's): 53%.

Stages:

```
Sat 21 Nov 2026 02:07:29 EST D-em start
Sat 21 Nov 2026 02:09:24 EST finished rc=0
Sat 21 Nov 2026 02:09:24 EST D-em-large start
Sat 21 Nov 2026 02:10:58 EST finished rc=0
Sat 21 Nov 2026 02:10:58 EST D-om start
Sat 21 Nov 2026 02:11:37 EST finished rc=0
Sat 21 Nov 2026 02:11:37 EST D-om-large start
Sat 21 Nov 2026 02:13:20 EST finished rc=0
Sat 21 Nov 2026 02:13:20 EST A start
Sat 21 Nov 2026 02:13:20 EST B start
Sat 21 Nov 2026 02:26:54 EST finished rc=0
Sat 21 Nov 2026 02:27:42 EST finished rc=0
Sat 21 Nov 2026 02:45:02 EST D-em start
Sat 21 Nov 2026 02:45:04 EST finished rc=0
Sat 21 Nov 2026 02:45:04 EST D-em-large start
Sat 21 Nov 2026 02:45:06 EST finished rc=0
Sat 21 Nov 2026 02:45:06 EST D-om start
Sat 21 Nov 2026 02:45:08 EST finished rc=0
Sat 21 Nov 2026 02:45:08 EST D-om-large start
Sat 21 Nov 2026 02:45:09 EST finished rc=0
Sat 21 Nov 2026 02:45:09 EST A start
Sat 21 Nov 2026 02:45:09 EST B start
Sat 21 Nov 2026 02:45:11 EST finished rc=0
Sat 21 Nov 2026 02:45:11 EST finished rc=0
Sat 21 Nov 2026 02:45:11 EST C-em start
Sat 21 Nov 2026 02:46:51 EST finished rc=0
Sat 21 Nov 2026 02:46:51 EST C-om start
Sat 21 Nov 2026 02:49:23 EST RELAUNCH after the ng_step stride fix (e-means parts D-em, A, C-em rerun; B, D-om kept: reproduced with the final code)
Sat 21 Nov 2026 02:49:23 EST D-em start
Sat 21 Nov 2026 02:50:48 EST finished rc=0
Sat 21 Nov 2026 02:50:48 EST D-em-large start
Sat 21 Nov 2026 02:52:16 EST finished rc=0
Sat 21 Nov 2026 02:52:16 EST D-om start
Sat 21 Nov 2026 02:52:18 EST finished rc=0
Sat 21 Nov 2026 02:52:18 EST D-om-large start
Sat 21 Nov 2026 02:52:20 EST finished rc=0
Sat 21 Nov 2026 02:52:20 EST B start
Sat 21 Nov 2026 02:52:20 EST A start
Sat 21 Nov 2026 02:52:21 EST finished rc=0
Sat 21 Nov 2026 03:03:30 EST finished rc=0
Sat 21 Nov 2026 03:03:30 EST C-em start
Sat 21 Nov 2026 03:04:51 EST finished rc=0
Sat 21 Nov 2026 03:04:51 EST C-om start
Sat 21 Nov 2026 03:05:48 EST finished rc=0
Sat 21 Nov 2026 03:05:48 EST C-em-10m start
Sat 21 Nov 2026 03:06:02 EST finished rc=1
Sat 21 Nov 2026 03:06:02 EST C-om-10m start
Sat 21 Nov 2026 03:06:29 EST finished rc=1
Sat 21 Nov 2026 03:06:29 EST all done
Sat 21 Nov 2026 03:08:42 EST C-em-10m (after the capture-time event-query fix) start
Sat 21 Nov 2026 03:09:23 EST finished rc=0
Sat 21 Nov 2026 03:09:23 EST C-om-10m retry start
Sat 21 Nov 2026 03:09:39 EST finished rc=0
Sat 21 Nov 2026 03:09:39 EST 10m done
```

