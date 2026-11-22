# The net-gain rule on the PR head driver (79440bc): verification of the port (2026-11-21)

Code: `code/experiments/gpu_fast/netgain_torch_fast.py` (one class for both drivers; `HEAD` when the head's
`faiss.contrib.e_means` is on the path), reference `netgain_torch_poc_head.py` (proof of concept on the head: plain
torch statistics, NumPy reference decision on the host, the driver's own `_apply_splits`, eager training). Runs on
host_a (RTX 3090 Ti), records in this directory; summary by `check_head.py`.

* **Fast class vs proof of concept** (both trained eagerly, so the driver's eager path is the same for both; `_e`
  configs): bit-identical final centroids in every mode (freeze / no freeze, with / without early stop, with / without
  the closing handover) on 9 of the 14 runs at T = 2 (shuttle, geonames, mnist and power at 1024, both seeds; codrna
  4096 seed 31) and on all 6 at T = 5 (shuttle, kdd, geonames at 1024, 2 seeds): 15 runs per mode
  (`eq_T2.jsonl`, `eq_T5.jsonl`); the other 5 are explained below.
  Early stop fired in 20 of these runs (geonames, power), at the same step in both.
  On kdd 4096, codrna 4096 and deep1m 16384 the two differ through the nearest-centroid search alone: with the
  proof of concept's search (FP32 faiss.knn on the host; config `ff_e` / `ffn_e`) the fast class is bit-identical to it
  on all 6 runs in both modes (`ff.jsonl`). faiss.knn's FP32 distances depend on the BLAS thread count (the proof
  of concept itself made 2969 splits with 4 threads and 3043 with 12 on kdd 4096 seed 30); the device search is
  deterministic and closer to exact (README: nearest-other-centroid search).
* **Device decision vs the C++ decision** on the same state (`fboth`, `fbothn`): 1810/1810 checks
  (T = 2: 344 + 694; T = 5: 342 + 430). Whole runs with the device decision equal those with the host (C++) decision:
  20/20 (freeze) and 20/20 (no freeze).
* **Graph replay vs eager training** differ in the last bits for the PR's own class as well (kdd 1024: same splits,
  same errors to 6 digits; `eager2.jsonl`): a property of the driver.
