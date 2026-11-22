# Runners and analyses

Run everything from this folder with this folder as the experiments directory (the runners import the modules by plain
names and read `grid.py` and `datasets.py` from `$EXPERIMENTS_DIR`):

```bash
cd research
export EXPERIMENTS_DIR=$PWD EMEANS_RR_DATA=/path/to/data NGP_IMPL=cpp OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
```

`NGP_IMPL=cpp` uses the C++ per-point sums and matching (identical results to `NGP_IMPL=py`, the pure Python reference).

**Data.** `python -I datasets.py prepare <name> --raw RAW_DIR --out $EMEANS_RR_DATA` for KDD Cup 99, Census 1990,
YearPredictionMSD, GeoNames, household power and Shuttle (`kdd_z census_z msd_z geonames power_z shuttle_z`), and
`python -I datasets_extra.py prepare <name> --raw RAW_DIR --out $EMEANS_RR_DATA` for the eight datasets added for round 2
(`datasets_extra.py urls <name>` prints the download links). Every record stores the SHA-256 of its training data.

**Which runner produced what.** Configuration names in the code: `pr` = cost-based revival of PR #5915, `lib` = the count
rule of FAISS 1.16, `ours` = distance-aware revival, `h21` = cost-based revival with the distance test,
`ngp2_full` = the first version of the rule (posterior means), `ngp2_full_medg_pm` = the rule of the paper,
`ngp2_full_medg_fh` = its Fay-Herriot variant.

| paper | runner | analysis | records (`../results/`) |
|---|---|---|---|
| round 1 (`ngp2_full` frozen) | `confirm_grid.py` | `cfm_analyze.py` | `round1` |
| round 2 (`ngp2_full_medg_pm`, `ngp2_full_medg_fh`) | `confirm2_grid.py` | `confirm2_analyze.py` | `round2` |
| diagnosis of the first version | `diag_t5.py`, `ext_grid.py`, `ext2_grid.py`, `ext3_grid.py` | | (development seeds) |
| C++ decision | `decide_equiv.py` | `deq_analyze.py` | `side_cpp_equivalence` |
| move-phase log and early stop | `stopmeasure_grid.py`, `stop_grid.py` | `stopm_analyze.py`, `stop_analyze.py` | `side_move_phase_log`, `side_early_stop` |
| check interval, newer driver | `interval_grid.py` with `ngp_interval.py`; `prhead_grid.py` with `ngp_head.py` | `side_analyze.py` | `side_interval`, `side_pr_head` |
| GPU e-means and o-means | `gpu_netgain_run.py` with `netgain_torch.py`, `omeans_netgain.py` | `side_analyze.py` | `gpu_and_omeans` |
| decision as a pure function | `netgain_rule_check.py` (4,926 of 4,926 checks identical) | | |
| round 3 (check interval; the freeze on the newer driver) | `round3/interval_grid.py` with `ngp_interval.py`; `round3/round3_prhead.py` with `round3/prhead_grid.py` and `ngp_head.py`; `round3/launch_r3.sh`, gate `round3/r3_gatecheck.py`, power analysis `round3/power/` | `round3/confirm3_analyze.py` | `round3` |
| final comparison (version 1.1's class, early stop, freeze and no freeze) | `final_grid.py`, `final_launch.sh`; checks `verify_head.py`, `final_check.py` | `final_analyze.py` | `final_comparison` |
| optimized GPU e-means and o-means | `../gpu/fast_run.py` | `../gpu/tools/` | `gpu_optimized/` |

The runners write one JSON file per run (atomically; a stopped grid resumes where it stopped and refuses records made
with other code; it does not check that a resumed run uses the same machine or thread count, so resume with the same
settings). `--host` must be one of the names in the placement table of `confirm2_grid.py` (`host_a`, `host_b`): for
another name a runner schedules nothing. The newer-driver runs need the PR's head commit 79440bc as an overlay (`FAISS_OVERLAY`, see
`prhead_grid.py`); the GPU runs need PyTorch with Triton and, for o-means, the o-means package.
