# The research code

These files are the code that produced the paper's runs. Released files are identical to the originals except that the
names of the two machines are replaced by `host_a` and `host_b` (in eight files: runners, one analysis and the placement tables); the
records name the originals, whose hashes are listed. `ngp_decide.cpp` differs from the version of the equivalence runs
(`side_cpp_equivalence`) by one comment line (the compiled library is byte-identical). `build.sh`, `verify_clean.py`,
`README.md`, `PROVENANCE.md` and this file are new.

| file | sha256[:16] released | sha256[:16] in the records | difference |
|---|---|---|---|
| `ablation.py` | bd0f15eb4d0a6641 | bd0f15eb4d0a6641 |  |
| `ablation_cpp.py` | bf6af35b14df3026 | bf6af35b14df3026 |  |
| `cfm_analyze.py` | b9d66135589227b3 | b9d66135589227b3 |  |
| `confirm2_analyze.py` | 26d8767d665ee60d | 26d8767d665ee60d |  |
| `confirm2_grid.py` | a7d3051c53ac6840 | 43c4d1907957f214 | host names replaced |
| `confirm_grid.py` | ff5797d29e329a50 | 495df544e1b8b426 | host names replaced |
| `datasets.py` | 568dd5ca2acf359d | 568dd5ca2acf359d |  |
| `datasets_extra.py` | be609c0fc22f4594 | be609c0fc22f4594 |  |
| `decide_equiv.py` | 4d3157fdf7289296 | 4d3157fdf7289296 |  |
| `deq_analyze.py` | f334784f105ca2cb | be44ea0762595ee2 | host names replaced |
| `diag_t5.py` | d41151e7b7fbd07a | d41151e7b7fbd07a |  |
| `ext2_grid.py` | d6b32377cd0d5bad | 2163d55e788dda35 | host names replaced |
| `ext3_grid.py` | 7bf50c60e2ba75c9 | 257175bad444d992 | host names replaced |
| `ext_grid.py` | ecfa345be94ae3ad | 1ed4cd97a97aaa8b | host names replaced |
| `gpu_netgain_run.py` | a9a0eec3c37c6cf4 | a9a0eec3c37c6cf4 |  |
| `grid.py` | 642a5dac414291bc | 4c0db0d4bb71ec55 | host names replaced; dates without clock times in two comments |
| `interval_grid.py` | 77eda941270d4c2e | 77eda941270d4c2e |  |
| `netgain_rule.py` | bdb2e0e9f0ee0f14 | bdb2e0e9f0ee0f14 |  |
| `netgain_rule_check.py` | 157eff0c1c5fddba | 157eff0c1c5fddba |  |
| `netgain_torch.py` | 1f70fe1327115572 | 1f70fe1327115572 |  |
| `ngp_decide.cpp` | 6b8f312a3dc826e4 | 6b8f312a3dc826e4 |  |
| `ngp_fast.py` | 4fce8e2c516dafb9 | 4fce8e2c516dafb9 |  |
| `ngp_head.py` | 69dbabd4318f2ce0 | 69dbabd4318f2ce0 |  |
| `ngp_interval.py` | d5d0a9d114bcbcd3 | d5d0a9d114bcbcd3 |  |
| `ngp_kernels.cpp` | be826b95e29abbd7 | be826b95e29abbd7 |  |
| `ngp_stop.py` | 314d70d535550769 | 314d70d535550769 |  |
| `ngp_variants.py` | 2206ca5608fd6c87 | 2206ca5608fd6c87 |  |
| `ngp_variants2.py` | 15999ad7b2e45d9d | 15999ad7b2e45d9d |  |
| `omeans_netgain.py` | c1e218c7e5f9a31a | c1e218c7e5f9a31a |  |
| `prhead_grid.py` | e3ab35e7b70e9dea | e3ab35e7b70e9dea |  |
| `side_analyze.py` | 06aa2e0ab72f6059 | 06aa2e0ab72f6059 |  |
| `stop_analyze.py` | 0b121fe4dd4c50cf | 0b121fe4dd4c50cf |  |
| `stop_grid.py` | 94206a093157183d | 37ec95f61866ac2e | host names replaced |
| `stopm_analyze.py` | 8ea8afc4eba03e75 | 8ea8afc4eba03e75 |  |
| `stopmeasure_grid.py` | 614d06213477c5da | 614d06213477c5da |  |
| `verify_clean.py` | aecc2f43ea21b568 | - | new (not used in the runs) |

## Added in version 1.1

Round 3 ran the files below in `round3/` (two of them are newer versions of `interval_grid.py` and `prhead_grid.py`
above, which stay as they were for the earlier runs) together with `ngp_interval.py`, `ngp_head.py`, `netgain_rule.py` and
the other round-2 files above. The final comparison ran `final_grid.py` with version 1.1's package (`net_gain_revival/`).

| file | sha256[:16] released | sha256[:16] in the records | difference |
|---|---|---|---|
| `round3/interval_grid.py` | d8661261e5bb586a | d8661261e5bb586a |  |
| `round3/prhead_grid.py` | bab5ed3e080c41d3 | bab5ed3e080c41d3 |  |
| `round3/round3_prhead.py` | 31ae07126f2abf6a | 31ae07126f2abf6a |  |
| `round3/r3_gatecheck.py` | 2a1fc09a960615ec | 2a1fc09a960615ec |  |
| `round3/launch_r3.sh` | 39460f1400d16cda | acb918fa4d7b4630 | host names and paths replaced |
| `round3/confirm3_analyze.py` | 73f0170bf7241e47 | b71360c2f93f2fa0 | host names replaced |
| `round3/power/power_calibration.py` | 4f2190ffe3f39460 | - | the power analysis (not a runner) |
| `final_grid.py` | 14c764365b826c50 | 14c764365b826c50 |  |
| `final_launch.sh` | ca78961c26650be9 | 62fb3f10eea2a70b | host names and paths replaced |
| `final_check.py` | 3d60ebddea482e26 | - | new (a check, not a runner) |
| `final_analyze.py` | f018ca176245a334 | - | new (analysis) |
| `verify_head.py` | 62a18c456476be94 | - | new (a check, not a runner) |
