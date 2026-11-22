# Net-gain revival on the GPU: GPU e-means and o-means

Optimized GPU implementations of the rule (added in version 1.1), inside two GPU k-means drivers:

- `netgain_torch_fast.NetGainEmeansFast`: the PyTorch e-means of FAISS pull request #5915
  (`faiss.contrib.e_means.e_means_torch.Emeans`), at the PR's head 79440bc or at 98deaca (whichever is on the path);
- `omeans_fast.OmeansNetGainFast`: [o-means](https://owlgebra.ai/blog/o-means/) (`faiss.contrib.o_means`, commit
  9ecbf0c of its repository).

Given the same per-centre state, the decision is the same as the CPU package's (`net_gain_revival.rule.decide`, and its
C++ port `net_gain_revival.native.decide`), which makes the same moves as the research code of the paper. The GPU
classes differ from the CPU classes only through the driver they run in (and, in o-means, its approximate search).

## How it works

- **Per batch**: the five per-centre sums of the decision (counts, margins d2 - d1, squared margins, p^2, p^4 of each
  point's projection on its cluster's split axis) are computed row-parallel in FP64 and added per centre by a
  deterministic segmented sum (`netgain_triton.py`: label-sorted, fixed order, no atomics), inside the driver's captured
  CUDA graph; the other per-centre state (points and batches since birth, inherited weight, prod(1 - alpha), the
  early-stop clock) is updated by one fused kernel. In o-means, the margin comes from modified copies of o-means' own
  two-level search kernels (the best two scores among the probed cells), so it is approximate where o-means' search is.
- **Per check** (every five batches), on the device and captured as a CUDA graph: every centre's nearest other centre
  (an FP32 tensor-core screen with a rigorous error bound, then FP64 among the kept candidates: the distance found is
  within about 5e-6 relative of the true minimum, and in every test it was the exact nearest centre; `nearest_other_state`), the estimates in five fused Triton kernels (`netgain_fused.py`), the
  Paule-Mandel prior (Newton's method from an FP32 pre-solve, then safeguarded FP64 Newton), the greedy matching
  (`netgain_device.py`), and the state updates after the decision (one kernel). The split is the driver's own kernel.
- **Options** as in the CPU class: `stat` ("medg_pm", the paper's rule, or "medg_fh"), `early_stop` ("hl" or None),
  `freeze` (default None: no freeze on the head driver, the freeze on 98deaca, where a centre moved in the last pass
  would keep its running position; o-means: `freeze=True` by default, the configuration evaluated there), and `check`
  ("device"; "host" = the C++ decision on the host after one copy; "both" = tests).

## Requirements and use

PyTorch with CUDA and Triton (tested with PyTorch 2.15, CUDA 13.2 and Triton 3.9 on one RTX 3090 Ti); the
`contrib/e_means` files of pull request #5915 (for GPU e-means) or the o-means package; the CPU package of this
repository with its C++ kernels built (`net_gain_revival/native/build.sh`), which the "host" and "both" checks and the
tests use.

```python
import torch
from netgain_torch_fast import NetGainEmeansFast        # this directory on sys.path, the repository root too

x = torch.randn(1_000_000, 96, device="cuda")
km = NetGainEmeansFast(96, 16384, nepochs=2)           # early_stop="hl"; no freeze on the head driver
centroids, _ = km.train(x)
```

`fast_run.py` runs the paper's GPU experiments (configurations `pr`, `ng` (freeze), `ngn` (no freeze), `om`, `om_ng`,
and the test configurations listed in its docstring; with `EXPERIMENTS_DIR` pointing to `research/`).

## Verification

- **Against independent proofs of concept** (`research/netgain_torch.py` for 98deaca, `netgain_torch_poc_head.py` for
  the head: plain PyTorch statistics, the NumPy decision, eager): final centres bit-identical in every mode on the head
  (freeze or not, early stop or not, closing handover or not; 15 runs per mode at 2 and 5 epochs); the other runs
  (k >= 4,096) differ only through the nearest-centre search, and with the proof of concept's FP32 `faiss.knn` they are
  bit-identical too. On 98deaca: 12 of 12 runs bit-identical without the early stop.
- **The device decision equals the C++ decision** given the same state in every check compared (more than 5,000 checks
  over e-means on both drivers and o-means).
- **Unit tests** (`tests/`): the fused estimates against the PyTorch path on 216 random and edge-case states (including
  strided inputs and NaN and underflow cases); the Paule-Mandel kernel against NumPy's reference on 241 cases; the
  nearest-centre search against an exact FP64 brute force on adversarial and trained sets.
- Three independent code reviews; every defect they found was fixed and re-verified. Two bugs were found by the
  comparisons above before any result was reported (a strided read in the per-step kernel, and in the fused estimates
  for o-means); the records made with the first one are kept as `*_INVALID_stepbug.jsonl`.

`tools/check_head.py` and `tools/check_v2.py` summarize the verification records in `results/gpu_optimized/`.

## Cost (one RTX 3090 Ti, 2 epochs)

Time of the rule divided by the time of the driver's own rule (minimum of three fits per seed, median over seeds,
geometric mean over 14 dataset-k combinations; `results/gpu_optimized/gpu_v3_timing/analysis.md`):

| | all 14 | 1M points, k = 16,384 |
|---|---|---|
| GPU e-means, head driver, no freeze (default) | 1.66 | 1.67 |
| GPU e-means, head driver, freeze | 1.33 | 1.37 |
| GPU e-means, 98deaca | 1.38 | 1.30 |
| o-means | 1.73 | 3.17 |

These runs take 20-400 ms, so the extra time is 10-170 ms per fit. Two parts make it: per-batch statistics, which grow
with the data like the driver's own work, and per-check work, whose number grows slowly with the data in e-means (6 checks
at 1M points and 7 at 10M with the freeze, 12 and 15 without it). On 10 million points (DEEP, k = 16,384, the GPU not
shared with other jobs) the overhead is 7% with the freeze and 12% without it (`gpu_scale/analysis_scale.md`), and on
larger data it should approach the share of the per-batch statistics (a few percent). o-means checks every five batches
of a fixed size, so its number of checks grows with the data and its overhead stays near 3 times at k = 16,384 (3.2 at
1M points, 2.9 at 10M).

## Scope, and what very large problems would need

Tested on one consumer GPU (24 GB, FP64 at 1/64 of the FP32 rate), up to 10 million points, k up to 16,384, d up to
784, FP32 inputs, the PR's Triton assignment kernels and exact assignment. The e-means paper's largest runs (H100 GPUs,
1 to 10 billion points, k up to 16.7 million, d = 1,024, approximate IVF assignment, BF16/FP8/INT8 inputs, several
GPUs) are outside what was tested. Porting the rule there needs:

1. **The PR's scale-out classes.** The rule is built into the single-GPU class. The PR's classes for data split across
   GPUs, centroids split across GPUs, IVF assignment and 8-bit inputs do not include it: the per-centre sums would have
   to be reduced across GPUs, and the check run on sharded centroids. With FP8 or INT8 inputs the class currently falls
   back to the host check.
2. **Second-nearest distances.** The statistics need each point's second-nearest distance, which those classes do not
   compute. Their kernels would need to keep the second-best score, as was done here for o-means; under approximate
   (IVF) assignment it is itself approximate.
3. **The nearest-centre search of the check.** It is exact and compares all pairs of centres (k^2 d): seconds per check
   at k = 1 million and d = 96 on an H100, but minutes at k = 2 million and d = 1,024, and hours at k = 16.7 million
   (where the centres alone need about 70 GB). An approximate search over the index of the centres that the PR's IVF
   assignment already builds would replace it.
4. **Tuning** on the target GPU (block sizes; the PR's CuTe kernels for H100 have not been run with these classes).

Two questions are open beyond the engineering: whether the decisions stay good when the margins come from an
approximate assignment, and whether the rule pays at that scale at all: with k = 16,384 it is level with the PR's rule
on test error at 1M points without the freeze and slightly worse with it (`gpu_scale/analysis_1m.md`), and at 10M points
the three rules are within 0.06% of each other (`gpu_scale/analysis_scale.md`).

## Files

| file | what |
|---|---|
| `netgain_torch_fast.py` | `NetGainEmeansFast` (GPU e-means, both drivers) |
| `omeans_fast.py` | `OmeansNetGainFast` (o-means) |
| `netgain_device.py`, `netgain_fused.py` | the decision on the device: estimates, Paule-Mandel, matching |
| `netgain_triton.py` | per-row values, segmented sums, the nearest-other-centre search, the fused state updates |
| `netgain_torch_poc_head.py` | the independent proof of concept on the head driver (reference for the tests) |
| `fast_run.py` | the runner of the GPU experiments |
| `tests/` | unit tests (need a CUDA GPU; `test_kernel_interp.py` also runs in Triton's interpreter) |
| `tools/` | verification summaries, analyses, benchmarks |

Records: `results/gpu_optimized/` (`gpu_final`: the first optimized version on 98deaca and o-means, accuracy on 34
cells and timing; `gpu_head`: the port to the head driver; `gpu_v2`: re-verification after the faster decision and the
exploratory accuracy of the freeze option on the head; `gpu_v3_timing`, `gpu_v4_timing`: timing of the final code;
`gpu_scale`: accuracy and timing at 1M and 10M points with k = 16,384).
