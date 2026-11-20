# Provenance of the research code and records

## What the research code is
Each file is in the state in which it produced the paper's confirmatory runs and the later runs (see `MANIFEST.md`).
Bugs found during development were fixed in these files before the runs that the paper reports; no released file is an
older, buggy version. The files keep configuration switches for options that were tried and abandoned (they are off in
the paper's configurations), so that the development results described in the technical report can be traced.

## Which records the released code reproduces
- `results/round1`, `results/round2`, `results/side_*`, `results/gpu_and_omeans`: made by exactly these files (their
  records carry the code hashes listed in `MANIFEST.md`; the only differences are the replaced host names). Runs are
  deterministic on a given machine; KDD Cup 99 gives slightly different results on different processors, which is why
  every comparison was run on one machine.
- `results/development/*`: made during development by earlier states of these files, which were not archived. Many of
  them are reproduced by the configurations that the files still contain (the development notes recorded exact
  reproductions after several changes), but not all are guaranteed to be. They document what we observed and are
  labelled exploratory in the technical report.

## Options in the code that are known to be flawed or superseded (all off in the paper's configurations)
- `matcher="v1"`: the first greedy matching, which could discard a protected candidate as a donor; replaced by `"v2"`
  (used by every net-gain configuration) and kept so that the earliest net-gain runs can be traced.
- `removal="ng_post3"` and its options (`memory`, `gain_rate`, `rate_mode`): the gain measured from the split; it churns
  (technical report, section "Dead ends").
- `confirm=True`, `confirm_adapt`, `confirm_wait="evidence"`: confirming moves before making them; worse.
- `gain_noise="measured"`, `gain_eb="tweedie"`, `gain_noise="compound"`, `moderate=True`: corrections that did not help.
- `trigger="event"` and `trigger="every"`: exact but saves nothing / unstable with posterior means.
- `margin_stride` > 1: sampling of the two-nearest search; dropped.
- `stat="mean"` (configuration `ngp2_full`): the first version, which runs away with long schedules (paper, Section 4.1).

## Numerical notes
- The Paule-Mandel bisection starts at log tau^2 = -30, so a root below e^-30 is returned as about e^-30 (the posterior
  then floors tau^2 at 1e-6).
- The C++ decision (`ngp_decide.cpp`) reproduces the NumPy reference bit for bit with NumPy 2.5 and OpenBLAS on CPUs
  without AVX-512; with AVX-512, NumPy's own exponential and logarithm differ in the last bits and only the decisions are
  guaranteed identical (they were in all 306 test jobs). NumPy 1.x chunks long sums differently.

## Errors found and corrected during the project (none affects a released record)
- First matching (v1) could discard a protected candidate: found by review before the posterior rules were run; v2.
- An early variant of the measured-gain rule produced NaNs that blocked moves (its apparent success was an artefact);
  fixed, and its results were discarded.
- The runner of the second round: several issues found in review before launch (equivalence gate, provenance checks,
  resume identity); fixed before any run of that round.
- The GPU runner read a property of the PyTorch class after training, when it is no longer defined; fixed before the
  GPU grid (only a smoke test had failed).
- Two analysis scripts (not runners) had errors that were fixed before their results were used: an underflow of a
  decay product in the move-phase analysis, and an inconsistent timing scope in the first C++ equivalence grid (whose
  records are therefore not released; `side_cpp_equivalence` is the corrected second grid).
