# Changelog

Versions are tagged; each release is archived on Zenodo with its own DOI. Records, research code and protocols of a
released version are never changed later; new studies are added next to them.

## v1.1.0 (2026-11)
Follows pull request #5915 to its current driver (commit 79440bc), and adds the third pre-registered round, a final
comparison of the released class, and optimized GPU implementations.
- `net_gain_revival/emeans.py`: the class for the PR's head driver. By default the rule now keeps moving until the end
  (the driver's closing handover sorts out the last pass), as the driver's own rule does; `freeze=True` restores the
  freeze of the earlier driver (no move once less than one pass remains). Same decisions as the research code of round
  3 (`research/ngp_head.py`), bit for bit, with the NumPy computations and with the C++ kernels.
- `net_gain_revival/emeans_98deaca.py`: version 1.0.0's class, unchanged; `NetGainRevivalEmeans` is the class for the
  installed driver. `native/ngp_head.cpp`: the per-point sums from the driver's own two-nearest search (the other C++
  files are unchanged). Tests for both drivers.
- `gpu/`: optimized GPU implementations for the PR's PyTorch e-means (either driver) and for o-means, with tests, tools
  and the records in `results/gpu_optimized/`.
- Round 3 (check interval; the freeze on the newer driver): protocol `docs/protocols/round3_protocol.md`, code
  `research/round3/`, records `results/round3.jsonl.gz`.
- Final comparison of the released class with the early stop, with and without the freeze, against the PR's rule:
  plan `docs/protocols/final_comparison_plan.md`, code `research/final_*.py` and `research/verify_head.py`, records
  `results/final_comparison.jsonl.gz`.

## v1.0.0 (2026-11)
Archived on Zenodo: [10.5281/zenodo.24389632](https://doi.org/10.5281/zenodo.24389632).
First release, accompanying the paper "Revive What Pays: Evidence-Weighted Net-Gain Revival for Mini-Batch k-Means":
the clean implementation (`net_gain_revival/`), the research code of the paper's runs (`research/`), the records of the
two pre-registered rounds, the later experiments and the development runs (`results/`), the protocols and the
technical report (`docs/`).
