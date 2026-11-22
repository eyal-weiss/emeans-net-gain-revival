# Net-gain revival for e-means

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24389631.svg)](https://doi.org/10.5281/zenodo.24389631)

Code, protocols and run records for *Revive What Pays: Evidence-Weighted Net-Gain Revival for Mini-Batch k-Means*
(Eyal Weiss, 2026). **The paper will appear on arXiv within a few days; this README will then link it.**

[e-means](https://arxiv.org/abs/2611.09501) (FAISS, `faiss.contrib.e_means`) periodically moves the centres of rarely
used clusters to split large ones (*revival*). FAISS [pull request #5915](https://github.com/facebookresearch/faiss/pull/5915)
proposes cost-based revival as the new default: it estimates the cost of removing each centre and the gain of splitting
each cluster. Those estimates are precise for large, long-lived clusters and mostly noise for small or young ones.
Net-gain revival moves a centre only if the predicted gain of the split it funds is **more likely than not** to exceed
the predicted cost of its removal, with every estimate shrunk toward the population of centres according to its evidence:

- rates: Gamma-Poisson posterior means;
- removal cost per point: an empirical-Bayes margin ratio times the squared distance to the nearest other centre;
- split gain: log-normal posterior **median** with a precision-weighted (Paule-Mandel) prior;
- a greedy matching of the cheapest removals with the largest gains, the pair guard of PR #5915, and a cool-down of one
  EMA half-life after each split;
- an automatic early end of the move phase once revival has stopped paying (one half-life without an accepted move).

No tuning parameter. In a pre-registered comparison against the rule of PR #5915 (14 datasets, 34 dataset-k
combinations, 20 unused seeds), it lowers the test error by 2.9% and the training error by 8.9% after two epochs, and by
5.0% and 7.1% after five. With k-means++ starts its test error is 2.4% higher (Shuttle, one isolated test point). It
takes about 7% longer than the PR's rule; the early stop saves about 5% with five or ten epochs.

**Version 1.1** follows the PR to its current driver (commit 79440bc), which moves centres until the end of training and
sorts out the last pass at the end. On that driver the rule no longer stops moving one pass before the end (the
freeze of the PR's earlier driver, now the option `freeze=True`): in a third pre-registered round, dropping the freeze
lowered the test error by 1.3% and the training error by 2.3% at two epochs and changed both by less than 1% at five.
A final comparison of the released class, with the early stop, on the same 34 combinations and seeds: without
the freeze it lowers the test error by 4.4% and the training error by 16.9% relative to the PR's rule at two epochs
(3.8% and 11.0% at five) and takes a median 7% longer (4% at five); with the freeze, 3.2% and 15.0% (3.9% and 10.6%)
for 2% (4%) more time. Dropping the freeze is therefore the default on that driver.
Version 1.1 also adds optimized GPU implementations for FAISS's GPU e-means and for o-means (`gpu/`).

## Installation

The rule runs on the driver of PR #5915, which is not in a FAISS release: at the PR's head (commit 79440bc, the
default since version 1.1), or at 98deaca, the commit of the paper's first two rounds, where `NetGainRevivalEmeans` is
version 1.0.0's class, unchanged (the package picks the class for the installed driver). Build FAISS from the commit
(CPU), then the two C++ kernels:

```bash
conda create -y -n emeans_pr5915 -c conda-forge python=3.12 "numpy>=2,<2.6" scipy "cmake>=3.24" "swig>=4.1" openblas \
    "libblas=*=*openblas" "liblapack=*=*openblas" cxx-compiler c-compiler make pip setuptools pytest
conda activate emeans_pr5915
git clone https://github.com/lcabannes/faiss-emeans.git faiss_src && cd faiss_src && git checkout 79440bc   # or 98deaca
cmake -B build -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=ON -DBUILD_TESTING=OFF -DFAISS_ENABLE_C_API=OFF \
      -DFAISS_OPT_LEVEL=avx2 -DBLA_VENDOR=OpenBLAS -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$(which python)"
make -C build -j 16 swigfaiss swigfaiss_avx2 && (cd build/faiss/python && pip install .) && cd ..
bash net_gain_revival/native/build.sh     # optional C++ kernels: faster, same results
python -m pytest tests                    # also compares with the research code if research/build.sh was run
```

## Usage

```python
import numpy as np
from net_gain_revival import NetGainRevivalEmeans

x = np.random.rand(200_000, 16).astype("float32")
km = NetGainRevivalEmeans(x.shape[1], 1024, nepochs=2)    # stat="medg_pm", early_stop="hl", freeze=False
centroids, _ = km.train(x)
print(km.n_splits, km.es_stop_step)
```

`stat="medg_fh"` uses a Fay-Herriot prior (the paper's alternative); `early_stop=None` keeps the move phase until the
end (or until the freeze), and a float `eps` also ends it when the predicted net gain of a check falls below `eps` times
the batch's squared error; `freeze=True` (head driver) stops moves once less than one pass remains, as the PR's earlier
driver did. `net_gain_revival.rule.decide` is the decision as a pure NumPy function of per-centre statistics, for other
implementations. Both classes (about 200 lines each) make the same moves and reach the same final centres, bit for bit,
as the research code that produced the paper's runs: on 98deaca, `research/verify_clean.py`
(`results/clean_equivalence.jsonl.gz`); on the head, `research/verify_head.py` and the reproduction part of the final
comparison (every run identical to round 3's records; `results/final_comparison.jsonl.gz`). This holds in every run
compared, not for every conceivable input: the C++ per-point sums add each projection's terms in order where NumPy's
`einsum` may not, and the head class computes the raw gain as ((2/pi) * scale) * |U| as version 1.0.0 does, where round
3's research class computes (2/pi) * (scale * |U|); with constructed inputs (huge cancelling coordinates, or a gain equal
to a cost to the last bit) either can break an exact tie differently.

## CPU and GPU

`NetGainRevivalEmeans` is a CPU implementation (the NumPy e-means of PR #5915, with optional C++ kernels); all results
of the paper's pre-registered rounds are CPU results. Version 1.1 adds optimized GPU implementations (`gpu/`): the rule
inside the PR's PyTorch e-means (either driver) and inside [o-means](https://owlgebra.ai/blog/o-means/), with the
per-batch statistics inside the drivers' CUDA graphs and the whole check on the device. Given the same state they make
the same decisions as the C++ decision. On one RTX 3090 Ti at two epochs they take 1.66 times as long as the PR's GPU
rule without the freeze (1.33 with it), and 1.73 times as long as o-means' own rule; these runs last 20-400 ms, so the
extra time is 10-170 ms per run, and its share falls on larger data (at 10 million points, 12% without the freeze and 7%
with it). In exploratory runs they behave like the CPU version: on the 34 combinations at two epochs, dropping the
freeze lowered the test error by 1.7% and the training error by 4.2%; in o-means the rule lowered the test error of
o-means' own rule by 32%. `gpu/README.md` gives the details, and what porting the rule to very large problems (billions
of points, millions of centres, several GPUs) would need. The unoptimized GPU proofs of concept of version 1.0.0 stay in
`research/`.

## Contents

- `net_gain_revival/`: the code to run: the class above (`emeans.py`), the decision as a pure function (`rule.py`),
  and optional C++ kernels (`native/`).
- `research/`: all the research code, kept as it produced the paper's runs and separate from the code to run: the rule on
  the PR's driver with every option that was tried (`ablation.py`, with the C++ parts in `ablation_cpp.py` and
  `ngp_kernels.cpp`), the round-2 variants (`ngp_variants.py`, `ngp_variants2.py`), the whole decision in C++
  (`ngp_fast.py`, `ngp_decide.cpp`), the early stop (`ngp_stop.py`), the decision as a pure function (`netgain_rule.py`),
  unoptimized GPU proofs of concept for FAISS's PyTorch e-means (`netgain_torch.py`) and for o-means
  (`omeans_netgain.py`; see "CPU and GPU"), and every runner and analysis. `research/README.md` maps runners to records,
  `research/MANIFEST.md` lists the code hashes, and `research/PROVENANCE.md` says which records the code reproduces and
  which options are known to be flawed or superseded.
- `results/`: the records of every run of the pre-registered rounds and of the later experiments (JSON lines; version
  1.0.0: 25,113 runs; version 1.1 adds `round3.jsonl.gz`, `final_comparison.jsonl.gz` and the GPU records in
  `gpu_optimized/`); `results/development/`: the records of the development runs (about 220,000 run and calibration
  records), made by earlier states of the research code.
- `gpu/` (version 1.1): the optimized GPU implementations, their tests and tools (see `gpu/README.md`).
- `docs/technical_report.pdf`: every alternative tried during development, what happened and why, with pointers to the
  records; read it to understand the released data. `docs/protocols/`: the pre-registered protocols (rounds 1-3), the
  analysis plan of the final comparison, the specification of the round-2 variants, and an index of everything tried.

The experiments ran on two machines, named `host_a` and `host_b` in the code and records. The rule builds on
[distance-aware revival](https://github.com/eyal-weiss/emeans-distance-revival) (arXiv:2611.18238), the earlier paper's
code. Third-party notices: `NOTICE`.

## Citation

The code and data are archived on Zenodo: version 1.0.0, [10.5281/zenodo.24389632](https://doi.org/10.5281/zenodo.24389632);
all versions, [10.5281/zenodo.24389631](https://doi.org/10.5281/zenodo.24389631).

```bibtex
@misc{weiss2026netgain,
  title  = {Revive What Pays: Evidence-Weighted Net-Gain Revival for Mini-Batch k-Means},
  author = {Weiss, Eyal},
  year   = {2026},
  note   = {arXiv identifier to be added}
}
```

## License

MIT (see `LICENSE`).
