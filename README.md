# Net-gain revival for e-means

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

## Installation

The rule runs on the driver of PR #5915 at commit 98deaca, which is not in a FAISS release. Build FAISS from that commit
(CPU), then the two C++ kernels:

```bash
conda create -y -n emeans_pr5915 -c conda-forge python=3.12 "numpy>=2,<2.6" scipy "cmake>=3.24" "swig>=4.1" openblas \
    "libblas=*=*openblas" "liblapack=*=*openblas" cxx-compiler c-compiler make pip setuptools pytest
conda activate emeans_pr5915
git clone https://github.com/lcabannes/faiss-emeans.git faiss_src && cd faiss_src && git checkout 98deaca
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
km = NetGainRevivalEmeans(x.shape[1], 1024, nepochs=2)    # stat="medg_pm", early_stop="hl"
centroids, _ = km.train(x)
print(km.n_splits, km.es_stop_step)
```

`stat="medg_fh"` uses a Fay-Herriot prior (the paper's alternative); `early_stop=None` keeps the move phase until the
PR's freeze, and a float `eps` also ends it when the predicted net gain of a check falls below `eps` times the batch's
squared error. `net_gain_revival.rule.decide` is the decision as a pure NumPy function of per-centre statistics, for
other implementations. `NetGainRevivalEmeans` (about 200 lines, `net_gain_revival/emeans.py`) makes the same moves and
reaches the same final centres, bit for bit, as the research code that produced the paper's runs
(`research/verify_clean.py`; `results/clean_equivalence.jsonl.gz`).

## CPU and GPU

`NetGainRevivalEmeans` is a CPU implementation (the NumPy e-means of PR #5915, with optional C++ kernels); all results
of the paper's two pre-registered rounds are CPU results. For GPU we provide only first, simple working versions, to check
that the rule carries over: `research/netgain_torch.py` for the PR's PyTorch e-means and `research/omeans_netgain.py` for
[o-means](https://owlgebra.ai/blog/o-means/). In exploratory runs on one GPU they behaved like the CPU version (training
error 31% lower than the PR's GPU rule on 12 dataset-k combinations; test error 67% lower than o-means' own rule on 9),
but they are **not optimized**: the decision runs on the host, the per-point statistics use plain PyTorch operations, and
the PyTorch e-means version gives up the library's CUDA-graph replay. They take about 2.3 times as long as the PR's GPU
rule without CUDA graphs (about 9 times with them) and about 5 times as long as o-means. An optimized GPU implementation
(statistics in the assignment kernels, the decision on the device) is future work.

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
- `results/`: the records of every run of the two pre-registered rounds and of the later experiments (JSON lines,
  25,113 runs); `results/development/`: the records of the development runs (about 220,000 run and calibration records),
  made by earlier states of the research code.
- `docs/technical_report.pdf`: every alternative tried during development, what happened and why, with pointers to the
  records; read it to understand the released data. `docs/protocols/`: the two pre-registered protocols, the
  specification of the round-2 variants, and an index of everything tried.

The experiments ran on two machines, named `host_a` and `host_b` in the code and records. The rule builds on
[distance-aware revival](https://github.com/eyal-weiss/emeans-distance-revival) (arXiv:2611.18238), the earlier paper's
code. Third-party notices: `NOTICE`.

## Citation

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
