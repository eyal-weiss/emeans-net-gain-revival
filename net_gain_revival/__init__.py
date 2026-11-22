"""Net-gain revival for e-means ("Revive What Pays: Evidence-Weighted Net-Gain Revival for Mini-Batch k-Means",
Weiss, 2026).

    from net_gain_revival import NetGainRevivalEmeans
    km = NetGainRevivalEmeans(d, k, nepochs=2)            # stat="medg_pm", early_stop="hl", freeze=False
    centroids, _ = km.train(x)                            # x: float32 (n, d); the interface of FAISS's NumpyEmeans

Requires FAISS built from pull request #5915 (README.md). NetGainRevivalEmeans is the class for the installed driver:
emeans.NetGainRevivalEmeans for the PR's head (79440bc; moves until the end by default, freeze=True as an option), or
emeans_98deaca.NetGainRevivalEmeans for 98deaca (version 1.0.0's class, unchanged: always with the PR's freeze), which
reproduces the paper's first two rounds. The optional C++ kernels (native/build.sh) make the per-point sums and the
decision faster; the results are the same without them. rule.decide is the decision as a pure function of per-centre
statistics, for other implementations.
"""
from faiss.contrib.e_means import NumpyEmeans as _Driver

from . import rule

if hasattr(_Driver, "_handover_start"):
    from .emeans import NetGainRevivalEmeans
else:
    from .emeans_98deaca import NetGainRevivalEmeans

__all__ = ["NetGainRevivalEmeans", "rule"]
