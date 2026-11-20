"""Net-gain revival for e-means ("Revive What Pays: Evidence-Weighted Net-Gain Revival for Mini-Batch k-Means",
Weiss, 2026).

    from net_gain_revival import NetGainRevivalEmeans
    km = NetGainRevivalEmeans(d, k, nepochs=2)            # stat="medg_pm", early_stop="hl"
    centroids, _ = km.train(x)                            # x: float32 (n, d); the interface of FAISS's NumpyEmeans

Requires FAISS built from pull request #5915 at commit 98deaca (README.md). The optional C++ kernels
(native/build.sh) make the per-point sums and the decision faster; the results are the same without them.
rule.decide is the decision as a pure function of per-centre statistics, for other implementations.
"""
from . import rule
from .emeans import NetGainRevivalEmeans

__all__ = ["NetGainRevivalEmeans", "rule"]
