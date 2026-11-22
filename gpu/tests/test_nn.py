"""Nearest-other search (netgain_triton.nearest_other_all) against an exact FP64 brute force (direct differences), on
real trained centroids and high-d adversarial sets. (Written 2026-11-21 for a TF32 screen variant, which was exact but
not faster on trained centroids and was dropped; NG_NN_TF32 has no effect on the current search.)
A query is correct when its returned squared distance (FP64 of the FP32 centroids) equals the exact minimum.
usage: python test_nn.py [real]"""
import os, sys
import numpy as np, torch
import netgain_triton as NT
dev = torch.device("cuda")


def exact_min(C):
    """Exact minimum by direct FP64 differences (no Gram-formula cancellation), in chunks of queries."""
    Cd = C.double()
    k, d = C.shape
    step = max(1, 2**26 // (k * d))
    best = torch.empty(k, dtype=torch.float64, device=dev)
    for a in range(0, k, step):
        D = (Cd[a:a + step, None, :] - Cd[None, :, :]).square().sum(-1)
        D[torch.arange(len(D)), torch.arange(a, a + len(D))] = float("inf")
        best[a:a + len(D)] = D.min(1).values
    return best


def check(name, C):
    C = C.float().contiguous()
    res = {}
    for flag in ("1", "0"):
        os.environ["NG_NN_TF32"] = flag
        nn, delta = NT.nearest_other_all(C)
        res[flag] = (nn, delta)
    nn1, d1 = res["1"]
    nn0, d0 = res["0"]
    ref = exact_min(C)
    ok1 = int((d1 == ref).sum())
    same = int(((nn1 == nn0) | (d1 == d0)).sum())
    print(f"{name:34s} k={len(C):6d} d={C.shape[1]:4d}: TF32 path exact for {ok1}/{len(C)}; same distance as TF32x3 "
          f"for {same}/{len(C)}", flush=True)
    return ok1 == len(C) and same == len(C)


g = torch.Generator(device="cpu").manual_seed(0)
allok = True
for d in (32, 64, 128, 784):
    k = 4096
    allok &= check(f"gaussian d={d}", torch.randn(k, d, generator=g).to(dev))
    X = torch.randn(k // 8, d, generator=g).repeat(8, 1)                       # exact duplicates
    allok &= check(f"8x duplicates d={d}", X.to(dev))
    X = torch.randn(k // 2, d, generator=g); X = torch.cat([X, X + 1e-6 * torch.randn(k // 2, d, generator=g)])
    allok &= check(f"near duplicates (1e-6) d={d}", X.to(dev))
    X = 1e3 + 1e-2 * torch.randn(k, d, generator=g)                              # far from the origin, tight
    allok &= check(f"offset 1e3 spread 1e-2 d={d}", X.to(dev))
    X = torch.randn(k, d, generator=g) * torch.logspace(-3, 3, k)[:, None]      # mixed scales
    allok &= check(f"mixed scales d={d}", X.to(dev))
if len(sys.argv) > 1:
    sys.path.insert(0, os.environ["EXPERIMENTS_DIR"])
    import grid; grid._init(8)
    import datasets
    from faiss.contrib.e_means.e_means_torch import Emeans
    for ds, k in (("deep1m", 16384), ("spacev1m", 16384), ("kdd_z", 4096), ("codrna_z", 4096), ("mnist_255", 1024),
                  ("census_z", 4096), ("msd_z", 4096), ("higgs_z", 1024), ("fashion_255", 1024)):
        X, _, _ = datasets.load(ds, 0)
        Xg = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).to(dev)
        rows = np.sort(np.random.RandomState(30).permutation(len(X))[:k])
        m = Emeans(X.shape[1], k, nepochs=2, seed=30)
        C, _ = m.train(Xg, init_centroids=Xg[torch.from_numpy(rows).to(dev)])
        allok &= check(f"{ds} trained", C)
print("ALL OK" if allok else "SOME MISMATCH")
