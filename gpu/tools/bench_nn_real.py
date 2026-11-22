"""A/B timing of nearest_other_all on trained centroids (the PR's GPU e-means, 2 epochs, seed 30): previous search
(netgain_triton_prev) vs the new one (TF32 screen, TF32x3 cascade, compacted second tier); alternate graph replays."""
import os, statistics, sys, time
import numpy as np, torch
sys.path.insert(0, os.environ["EXPERIMENTS_DIR"])
import grid; grid._init(8)
import datasets
from faiss.contrib.e_means.e_means_torch import Emeans
import netgain_triton as NT
import netgain_triton_prev as NP
from bench_nn import graph_of
dev = torch.device("cuda")
for ds, k in (("deep1m", 16384), ("spacev1m", 16384), ("kdd_z", 4096), ("census_z", 4096), ("msd_z", 4096),
              ("geonames", 4096), ("mnist_255", 1024), ("fashion_255", 1024), ("kdd_z", 1024), ("msd_z", 1024),
              ("higgs_z", 1024), ("census_z", 256), ("shuttle_z", 256)):
    X, _, _ = datasets.load(ds, 0)
    Xg = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).to(dev)
    rows = np.sort(np.random.RandomState(30).permutation(len(X))[:k])
    C, _ = Emeans(X.shape[1], k, nepochs=2, seed=30).train(Xg, init_centroids=Xg[torch.from_numpy(rows).to(dev)])
    C = C.float().contiguous()
    ga, gb = graph_of(lambda: NP.nearest_other_all(C)), graph_of(lambda: NT.nearest_other_all(C))
    ta, tb = [], []
    for _ in range(60):
        for gr, t in ((ga, ta), (gb, tb)):
            torch.cuda.synchronize(); t0 = time.perf_counter(); gr.replay(); torch.cuda.synchronize()
            t.append(1e3 * (time.perf_counter() - t0))
    print(f"{ds:11s} k={k:6d} d={X.shape[1]:4d}: previous {min(ta):7.3f} ms | new {min(tb):7.3f} ms  "
          f"({min(tb) / min(ta):.2f}x)", flush=True)
