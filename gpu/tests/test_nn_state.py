"""nearest_other_state (fused glue) against nearest_other_all + the torch conversions: nn, FP32-rounded delta, dup."""
import os, sys
import numpy as np, torch
import netgain_triton as NT
dev = torch.device("cuda")
g = torch.Generator(device="cpu").manual_seed(0)


def ref(C):
    nn, delta = NT.nearest_other_all(C)
    return nn, delta.float().double(), (C == C[nn]).all(dim=1).to(torch.int8)


def check(name, C):
    C = C.float().contiguous().to(dev)
    a, b = ref(C), NT.nearest_other_state(C)
    dn = int((a[0] != b[0]).sum()); dd = int(((a[1] != b[1]) & ~(torch.isnan(a[1]) & torch.isnan(b[1]))).sum())
    du = int((a[2] != b[2]).sum())
    # an nn difference is acceptable only at an exact distance tie
    Cd = C.double()
    tie_ok = bool(((Cd[a[0]] - Cd).square().sum(1)[a[0] != b[0]] == (Cd[b[0]] - Cd).square().sum(1)[a[0] != b[0]]).all())
    print(f"{name:30s} k={len(C):6d} d={C.shape[1]:4d}: nn differ {dn} (ties: {tie_ok}), delta differ {dd}, dup differ {du}")
    return dd == 0 and du == 0 and (dn == 0 or tie_ok)


ok = True
for d in (2, 3, 8, 32, 96, 784):
    for k in (6, 7, 64, 1000, 4096):
        ok &= check(f"gaussian", torch.randn(k, d, generator=g))
        X = torch.randn(max(1, k // 4), d, generator=g).repeat(4, 1)[:k]
        if len(X) == k:
            ok &= check(f"4x duplicates", X)
        ok &= check(f"offset 1e3 spread 1e-2", 1e3 + 1e-2 * torch.randn(k, d, generator=g))
for name, vals in (("underflow", [2.0**-80 * v for v in (0, 5, 6, 7, 8, 1, 9)]),
                   ("1-D collisions", [2.0**20 + v for v in (0, 32, 64, 96, 128, 1)])):
    ok &= check(name, torch.tensor([[v] for v in vals]))
X = torch.randn(4096, 96, generator=g); X[::7] = float("inf")
ok &= check("with inf rows", X)
if len(sys.argv) > 1:
    sys.path.insert(0, os.environ["EXPERIMENTS_DIR"])
    import grid; grid._init(8)
    import datasets
    from faiss.contrib.e_means.e_means_torch import Emeans
    for ds, k in (("deep1m", 16384), ("kdd_z", 4096), ("geonames", 4096), ("road3d_z", 4096), ("mnist_255", 1024),
                  ("shuttle_z", 256), ("codrna_z", 4096), ("census_z", 1024)):
        X, _, _ = datasets.load(ds, 0)
        Xg = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).to(dev)
        rows = np.sort(np.random.RandomState(30).permutation(len(X))[:k])
        C, _ = Emeans(X.shape[1], k, nepochs=2, seed=30).train(Xg, init_centroids=Xg[torch.from_numpy(rows).to(dev)])
        ok &= check(f"{ds} trained", C)
print("ALL OK" if ok else "DIFFERENCES")
