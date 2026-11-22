"""Checks for the review's findings: adversarial nearest search (centring), and k = 1 in the e-means class."""
import os, sys
import numpy as np, torch
import netgain_triton as NT
dev = torch.device("cuda")
# review example: 1-D centroids far from the origin, FP32 score collisions
C = torch.tensor([[2.0**20 + v] for v in (0, 32, 64, 96, 128, 1)], device=dev)
nn, delta = NT.nearest_other_all(C)
nn_e = [int(np.argmin([abs(float(C[i] - C[j])) if j != i else np.inf for j in range(len(C))])) for i in range(len(C))]
print("adversarial 1-D: nn", nn.tolist(), "exact", nn_e, "delta[0]", float(delta[0]), "->", "OK" if nn.tolist() == nn_e else "MISMATCH")
# random clouds far from the origin, compared with an exact FP64 search
g = torch.Generator(device="cpu").manual_seed(0)
for off, k, d in ((1e4, 2048, 2), (1e3, 4096, 8), (1e5, 1024, 3)):
    X = (torch.randn(k, d, generator=g) * 0.01 + off).to(dev)
    nn, _ = NT.nearest_other_all(X)
    Xd = X.double(); D2 = torch.cdist(Xd, Xd).square(); D2.fill_diagonal_(float("inf"))
    best = D2.min(1).values
    got = D2[torch.arange(k, device=dev), nn]
    print(f"offset {off:g} k={k} d={d}: nearest found exactly for {int((got == best).sum())} of {k}")
if len(sys.argv) > 1:
    sys.path.insert(0, os.environ["EXPERIMENTS_DIR"])
    import datasets
    from netgain_torch_fast import NetGainEmeansFast
    X, _, _ = datasets.load("shuttle_z", 0)
    Xg = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).to(dev)
    m = NetGainEmeansFast(X.shape[1], 1, nepochs=2, seed=1)
    Cf, _ = m.train(Xg, init_centroids=Xg[:1].clone())
    print("k = 1 run:", "OK" if torch.isfinite(Cf).all() else "BAD", "splits", m.n_splits)
# round-2 corner cases
for name, vals in (("underflow", [2.0**-80 * v for v in (0, 5, 6, 7, 8, 1)]),
                   ("overflow k=4", [-2.0**65, 0.0, 2.0**65, 2.0**65 + 2.0**42]),
                   ("singleton", [7.0])):
    X = torch.tensor([[v] for v in vals], device=dev)
    nn, delta = NT.nearest_other_all(X)
    Xd = X.double(); D2 = (Xd - Xd.T).square(); D2.fill_diagonal_(float("inf"))
    ok = len(vals) == 1 and bool(torch.isinf(delta).all()) or (len(vals) > 1 and bool((delta == D2.min(1).values).all()))
    print(f"{name}: nn {nn.tolist()} -> {'OK' if ok else 'WRONG'}")
