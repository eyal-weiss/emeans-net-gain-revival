"""GPU proof of concept: the PR's GPU class (cost-based revival, its default) against the same class with the net-gain
rule (netgain_torch.NetGainEmeans), on one GPU. 2026-11-19 evening; exploratory, not optimized.

Configs: pr (the PR's Emeans as is: CUDA-graph replay where eligible), pr_eager (the same rule forced to train eagerly
like ours, for a fair time comparison), medg_pm, medg_fh (NetGainEmeans); run with the PR overlay on PYTHONPATH.
o-means (run WITHOUT the overlay, o-means as published): om (Omeans, its own count rule), om_medg_pm, om_medg_fh
(omeans_netgain.OmeansNetGain).
Starts: the round-2 random starts (sorted RandomState(seed).permutation(n)[:k]) or, for k-means++, none here.
Scores: grid._score on the host (test and training MSE, empty centroids), as every CPU record.
Records: OUT/runs/<key>.json (atomic; resumable).
usage: PYTHONPATH=<faiss_pr98>:<this dir> python gpu_netgain_run.py OUT CELLS SEEDS T CONFIGS
  CELLS = ds:k,ds:k,...   SEEDS = 30,31   CONFIGS = pr,pr_eager,medg_pm,medg_fh
"""
import json, os, sys, time

import numpy as np
import torch

out, cells, seeds, T, configs = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5].split(",")
cells = [(c.split(":")[0], int(c.split(":")[1])) for c in cells.split(",")]
seeds = [int(s) for s in seeds.split(",")]
sys.argv = sys.argv[:1]
sys.path.insert(0, os.environ["EXPERIMENTS_DIR"])
import grid  # noqa: E402
grid._init(int(os.environ.get("OMP_NUM_THREADS", "4")))
import datasets  # noqa: E402


def make(cfg, d, k, T, seed):
    if cfg.startswith("om"):
        if cfg == "om":
            from faiss.contrib.o_means.o_means import Omeans
            return Omeans(d, k, nepochs=T, seed=seed)
        import omeans_netgain as ON
        return ON.OmeansNetGain(d, k, nepochs=T, seed=seed, stat=cfg[3:])
    from faiss.contrib.e_means.e_means_torch import Emeans
    if cfg == "pr":
        return Emeans(d, k, nepochs=T, seed=seed)
    if cfg == "pr_eager":
        class EagerEmeans(Emeans):
            """The PR's class with its own rule, made to train eagerly (an override disables CUDA-graph replay)."""
            def _assign_rows(self, *a, **kw):
                return super()._assign_rows(*a, **kw)
        return EagerEmeans(d, k, nepochs=T, seed=seed)
    import netgain_torch as NT
    return NT.NetGainEmeans(d, k, nepochs=T, seed=seed, stat=cfg)


def write(rec):
    path = os.path.join(out, "runs", rec["key"].replace("|", "__") + ".json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "w") as fh:
        fh.write(json.dumps(rec))
    os.replace(tmp, path)


dev = torch.device("cuda")
for ds, k in cells:
    X, Xte, info = datasets.load(ds, 0)
    n, d = X.shape
    Xg = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).to(dev)
    for seed in seeds:
        rows = np.sort(np.random.RandomState(seed).permutation(n)[:k])
        init = torch.from_numpy(np.ascontiguousarray(X[rows], dtype=np.float32)).to(dev)
        for cfg in configs:
            key = f"gpu|{cfg}|{ds}|k{k}|s{seed}|random|T{T}"
            if os.path.exists(os.path.join(out, "runs", key.replace("|", "__") + ".json")):
                continue
            m = make(cfg, d, k, T, seed)
            replay = None
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            if hasattr(m, "_replay_eligible") and not cfg.startswith("om"):
                orig = m._replay_eligible                    # record whether the PR's class chose graph replay

                def probe(x, _orig=orig):
                    nonlocal_flag["v"] = bool(_orig(x))
                    return nonlocal_flag["v"]
                nonlocal_flag = {"v": None}
                m._replay_eligible = probe
            C, _ = m.train(Xg, init_centroids=init)
            torch.cuda.synchronize()
            if hasattr(m, "_replay_eligible") and not cfg.startswith("om"):
                replay = nonlocal_flag["v"]
            secs = time.perf_counter() - t0
            Cn = C.detach().float().cpu().numpy() if torch.is_tensor(C) else np.asarray(C, dtype=np.float32)
            rs = getattr(m, "revival_stats", None) or {}
            rec = dict(key=key, config=cfg, dataset=ds, k=k, seed=seed, init="random", T=T, seconds=secs,
                       n_splits=int(rs.get("splits", getattr(m, "n_splits", -1))),
                       n_checks=int(rs.get("checks", getattr(m, "n_checks", 0))),
                       rule_seconds=float(getattr(m, "rule_seconds", 0.0)),
                       backend=str((m.recipe or {}).get("assignment_backend")) if isinstance(m.recipe, dict) else None,
                       replay=replay,
                       gpu=torch.cuda.get_device_name(0), host=os.uname().nodename,
                       finished=time.strftime("%Y-%m-%d %H:%M:%S %Z"))
            rec.update(grid._score(Cn, X, Xte, info, ds, k))
            write(rec)
            print(json.dumps({x: rec[x] for x in ("key", "test_mse", "train_mse", "empty", "n_splits", "seconds",
                                                  "rule_seconds", "replay")}), flush=True)
            del m
            torch.cuda.empty_cache()
