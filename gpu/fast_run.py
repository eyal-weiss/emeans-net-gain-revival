"""Runs for the optimized GPU net-gain revival: correctness against the proof of concept, and timing.

Configs (PR overlay at 98deaca on PYTHONPATH, plus this dir, the release root and the research dir):
  pr        the PR's Emeans (its own rule; CUDA-graph replay where eligible)
  pr_eager  the same, forced to train eagerly
  poc       the proof of concept (research/netgain_torch.NetGainEmeans, medg_pm, eager, host decision in NumPy)
  ff        NetGainEmeansFast, knn="faiss" (the CPU classes' search on the host), no early stop  [exact vs poc]
  fd        NetGainEmeansFast, device search, no early stop
  fh        NetGainEmeansFast, device search, early stop "hl"                                     [default]
  fnh       NetGainEmeansFast, device search, early stop "hl", no freeze
  (device-check variants: fboth, fdev, fdevh = ng; no freeze: fbothn, fdevn, fdevnh = ngn, fdn = host check)
With the PR head overlay (79440bc) on PYTHONPATH instead, the same configs run on the head's driver (no freeze in the
driver; closing handover), plus the head proof of concept (netgain_torch_poc_head): poch / pochn (freeze / no freeze,
early stop "hl"), poch0 / pochn0 (without early stop); any config + "_e" trains eagerly, + "_x" without the closing handover. Records carry the driver and the closing-handover route.
Records: one JSON line per run on stdout and in OUT (append); sha of the final centroids for exact comparisons.
usage: python fast_run.py OUT DEVICE CELLS SEEDS T CONFIGS     (CELLS = ds:k,...  SEEDS = 30,31  DEVICE = cpu|cuda)
"""
import hashlib, json, os, sys, time

import numpy as np
import torch

out, device, cells, seeds, T, configs = (sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]),
                                         sys.argv[6].split(","))
cells = [(c.split(":")[0], int(c.split(":")[1])) for c in cells.split(",")]
seeds = [int(s) for s in seeds.split(",")]
sys.argv = sys.argv[:1]
sys.path.insert(0, os.environ["EXPERIMENTS_DIR"])
import grid  # noqa: E402
grid._init(int(os.environ.get("OMP_NUM_THREADS", "4")))
import datasets  # noqa: E402

dev = torch.device(device)


def make(cfg, d, k, T, seed):
    if cfg.endswith("_e"):                   # the same, trained eagerly (no graph replay; nothing else changes)
        m = make(cfg[:-2], d, k, T, seed)
        m._capture_operations_match = lambda owner=None: False
        return m
    if cfg.endswith("_x"):                   # (head) without the closing handover
        m = make(cfg[:-2], d, k, T, seed)
        m.closing_handover = False
        return m
    if cfg.startswith("om"):      # o-means (run WITHOUT the PR overlay): om, om_poc, om_fd, om_fh, om_fnh
        if cfg == "om":
            from faiss.contrib.o_means.o_means import Omeans
            return Omeans(d, k, nepochs=T, seed=seed)
        if cfg == "om_poc":
            import omeans_netgain
            return omeans_netgain.OmeansNetGain(d, k, nepochs=T, seed=seed, stat="medg_pm")
        from omeans_fast import OmeansNetGainFast
        if cfg == "om_ng":
            cfg = "om_devh"
        opts = dict(om_fd=dict(early_stop=None, check="host"), om_fh=dict(early_stop="hl", check="host"),
                    om_fnh=dict(early_stop="hl", freeze=False, check="host"),
                    om_both=dict(early_stop=None, check="both"), om_dev=dict(early_stop=None, check="device"),
                    om_devh=dict(early_stop="hl", check="device"))[cfg]
        return OmeansNetGainFast(d, k, nepochs=T, seed=seed, stat="medg_pm", **opts)
    from faiss.contrib.e_means.e_means_torch import Emeans
    if cfg == "pr":
        return Emeans(d, k, nepochs=T, seed=seed)
    if cfg == "pr_eager":
        class EagerEmeans(Emeans):
            def _assign_rows(self, *a, **kw):
                return super()._assign_rows(*a, **kw)
        return EagerEmeans(d, k, nepochs=T, seed=seed)
    if cfg == "poc":
        import netgain_torch
        return netgain_torch.NetGainEmeans(d, k, nepochs=T, seed=seed, stat="medg_pm")
    if cfg.startswith("poch"):
        import netgain_torch_poc_head as PH
        return PH.NetGainEmeansPocHead(d, k, nepochs=T, seed=seed, stat="medg_pm", freeze=not cfg.startswith("pochn"),
                                       early_stop=None if cfg.endswith("0") else "hl")
    from netgain_torch_fast import NetGainEmeansFast
    if cfg == "ng":                          # the optimized rule as reported: device check, early stop, freeze
        cfg = "fdevh"
    if cfg == "ngn":                         # the same without the freeze
        cfg = "fdevnh"
    opts = dict(ff=dict(knn="faiss", early_stop=None, freeze=True, check="host"),
                fd=dict(early_stop=None, freeze=True, check="host"),
                ffn=dict(knn="faiss", early_stop=None, freeze=False, check="host"),
                fh=dict(early_stop="hl", freeze=True, check="host"), fnh=dict(early_stop="hl", freeze=False, check="host"),
                fboth=dict(early_stop=None, freeze=True, check="both"), fdev=dict(early_stop=None, freeze=True, check="device"),
                fdevh=dict(early_stop="hl", freeze=True, check="device"),
                fdn=dict(early_stop=None, freeze=False, check="host"),
                fbothn=dict(early_stop=None, freeze=False, check="both"),
                fdevn=dict(early_stop=None, freeze=False, check="device"),
                fdevnh=dict(early_stop="hl", freeze=False, check="device"))[cfg]
    return NetGainEmeansFast(d, k, nepochs=T, seed=seed, stat="medg_pm", **opts)


REPS = int(os.environ.get("NG_TIMING", "0"))          # > 0: timing only (no scoring), REPS fits per (cell, seed, cfg)
done = set()
if os.path.exists(out):
    for line in open(out):
        try:
            r = json.loads(line)
            done.add((r["config"], r["dataset"], r["k"], r["seed"], r.get("rep", 0)))
        except Exception:
            pass

for ds, k in cells:
    if all((cfg, ds, k, s, rp) in done for cfg in configs for s in seeds for rp in range(max(REPS, 1))):
        continue
    X, Xte, info = datasets.load(ds, 0)
    n, d = X.shape
    Xg = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).to(dev)
    for seed in seeds:
        rows = np.sort(np.random.RandomState(seed).permutation(n)[:k])
        init = torch.from_numpy(np.ascontiguousarray(X[rows], dtype=np.float32)).to(dev)
        for rep, cfg in [(rp, c) for rp in range(max(REPS, 1)) for c in configs]:
            if (cfg, ds, k, seed, rep) in done:
                continue
            m = make(cfg, d, k, T, seed)
            flag = {"v": None}
            if hasattr(m, "_replay_eligible"):
                orig = m._replay_eligible

                def probe(x, _orig=orig):
                    flag["v"] = bool(_orig(x))
                    return flag["v"]
                m._replay_eligible = probe
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            C, _ = m.train(Xg, init_centroids=init)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            secs = time.perf_counter() - t0
            Cn = np.ascontiguousarray(C.detach().float().cpu().numpy())
            rec = dict(config=cfg, dataset=ds, k=k, seed=seed, T=T, device=device, seconds=secs,
                       n_splits=int(getattr(m, "n_splits", -1)), n_checks=int(getattr(m, "n_checks", 0)),
                       rule_seconds=float(getattr(m, "rule_seconds", 0.0)), replay=flag["v"],
                       es_stop_step=getattr(m, "es_stop_step", None),
                       check_agree=getattr(m, "check_agree", None), check_total=getattr(m, "check_total", None),
                       captured=getattr(m, "check_captured", None), device_check=getattr(m, "device_check", None), capture_error=getattr(m, "capture_error", None),
                       sha=hashlib.sha256(Cn.tobytes()).hexdigest()[:16],
                       gpu=torch.cuda.get_device_name(0) if dev.type == "cuda" else None, host=os.uname().nodename)
            rec["rep"] = rep
            rec["driver"] = "head" if hasattr(type(m), "_handover_start") else ("98deaca" if not cfg.startswith("om")
                                                                                 else "omeans")
            rec["handover"] = getattr(m, "recipe", {}).get("closing_handover") if hasattr(m, "recipe") else None
            rec["handover_stats"] = getattr(m, "closing_handover_stats", None)
            if not REPS:
                rec.update(grid._score(Cn, X, Xte, info, ds, k))
            line = json.dumps(rec)
            print(line, flush=True)
            with open(out, "a") as fh:
                fh.write(line + "\n")
