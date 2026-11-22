"""EXPLORATORY check of the latest FAISS PR #5915 head (2026-11-19; not pre-registered; descriptive only).

The PR moved on after the commit all our runs use (98deaca): 8 commits on 18-19 Nov, Python only (no C++ change),
including "Drop the revival guard" and "twin reassignment, closing mean for every centroid, no freeze". This runner
trains the PR's own NumpyEmeans with its defaults (cost-based revival; at the head also the closing handover), from
the round-2 starts, so the results can be put next to the round-2 records of `pr` (our subclass at 98deaca) and
`medg_pm` at the same keys:
  PH  : 34 cells, seeds 30-49, random starts, T = 2 (round-2 part P keys)
  PH5 : 34 cells, seeds 30-39, random starts, T = 5 (round-2 part S5 keys)
  PV  : the plain class at 98deaca (no overlay) on 34 cells, seeds 30-31, T = 2: must equal round-2 `pr` exactly
        (checks that this runner reproduces the `pr` config)
--configs: "pr" (the PR's own class, default) and/or our rule on the head's driver (ngp_head.NetGainHead), named
  ngh:<stat>:<freeze|nofreeze>:<guard|noguard>, e.g. ngh:medg_pm:nofreeze:guard (head overlay only).
  ngh:medg_pm:freeze:guard is our round-2 rule on the new driver: expected to reproduce round-2 medg_pm records.
With FAISS_OVERLAY=<dir> the faiss package is imported from <dir> (a copy of the installed PR build with the head's
contrib/e_means files); the config name records which version ran (pr_head79440bc or pr_plain98deaca).
Records: OUT/runs/<key>.json (atomic, resumable; the runner refuses other code). Same host placement as round 2.
usage: [FAISS_OVERLAY=dir] python -u prhead_grid.py --host H --out DIR --parts PH,PH5 [--workers 8 --threads 4]
"""
import os, sys

OVERLAY = os.environ.get("FAISS_OVERLAY", "")
if OVERLAY:
    sys.path.insert(0, OVERLAY)
import argparse, hashlib, json, random, time, traceback  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import confirm2_grid as C2  # noqa: E402  (cells, placement, sizes, atomic writes; imports grid)
import grid  # noqa: E402

R = lambda lo, hi: list(range(lo, hi + 1))
PARTS = {"PH": (C2.CELLS34, R(30, 49), 2), "PH5": (C2.CELLS34, R(30, 39), 5), "PV": (C2.CELLS34, R(30, 31), 2)}


def version():
    import faiss
    d = os.path.join(os.path.dirname(faiss.__file__), "contrib", "e_means")
    h = {f: C2._sha(os.path.join(d, f)) for f in ("base.py", "e_means.py")}
    if h == {"base.py": "60a4f4472462404b", "e_means.py": "9649c999d94d95c7"}:
        return "pr_plain98deaca", h
    if h == {"base.py": "e33dc8ad201f9d7d", "e_means.py": "89b9f3f537e613d9"}:
        return "pr_head79440bc", h
    raise SystemExit(f"unknown e_means version {h}")


def identity():
    """Round 2's full identity (grid.py, datasets.py, the imported FAISS build, i.e. the overlay when set, and the
    Python/NumPy/FAISS versions) plus this runner, the round-3 wrapper, the head class and the rule."""
    out = dict(C2.code_identity("py"))
    for f in ("prhead_grid.py", "round3_prhead.py", "ngp_head.py", "netgain_rule.py"):
        if os.path.exists(os.path.join(HERE, f)):
            out[f] = C2._sha(os.path.join(HERE, f))
    return out


def run_key(cfg, ds, k, seed, T):
    return f"prh|{cfg}|{ds}|k{k}|s{seed}|random|T{T}"


def config_name(spec, version_name):
    if spec == "pr":
        return version_name
    _, stat, fz, gd = spec.split(":")
    assert version_name == "pr_head79440bc" and fz in ("freeze", "nofreeze") and gd in ("guard", "noguard")
    return f"ngh79440bc_{stat}_{fz}_{gd}"


def run_job(args):
    out, ds, k, seed, T, spec = args
    import numpy as np
    import datasets
    import faiss
    from faiss.contrib.e_means.e_means import NumpyEmeans
    vname, vh = version()
    cfg = config_name(spec, vname)
    key = run_key(cfg, ds, k, seed, T)
    try:
        X, Xte, info = datasets.load(ds, seed)
        n, d = X.shape
        rows = np.sort(np.random.RandomState(seed).permutation(n)[:k])          # the round-2 random start
        init = np.ascontiguousarray(X[rows], dtype=np.float32)
        if spec == "pr":
            m = NumpyEmeans(d, k, nepochs=T, seed=seed)
        else:
            import ngp_head
            _, stat, fz, gd = spec.split(":")
            m = ngp_head.NetGainHead(d, k, nepochs=T, seed=seed, stat=stat, freeze=fz == "freeze", guard=gd == "guard")
        t0 = time.perf_counter()
        C, _ = m.train(X, init_centroids=init)
        secs = time.perf_counter() - t0
        rec = dict(key=key, config=cfg, dataset=ds, k=k, seed=seed, init="random", T=T, seconds=secs,
                   faiss_file=faiss.__file__, e_means=vh, runner=C2._sha(os.path.abspath(__file__)),
                   handover=getattr(m, "closing_handover_stats", None), data_sha=str(info["train_sha256"])[:16],
                   data_test_sha=str(info["test_sha256"])[:16], code=identity(),
                   n_splits=int(getattr(m, "n_splits", -1)), n_checks=int(getattr(m, "n_checks", 0)),
                   n_moves_last_pass=int(getattr(m, "n_moves_last_pass", 0)),
                   ngp_head=C2._sha(os.path.join(HERE, "ngp_head.py")), rule=C2._sha(os.path.join(HERE, "netgain_rule.py")))
        rec.update(grid._score(C, X, Xte, info, ds, k))
        rec.update(n=int(n), d=int(d), threads=grid._THREADS, host=os.uname().nodename,
                   finished=time.strftime("%Y-%m-%d %H:%M:%S %Z"))
        C2._write_run(out, rec)
        return [rec], []
    except Exception:
        return [], [dict(key=key, error=traceback.format_exc())]


def main():
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from multiprocessing import get_context
    p = argparse.ArgumentParser()
    p.add_argument("--host", required=True); p.add_argument("--out", required=True)
    p.add_argument("--parts", default="PH,PH5")
    p.add_argument("--workers", type=int, default=8); p.add_argument("--threads", type=int, default=4)
    p.add_argument("--max-jobs", type=int, default=0)
    p.add_argument("--configs", default="pr")
    a = p.parse_args()
    vname, vh = version()
    specs = a.configs.split(",")
    print(f"e_means version: {vname} {vh} from {OVERLAY or 'the installed package'}; configs {specs}", flush=True)
    done = C2.finished_keys(a.out)
    for key in sorted(done):
        r = json.load(open(C2._run_file(a.out, key)))
        if r.get("runner") != C2._sha(os.path.abspath(__file__)) or r.get("code") != identity():
            raise SystemExit(f"refusing to resume: {key} was made with another runner or other code")
        if (r.get("data_sha"), r.get("data_test_sha")) != C2.data_identity(r["dataset"]):   # hashes the files
            raise SystemExit(f"refusing to resume: {key} was made with other data")
    jobs = []
    for rank, part in enumerate(a.parts.split(",")):
        cells, seeds, T = PARTS[part]
        for s in seeds:
            for ds, k in cells:
                for spec in specs:
                    if C2.HOST[ds] != a.host or run_key(config_name(spec, vname), ds, k, s, T) in done:
                        continue
                    jobs.append(((rank, s, -C2.SIZE[ds][0] * max(C2.SIZE[ds][1], 16) * k * T, spec),
                                 (a.out, ds, k, s, T, spec)))
    jobs = [j for _, j in sorted(jobs)]
    if a.max_jobs:
        jobs = jobs[:a.max_jobs]
    for ds in sorted({j[1] for j in jobs}):          # the data files must match their recorded hashes (refuses otherwise)
        C2.data_identity(ds)
    print(f"{time.strftime('%H:%M:%S')} {len(jobs)} runs to do on {a.host} ({len(done)} finished)", flush=True)
    n_ok = n_err = 0
    with ProcessPoolExecutor(a.workers, mp_context=get_context("spawn"), initializer=_init,
                             initargs=(a.threads, OVERLAY)) as ex:
        for f in as_completed([ex.submit(run_job, j) for j in jobs]):
            recs, errs = f.result()
            if errs:
                with open(os.path.join(a.out, "errors.jsonl"), "a") as fh:
                    fh.write("".join(json.dumps(e) + "\n" for e in errs))
            n_ok += len(recs); n_err += len(errs)
            print(f"{time.strftime('%H:%M:%S')} runs ok {n_ok} errors {n_err}", flush=True)
    sys.exit(1 if n_err else 0)


def _init(threads, overlay):
    if overlay and overlay not in sys.path:      # spawned workers: same faiss as the parent
        sys.path.insert(0, overlay)
    grid._init(threads)


if __name__ == "__main__":
    main()
