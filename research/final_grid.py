"""Final comparison on the released implementation (2026-11-21/22): net_gain_revival.NetGainRevivalEmeans (version
1.1's class for the PR's head driver 79440bc, C++ kernels) with the early stop, with and without the freeze, against the
PR's own rule on the same driver. Same cells, starts, host placement and scoring as round 3's Part F (prhead_grid.py),
on the same seeds, so that every run pairs with a round-3 run:
  R  : 34 cells x seeds 200-201, T = 2, our rule WITHOUT the early stop (freeze and no freeze): must equal round 3's
       records of research/ngp_head.py (ngh79440bc_medg_pm_{freeze,nofreeze}_guard) in test and training error
  F2 : 34 cells x seeds 200-259, T = 2: pr, rel:freeze:hl, rel:nofreeze:hl
  F5 : 34 cells x seeds 200-239, T = 5: as F2
The PR's runs (pr_head79440bc) must also equal round 3's records at the same keys.
Records: OUT/runs/<key>.json (atomic, resumable; the runner refuses to resume over other code or data).
usage: FAISS_OVERLAY=<faiss_head_79440bc> PYTHONPATH=<package dir>:<round-3 code>:<experiments> \
       python -u final_grid.py --host H --out DIR --parts R,F2,F5 [--workers 8 --threads 4]
"""
import os
import sys

OVERLAY = os.environ.get("FAISS_OVERLAY", "")
if OVERLAY:
    sys.path.insert(0, OVERLAY)
import argparse, json, time, traceback  # noqa: E402,E401

import confirm2_grid as C2  # noqa: E402  (cells, placement, sizes, data hashes, atomic writes; imports grid)
import grid  # noqa: E402

R = lambda lo, hi: list(range(lo, hi + 1))
SPECS = {"R": ["rel:freeze:none", "rel:nofreeze:none"],
         "F2": ["pr", "rel:freeze:hl", "rel:nofreeze:hl"], "F5": ["pr", "rel:freeze:hl", "rel:nofreeze:hl"]}
PARTS = {"R": (C2.CELLS34, R(200, 201), 2), "F2": (C2.CELLS34, R(200, 259), 2), "F5": (C2.CELLS34, R(200, 239), 5)}


def version():
    import faiss
    d = os.path.join(os.path.dirname(faiss.__file__), "contrib", "e_means")
    h = {f: C2._sha(os.path.join(d, f)) for f in ("base.py", "e_means.py")}
    if h != {"base.py": "e33dc8ad201f9d7d", "e_means.py": "89b9f3f537e613d9"}:
        raise SystemExit(f"needs the driver at 79440bc, found {h}")
    return h


def identity():
    """Round 2's full identity (grid.py, datasets.py, the imported FAISS build, the Python/NumPy/FAISS versions), this
    runner, and the released package with its compiled kernels."""
    import net_gain_revival
    pkg = os.path.dirname(os.path.abspath(net_gain_revival.__file__))
    out = dict(C2.code_identity("py"))
    out["final_grid.py"] = C2._sha(os.path.abspath(__file__))
    for f in ("__init__.py", "emeans.py", "rule.py", "native/__init__.py", "native/ngp_head.cpp",
              "native/ngp_decide.cpp", "native/ngp_kernels.cpp", "native/libnetgain.so"):
        out["net_gain_revival/" + f] = C2._sha(os.path.join(pkg, f))
    return out


def config_name(spec):
    if spec == "pr":
        return "pr_head79440bc"
    _, fz, es = spec.split(":")
    assert fz in ("freeze", "nofreeze") and es in ("hl", "none")
    return f"rel79440bc_{fz}_{es}"


def run_key(cfg, ds, k, seed, T):
    return f"fin|{cfg}|{ds}|k{k}|s{seed}|random|T{T}"


def run_job(args):
    out, ds, k, seed, T, spec = args
    import numpy as np
    import datasets
    import faiss
    from faiss.contrib.e_means.e_means import NumpyEmeans
    vh = version()
    cfg = config_name(spec)
    key = run_key(cfg, ds, k, seed, T)
    try:
        X, Xte, info = datasets.load(ds, seed)
        n, d = X.shape
        rows = np.sort(np.random.RandomState(seed).permutation(n)[:k])          # the round-2 random start
        init = np.ascontiguousarray(X[rows], dtype=np.float32)
        if spec == "pr":
            m = NumpyEmeans(d, k, nepochs=T, seed=seed)
        else:
            from net_gain_revival import NetGainRevivalEmeans
            _, fz, es = spec.split(":")
            m = NetGainRevivalEmeans(d, k, nepochs=T, seed=seed, freeze=fz == "freeze",
                                     early_stop=None if es == "none" else es, use_native=True)
        t0 = time.perf_counter()
        C, _ = m.train(X, init_centroids=init)
        secs = time.perf_counter() - t0
        rec = dict(key=key, config=cfg, dataset=ds, k=k, seed=seed, init="random", T=T, seconds=secs,
                   faiss_file=faiss.__file__, e_means=vh, runner=C2._sha(os.path.abspath(__file__)),
                   handover=getattr(m, "closing_handover_stats", None), data_sha=str(info["train_sha256"])[:16],
                   data_test_sha=str(info["test_sha256"])[:16], code=identity(),
                   n_splits=int(getattr(m, "n_splits", -1)), n_checks=int(getattr(m, "n_checks", 0)),
                   n_candidates=int(getattr(m, "n_candidates", 0)),
                   n_moves_last_pass=int(getattr(m, "n_moves_last_pass", 0)),
                   es_stop_step=getattr(m, "es_stop_step", None), total_steps=int(m.recipe["total_steps"]),
                   centroids_sha=__import__("hashlib").sha256(np.ascontiguousarray(C).tobytes()).hexdigest()[:16])
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
    p.add_argument("--parts", default="R,F2,F5")
    p.add_argument("--workers", type=int, default=8); p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    vh = version()
    print(f"e_means {vh} from {OVERLAY or 'the installed package'}", flush=True)
    os.makedirs(os.path.join(a.out, "runs"), exist_ok=True)
    done = C2.finished_keys(a.out)
    ident = identity()
    for key in sorted(done):
        r = json.load(open(C2._run_file(a.out, key)))
        if r.get("runner") != C2._sha(os.path.abspath(__file__)) or r.get("code") != ident:
            raise SystemExit(f"refusing to resume: {key} was made with another runner or other code")
        if (r.get("data_sha"), r.get("data_test_sha")) != C2.data_identity(r["dataset"]):   # hashes the files
            raise SystemExit(f"refusing to resume: {key} was made with other data")
    for part in a.parts.split(","):                  # parts in order: each starts when the previous one is done
        cells, seeds, T = PARTS[part]
        jobs = []
        for s in seeds:
            for ds, k in cells:
                for spec in SPECS[part]:
                    if C2.HOST[ds] != a.host or run_key(config_name(spec), ds, k, s, T) in done:
                        continue
                    jobs.append(((s, -C2.SIZE[ds][0] * max(C2.SIZE[ds][1], 16) * k * T, spec),
                                 (a.out, ds, k, s, T, spec)))
        jobs = [j for _, j in sorted(jobs)]
        for ds in sorted({j[1] for j in jobs}):      # the data files must match their recorded hashes
            C2.data_identity(ds)
        print(f"{time.strftime('%H:%M:%S')} part {part}: {len(jobs)} runs to do on {a.host}", flush=True)
        n_ok = n_err = 0
        with ProcessPoolExecutor(a.workers, mp_context=get_context("spawn"), initializer=_init,
                                 initargs=(a.threads, OVERLAY)) as ex:
            for f in as_completed([ex.submit(run_job, j) for j in jobs]):
                recs, errs = f.result()
                if errs:
                    with open(os.path.join(a.out, "errors.jsonl"), "a") as fh:
                        fh.write("".join(json.dumps(e) + "\n" for e in errs))
                n_ok += len(recs); n_err += len(errs)
                if (n_ok + n_err) % 50 == 0 or n_ok + n_err == len(jobs):
                    print(f"{time.strftime('%H:%M:%S')} part {part}: ok {n_ok} errors {n_err} of {len(jobs)}", flush=True)
        if n_err:
            sys.exit(1)
    sys.exit(0)


def _init(threads, overlay):
    if overlay and overlay not in sys.path:      # spawned workers: same faiss as the parent
        sys.path.insert(0, overlay)
    grid._init(threads)


if __name__ == "__main__":
    main()
