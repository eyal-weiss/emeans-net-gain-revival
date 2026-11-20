"""EXPLORATORY extended grid (2026-11-19, after the confirmatory study; not pre-registered).

  N2/N5/NK: the additional small-to-medium datasets (datasets_extra.py) with every main config and the four
            exploratory variants (ngp_variants.py): random starts T = 2 (seeds 0-19), T = 5 (seeds 0-4), k-means++ starts (seeds 0-9).
            Seeds 30-49 stay unused (reserved for a later confirmatory round).
  O2/O5/OK: the 16 development cells with the four variants only, at the confirmatory study's keys (C1 seeds
            10-29, C6 T = 5 seeds 10-14, C2 k-means++ seeds 0-9); the other configs are the confirmatory records.
No sampling strides (dropped 2026-11-19: with the statistics in C++ sampling saves almost no time).
Records: RUNDIR/runs/<key>.json (atomic), as confirm_grid.py. Run with NGP_IMPL=cpp.
usage: python -u ext_grid.py --host H --out RUNDIR --parts N2,N5,NK,O2,O5,OK [--workers 8 --threads 4] --kmpp-dir DIR
"""
import argparse, json, os, random, sys, time, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import confirm_grid as CG  # noqa: E402  (helpers: record writing, loading, cells)
import grid  # noqa: E402

MAIN = ["lib", "pr", "ours", "h21", "ngp2_full"]
VAR = ["ngp2_full_medg", "ngp2_full_med", "ngp2_full_pm", "ngp2_full_medg_pm"]   # ngp_variants.py
NEW_CELLS = [("mnist_mm", 256), ("mnist_mm", 1024), ("fashion_mm", 256), ("fashion_mm", 1024),
             ("corel_z", 256), ("corel_z", 1024), ("road3d_z", 256), ("road3d_z", 1024), ("road3d_z", 4096),
             ("codrna_z", 256), ("codrna_z", 1024), ("codrna_z", 4096), ("ijcnn1_z", 256), ("ijcnn1_z", 1024),
             ("sensit_z", 256), ("sensit_z", 1024), ("higgs_z", 256), ("higgs_z", 1024)]
R = lambda lo, hi: list(range(lo, hi + 1))
PARTS = {
    "N2": [(NEW_CELLS, R(0, 19), "random", 2, MAIN + VAR)],
    "N5": [(NEW_CELLS, R(0, 4), "random", 5, MAIN + VAR)],
    "NK": [(NEW_CELLS, R(0, 9), "kmpp", 2, MAIN + VAR)],
    "O2": [(CG.CELLS16, R(10, 29), "random", 2, VAR)],
    "O5": [(CG.CELLS16, R(10, 14), "random", 5, VAR)],
    "OK": [(CG.CELLS16, R(0, 9), "kmpp", 2, VAR)],
}
NEW_DS = sorted({ds for ds, _ in NEW_CELLS})
HOST = dict(CG.HOST, **{ds: "host_b" for ds in NEW_DS})
SIZE = dict(grid.SIZE, mnist_mm=(5e4, 784), fashion_mm=(5e4, 784), corel_z=(4.8e4, 16), road3d_z=(4.1e5, 3),
            codrna_z=(3.2e5, 8), ijcnn1_z=(1.2e5, 22), sensit_z=(7.9e4, 100), higgs_z=(7.8e4, 28))


def run_key(cfg, ds, k, seed, start, T):
    return f"ext|{cfg}|{ds}|k{k}|s{seed}|{start}|T{T}"


def code_identity():
    files = ["ablation.py", "ablation_cpp.py", "ngp_kernels.cpp", "ngp_kernels.so", "ngp_variants.py", "ext_grid.py"]
    return {f: CG._sha(os.path.join(HERE, f)) for f in files}


def plan(parts, host):
    seen, jobs = set(), []
    for rank, p in enumerate(parts):
        for cells, seeds, start, T, cfgs in PARTS[p]:
            for ds, k in cells:
                if HOST[ds] != host:
                    continue
                for s in seeds:
                    todo = [c for c in cfgs if (c, ds, k, s, start, T) not in seen]
                    seen.update((c, ds, k, s, start, T) for c in todo)
                    if todo:
                        jobs.append((rank, p, ds, k, s, start, T, todo))
    return jobs


def run_job(args):
    out, ds, k, seed, start, T, configs, part, kdir = args
    import numpy as np
    import datasets
    import ngp_variants as V
    recs, errs = [], []
    try:
        X, Xte, info = datasets.load(ds, seed)
        n, d = X.shape
        if start == "random":
            rows = np.sort(np.random.RandomState(seed).permutation(n)[:k])
            init = np.ascontiguousarray(X[rows], dtype=np.float32)
        else:
            init = np.ascontiguousarray(np.load(CG.kmpp_path(kdir, ds, k, seed)), dtype=np.float32)
            assert init.shape == (k, d), (init.shape, k, d)
    except Exception:
        return [], [dict(dataset=ds, k=k, seed=seed, start=start, T=T, error=traceback.format_exc())]
    order = list(configs)
    random.Random(f"{ds}|{k}|{seed}|{start}|T{T}").shuffle(order)
    code = code_identity()
    for cfg in order:
        key = run_key(cfg, ds, k, seed, start, T)
        try:
            spec = V.CONFIGS[cfg]
            removal, donor, split, timing = spec[:4]
            extra = {} if len(spec) <= 4 else (spec[4] if isinstance(spec[4], dict) else {"w": spec[4]})
            V.TIMES.clear()
            load0 = os.getloadavg()[0]
            m = V.AblationEmeans(d, k, removal, donor, split, timing, nepochs=T, seed=seed, **extra)
            t0 = time.perf_counter()
            C, _ = m.train(X, init_centroids=init)
            secs = time.perf_counter() - t0
            rec = dict(key=key, config=cfg, dataset=ds, k=k, seed=seed, init=start, T=T, part=part,
                       impl=os.environ.get("NGP_IMPL", "py"), seconds=secs,
                       times={kk: float(v) for kk, v in V.TIMES.items()}, n_splits=int(m.n_splits),
                       batch=int(m.recipe["batch_size"]), steps_per_pass=int(m.recipe["steps_per_pass"]),
                       alpha0=float(m.recipe["alpha0"]), load_before=load0, load_after=os.getloadavg()[0], code=code)
            rec.update(grid._score(C, X, Xte, info, ds, k))
            rec.update(n=int(n), d=int(d), threads=grid._THREADS, host=os.uname().nodename,
                       finished=time.strftime("%Y-%m-%d %H:%M:%S"))
            CG._write_run(out, rec)
            recs.append(rec)
        except Exception:
            errs.append(dict(key=key, error=traceback.format_exc()))
    return recs, errs


def kmpp_inits(parts, host, kdir, workers):
    """k-means++ starts (grid.py's: sklearn kmeans_plusplus, random_state = seed) for the kmpp jobs that lack one."""
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    need = sorted({(ds, k, s) for _, _, ds, k, s, st, _, _ in plan(parts, host) if st == "kmpp"
                   and not os.path.exists(CG.kmpp_path(kdir, ds, k, s))})
    if not need:
        return
    print(f"{time.strftime('%H:%M:%S')} computing {len(need)} k-means++ starts", flush=True)
    out = os.path.dirname(kdir.rstrip("/"))        # grid.kmpp_job writes OUT/kmpp/<ds>_k<k>_s<seed>.npy
    assert os.path.basename(kdir.rstrip("/")) == "kmpp", kdir
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn"), initializer=grid._init, initargs=(4,)) as ex:
        for path, secs in ex.map(grid.kmpp_job, [(out, ds, k, s) for ds, k, s in need]):
            print(f"  {os.path.basename(path)} {secs:.1f} s", flush=True)


def main():
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from multiprocessing import get_context
    p = argparse.ArgumentParser()
    p.add_argument("--host", required=True); p.add_argument("--out", required=True)
    p.add_argument("--parts", default="N2,N5,NK,O2,O5,OK")
    p.add_argument("--workers", type=int, default=8); p.add_argument("--threads", type=int, default=4)
    p.add_argument("--kmpp-dir", required=True)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    parts = a.parts.split(",")
    assert all(x in PARTS for x in parts), parts
    J = plan(parts, a.host)
    os.makedirs(a.out, exist_ok=True)
    finished = CG.load_runs(a.out)
    current = code_identity()
    stale = sorted(k for k, r in finished.items() if r.get("code") != current)
    if stale:
        print(f"refusing to resume: {len(stale)} finished runs were made with other code, e.g. {stale[0]}", flush=True)
        sys.exit(2)
    jobs = []
    for rank, part, ds, k, s, st, T, cfgs in J:
        todo = [c for c in cfgs if run_key(c, ds, k, s, st, T) not in finished]
        if todo:
            jobs.append((rank, (a.out, ds, k, s, st, T, todo, part, a.kmpp_dir)))
    cost = lambda j: SIZE[j[1]][0] * max(SIZE[j[1]][1], 16) * j[2] * j[5] * len(j[6])
    jobs.sort(key=lambda rj: (rj[0], -cost(rj[1])))
    jobs = [j for _, j in jobs]
    print(f"{time.strftime('%H:%M:%S')} {len(jobs)} jobs ({sum(len(j[6]) for j in jobs)} runs) on {a.host}, "
          f"parts {parts}, workers {a.workers} x threads {a.threads}", flush=True)
    if a.dry_run:
        return
    kmpp_inits(parts, a.host, a.kmpp_dir, a.workers)
    n_ok = n_err = 0
    with ProcessPoolExecutor(a.workers, mp_context=get_context("spawn"), initializer=grid._init,
                             initargs=(a.threads,)) as ex:
        for f in as_completed([ex.submit(run_job, j) for j in jobs]):
            recs, errs = f.result()
            if errs:
                with open(os.path.join(a.out, "errors.jsonl"), "a") as fh:
                    for e in errs:
                        fh.write(json.dumps(e) + "\n")
            n_ok += len(recs); n_err += len(errs)
            print(f"{time.strftime('%H:%M:%S')} runs ok {n_ok} errors {n_err}", flush=True)
    open(os.path.join(a.out, "DONE"), "w").write(f"{n_ok} ok, {n_err} errors\n")
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()
