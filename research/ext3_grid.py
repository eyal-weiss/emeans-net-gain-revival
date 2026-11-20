"""EXPLORATORY broad check of medg_fh (Fay-Herriot prior for the gain; ngp_variants2.py), 2026-11-19 afternoon.
Not pre-registered. The same keys as the extended grid (ext_grid.py): only the new config is run; its comparators are the
extended-grid and confirmatory records.
  FO2/FO5/FOK: the 16 development cells at the C1, C6 and C2 keys;  FN2/FN5/FNK: the 18 new cells (seeds as in ext_grid)
  FT3: the 12 stress cells at T = 3 (ext2 keys)
Every dataset is stored on host_b; --cells old|new|all chooses what this host runs.
Previous docstring (ext2_grid.py, from which this runner is derived):

  T3 / T10 : more passes (T = 3, 10) on 12 cells (the KDD and Shuttle cells, one cell of each new dataset), seeds 0-4
  EB       : a revival check after EVERY batch (frozen: ngp2b, unstable in development; medg_pm_b), T = 2, seeds 0-4
  SY / SY5 : synthetic far groups (synth_far:m, k = 105: can the rule find 5 far groups the random start missed?)
             and heterogeneous scales (synth_hetero:d), seeds 0-9 (T = 2) and 0-4 (T = 5); scored in-sample
  C3V      : Covertype (z and min-max) at the confirmatory C3 keys, the four variants
  C4V      : the three 1M embeddings at the confirmatory C4 keys, medg_pm only
Records: RUNDIR/runs/<key>.json with key prefix ext2. Run with NGP_IMPL=cpp.
usage: python -u ext2_grid.py --host H --out RUNDIR --parts T3,T10,EB,SY,SY5,C3V,C4V [--workers 8 --threads 4] --kmpp-dir DIR
"""
import argparse, json, os, random, sys, time, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import confirm_grid as CG  # noqa: E402
import ext_grid as E  # noqa: E402
import ext2_grid as E2  # noqa: E402
import grid  # noqa: E402

FH = ["ngp2_full_medg_fh"]
R = lambda lo, hi: list(range(lo, hi + 1))
PARTS = {
    "FO2": [(CG.CELLS16, R(10, 29), "random", 2, FH)],
    "FO5": [(CG.CELLS16, R(10, 14), "random", 5, FH)],
    "FOK": [(CG.CELLS16, R(0, 9), "kmpp", 2, FH)],
    "FN2": [(E.NEW_CELLS, R(0, 19), "random", 2, FH)],
    "FN5": [(E.NEW_CELLS, R(0, 4), "random", 5, FH)],
    "FNK": [(E.NEW_CELLS, R(0, 9), "kmpp", 2, FH)],
    "FT3": [(E2.STRESS, R(0, 4), "random", 3, FH)],
}
OLD_DS = {ds for ds, _ in CG.CELLS16}
SIZE = dict(E.SIZE)




def run_key(cfg, ds, k, seed, start, T):
    return f"ext3|{cfg}|{ds}|k{k}|s{seed}|{start}|T{T}"


def code_identity():
    files = ["ablation.py", "ablation_cpp.py", "ngp_kernels.cpp", "ngp_kernels.so", "ngp_variants.py", "ngp_variants2.py", "ext2_grid.py", "ext3_grid.py"]
    return {f: CG._sha(os.path.join(HERE, f)) for f in files}


def plan(parts, host, which="all"):
    seen, jobs = set(), []
    for rank, p in enumerate(parts):
        for cells, seeds, start, T, cfgs in PARTS[p]:
            for ds, k in cells:
                if which != "all" and (ds in OLD_DS) != (which == "old"):
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
    import ngp_variants2 as V
    CONF = V.CONFIGS
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
            spec = CONF[cfg]
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


def main():
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from multiprocessing import get_context
    p = argparse.ArgumentParser()
    p.add_argument("--host", required=True); p.add_argument("--out", required=True)
    p.add_argument("--parts", required=True)
    p.add_argument("--workers", type=int, default=8); p.add_argument("--threads", type=int, default=4)
    p.add_argument("--kmpp-dir", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--cells", default="all", choices=["old", "new", "all"])
    a = p.parse_args()
    parts = a.parts.split(",")
    assert all(x in PARTS for x in parts), parts
    J = plan(parts, a.host, a.cells)
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
