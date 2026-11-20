"""EXPLORATORY revival-check interval ablation (ngp_interval.py; 2026-11-19 evening; not pre-registered).

  IV : 34 cells (round-2 cells), development seeds 0-4, random starts, T = 2;
       pr and medg_pm with a check after every 1, 2, 5, 10, 20 batches, and medg_pm with the event trigger.
Seeds 30-49 are the round-2 confirmatory seeds; 50 and above stay unused. Same host placement as round 2.
Records: OUT/runs/<key>.json (atomic; resumable; refuses other code). Run with NGP_IMPL=cpp.
usage: python -u interval_grid.py --host H --out DIR [--workers 8 --threads 4] [--seeds 0,1] [--datasets a,b]
"""
import argparse, json, os, random, sys, time, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import confirm2_grid as C2  # noqa: E402
import grid  # noqa: E402

CONFIGS = ["pr@1", "pr@2", "pr@5", "pr@10", "pr@20",
           "medg_pm@1", "medg_pm@2", "medg_pm@5", "medg_pm@10", "medg_pm@20", "medg_pm@ev"]
SEEDS = [0, 1, 2, 3, 4]


def run_key(cfg, ds, k, seed, T):
    return f"ivl|{cfg}|{ds}|k{k}|s{seed}|random|T{T}"


def code_identity():
    files = ["interval_grid.py", "ngp_interval.py", "ngp_variants.py", "ngp_variants2.py", "ablation.py",
             "ablation_cpp.py", "ngp_kernels.cpp", "ngp_kernels.so"]
    return {f: C2._sha(os.path.join(HERE, f)) for f in files}


def run_job(args):
    out, ds, k, seed, T, configs = args
    import numpy as np
    import datasets
    import ngp_interval as V
    recs, errs = [], []
    try:
        X, Xte, info = datasets.load(ds, seed)
        n, d = X.shape
        rows = np.sort(np.random.RandomState(seed).permutation(n)[:k])
        init = np.ascontiguousarray(X[rows], dtype=np.float32)
    except Exception:
        return [], [dict(dataset=ds, k=k, seed=seed, error=traceback.format_exc())]
    order = list(configs)
    random.Random(f"{ds}|{k}|{seed}|T{T}").shuffle(order)
    code = code_identity()
    for cfg in order:
        key = run_key(cfg, ds, k, seed, T)
        try:
            spec = V.CONFIGS[cfg]
            V.TIMES.clear()
            m = V.AblationEmeans(d, k, *spec[:4], nepochs=T, seed=seed, **spec[4])
            t0 = time.perf_counter()
            C, _ = m.train(X, init_centroids=init)
            secs = time.perf_counter() - t0
            rec = dict(key=key, config=cfg, dataset=ds, k=k, seed=seed, init="random", T=T, seconds=secs,
                       times={kk: float(v) for kk, v in V.TIMES.items()}, n_splits=int(m.n_splits),
                       n_checks=int(getattr(m, "n_checks", 0)), n_full_checks=int(getattr(m, "n_full_checks", 0)),
                       steps=int(m._step), impl=os.environ.get("NGP_IMPL", "py"), code=code,
                       data_sha=str(info["train_sha256"])[:16])
            rec.update(grid._score(C, X, Xte, info, ds, k))
            rec.update(n=int(n), d=int(d), threads=grid._THREADS, host=os.uname().nodename,
                       finished=time.strftime("%Y-%m-%d %H:%M:%S %Z"))
            C2._write_run(out, rec)
            recs.append(rec)
        except Exception:
            errs.append(dict(key=key, error=traceback.format_exc()))
    return recs, errs


def main():
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from multiprocessing import get_context
    p = argparse.ArgumentParser()
    p.add_argument("--host", required=True); p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=8); p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seeds", default=""); p.add_argument("--datasets", default="")
    p.add_argument("--configs", default="")
    a = p.parse_args()
    assert os.environ.get("NGP_IMPL") == "cpp"
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else SEEDS
    assert not any(30 <= s <= 99 for s in seeds), "seeds 30-99 are reserved"
    dsets = set(a.datasets.split(",")) if a.datasets else None
    cfgs = a.configs.split(",") if a.configs else CONFIGS
    current = code_identity()
    done = C2.finished_keys(a.out)
    for key in done:
        if json.load(open(C2._run_file(a.out, key))).get("code") != current:
            raise SystemExit(f"refusing to resume: {key} was made with other code")
    jobs = []
    for s in seeds:
        for ds, k in C2.CELLS34:
            if (dsets is None and C2.HOST[ds] != a.host) or (dsets is not None and ds not in dsets):
                continue
            todo = [c for c in cfgs if run_key(c, ds, k, s, 2) not in done]
            if todo:
                jobs.append(((s, -C2.SIZE[ds][0] * max(C2.SIZE[ds][1], 16) * k), (a.out, ds, k, s, 2, todo)))
    jobs = [j for _, j in sorted(jobs)]
    print(f"{time.strftime('%H:%M:%S')} {len(jobs)} jobs ({sum(len(j[5]) for j in jobs)} runs) on {a.host}", flush=True)
    n_ok = n_err = 0
    with ProcessPoolExecutor(a.workers, mp_context=get_context("spawn"), initializer=grid._init,
                             initargs=(a.threads,)) as ex:
        for f in as_completed([ex.submit(run_job, j) for j in jobs]):
            recs, errs = f.result()
            if errs:
                with open(os.path.join(a.out, "errors.jsonl"), "a") as fh:
                    fh.write("".join(json.dumps(e) + "\n" for e in errs))
            n_ok += len(recs); n_err += len(errs)
            print(f"{time.strftime('%H:%M:%S')} runs ok {n_ok} errors {n_err}", flush=True)
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()
