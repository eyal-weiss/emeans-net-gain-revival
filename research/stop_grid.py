"""Automatic early end of the move phase: real runs (2026-11-19 night; exploratory but pre-specified below).

Configs (one job = one key; the four configs run in one process in a shuffled order, so time ratios are paired):
  pr        : the PR's rule at 98deaca (as round 2)
  medg_pm   : our rule, C++ decision (ngp_fast; identical decisions to round 2), no early stop
  medg_hl   : medg_pm + early stop "hl" (quiet = no accepted move; parameter-free)
  medg_e4   : medg_pm + early stop eps = 1e-4 (quiet = predicted net gain < 0.01% of the batch's squared error)
Parts (fresh seeds: the stop thresholds were chosen from the offline analysis of seeds 30-32):
  S2  : 34 cells x seeds 50-59, T = 2      S5 : 34 cells x seeds 50-59, T = 5      S10 : 34 cells x seeds 50-52, T = 10
Pre-specified analysis (stop_analyze.py): for each T and each stop config vs medg_pm and vs pr: seed-level
L_s = mean over cells of log(a/b) for test and training error, exp(mean) - 1, Hodges-Lehmann, Wilcoxon p (descriptive),
seeds better; per-dataset ratios; paired wall-clock ratio (median over runs) and the share of the move phase completed
when the rule fired; how often it fired. Hosts and placement as round 2 (KDD, GeoNames, Power, Shuttle on host_a).
Records: OUT/runs/<key>.json, key stp|cfg|ds|k|s|random|T (atomic, resumable).
usage: NGP_IMPL=cpp python -u stop_grid.py --host H --out DIR [--parts S2,S5,S10 --workers 7 --threads 4]
"""
import os, sys
import argparse, json, random, time, traceback  # noqa: E401

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import confirm2_grid as C2  # noqa: E402
import grid  # noqa: E402

R = lambda lo, hi: list(range(lo, hi + 1))
PARTS = {"S2": (C2.CELLS34, R(50, 59), 2), "S5": (C2.CELLS34, R(50, 59), 5), "S10": (C2.CELLS34, R(50, 52), 10)}
CONFIGS = ("pr", "medg_pm", "medg_hl", "medg_e4")
STOP = {"medg_pm": None, "medg_hl": "hl", "medg_e4": 1e-4}


def run_key(cfg, ds, k, seed, T):
    return f"stp|{cfg}|{ds}|k{k}|s{seed}|random|T{T}"


def code_ids():
    f = lambda n: C2._sha(os.path.join(HERE, n))
    return dict(runner=C2._sha(os.path.abspath(__file__)), ngp_stop=f("ngp_stop.py"), ngp_fast=f("ngp_fast.py"),
                ngp_decide=f("ngp_decide.cpp"), so=f("ngp_decide.so"), kernels=f("ngp_kernels.so"),
                variants=f("ngp_variants.py"), variants2=f("ngp_variants2.py"), ablation=f("ablation.py"),
                ablation_cpp=f("ablation_cpp.py"))


def run_job(args):
    out, ds, k, seed, T, todo = args
    import numpy as np
    import datasets
    import ngp_stop as S
    V = S._F._V
    recs, errs = [], []
    try:
        X, Xte, info = datasets.load(ds, seed)
        n, d = X.shape
        rows = np.sort(np.random.RandomState(seed).permutation(n)[:k])
        init = np.ascontiguousarray(X[rows], dtype=np.float32)
    except Exception:
        return [], [dict(dataset=ds, k=k, seed=seed, T=T, error=traceback.format_exc())]
    order = list(todo)
    random.Random(f"stp|{ds}|{k}|{seed}|T{T}").shuffle(order)
    code = code_ids()
    for cfg in order:
        key = run_key(cfg, ds, k, seed, T)
        try:
            V.TIMES.clear()
            if cfg == "pr":
                spec = V.CONFIGS["pr"]
                extra = {} if len(spec) <= 4 else (spec[4] if isinstance(spec[4], dict) else {"w": spec[4]})
                m = V.AblationEmeans(d, k, *spec[:4], nepochs=T, seed=seed, **extra)
            else:
                spec = V.CONFIGS["ngp2_full_medg_pm"]
                m = S.AblationEmeans(d, k, *spec[:4], nepochs=T, seed=seed, early_stop=STOP[cfg], **spec[4])
            load0 = os.getloadavg()[0]
            t0 = time.perf_counter()
            C, _ = m.train(X, init_centroids=init)
            secs = time.perf_counter() - t0
            if cfg != "pr":
                assert getattr(m, "n_fast", 0) > 0, "the C++ decision was not used"
            spp, tot = int(m.recipe["steps_per_pass"]), int(m.recipe["total_steps"])
            stop = getattr(m, "es_stop_step", None)
            rec = dict(key=key, config=cfg, dataset=ds, k=k, seed=seed, init="random", T=T, seconds=secs,
                       times={kk: float(v) for kk, v in V.TIMES.items()}, n_splits=int(m.n_splits),
                       early_stop=STOP.get(cfg), es_stop_step=stop,
                       es_stopped_frac=None if stop is None else (stop + 1) / max(tot - spp, 1),
                       batch=int(m.recipe["batch_size"]), steps_per_pass=spp, total_steps=tot,
                       alpha0=float(m.recipe["alpha0"]), load_before=load0, load_after=os.getloadavg()[0],
                       data_sha=str(info["train_sha256"])[:16], code=code)
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
    p.add_argument("--parts", default="S2,S5,S10")
    p.add_argument("--workers", type=int, default=7); p.add_argument("--threads", type=int, default=4)
    p.add_argument("--max-jobs", type=int, default=0)
    a = p.parse_args()
    assert os.environ.get("NGP_IMPL") == "cpp"
    done = C2.finished_keys(a.out)
    ids = code_ids()
    for key in sorted(done):
        r = json.load(open(C2._run_file(a.out, key)))
        if r.get("code") != ids:
            raise SystemExit(f"refusing to resume: {key} was made with other code")
    jobs = []
    for rank, part in enumerate(a.parts.split(",")):
        cells, seeds, T = PARTS[part]
        for s in seeds:
            for ds, k in cells:
                todo = [c for c in CONFIGS if run_key(c, ds, k, s, T) not in done]
                if C2.HOST[ds] != a.host or not todo:
                    continue
                jobs.append(((rank, s, -C2.SIZE[ds][0] * max(C2.SIZE[ds][1], 16) * k * T), (a.out, ds, k, s, T, todo)))
    jobs = [j for _, j in sorted(jobs)]
    if a.max_jobs:
        jobs = jobs[:a.max_jobs]
    print(f"{time.strftime('%H:%M:%S')} {len(jobs)} jobs to do on {a.host} ({len(done)} runs finished)", flush=True)
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
