"""Confirmatory study of the frozen rule ngp2_full (protocol: arxiv/protocol_ngp.md, 2026-11-19).

Parts (see PARTS): C1 primary (16 development cells, new seeds 10-29, random starts, T = 2); C2 k-means++ starts
(seeds 0-9, v1 cache); C3 Covertype (z-scored and min-max); C4 embeddings (DEEP/SIFT/SPACEV 1M, k = 16384); C5
sampling-rate ablation; C6 T = 5 passes; C7 DEEP 10M at k = 65536. TIMING: one job at a time, 8 threads.

Jobs: one per (part, dataset, k, seed, start, T) with the configs that part needs there (data loaded once,
configs in a shuffled order), or one per config with --split-configs. Every finished run is written at once to
RUNDIR/runs/<key>.json (atomic); resuming skips those runs and refuses if they were made with other code (hashes in
each record). results.jsonl is an append-only log of the same records, for convenience.
usage: python -u confirm_grid.py --host H --out RUNDIR --parts C1,C2,... [--workers 6 --threads 4]
       [--kmpp-dir DIR] [--split-configs] [--dry-run] [--smoke]
"""
import argparse, hashlib, json, os, random, sys, time, traceback

EXP = os.environ.get("EXPERIMENTS_DIR")
os.environ.setdefault("EMEANS_RR_DATA", os.environ.get("DISTANCE_REVIVAL_DATA", ""))
sys.path.insert(0, EXP)
import grid  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = ["lib", "pr", "ours", "h21", "ngp2_full"]
STRIDES = ["ngp2_full", "ngp2_s2", "ngp2_s4", "ngp2", "ngp2_s16", "ngp2_s32"]      # sampling 1 row in s = 1..32
CELLS16 = [("kdd_z", 256), ("kdd_z", 1024), ("kdd_z", 4096), ("census_z", 256), ("census_z", 1024),
           ("census_z", 4096), ("msd_z", 256), ("msd_z", 1024), ("msd_z", 4096), ("geonames", 256),
           ("geonames", 1024), ("geonames", 4096), ("power_z", 256), ("power_z", 1024), ("shuttle_z", 256),
           ("shuttle_z", 1024)]
COVTYPE = [(ds, k) for ds in ("covtype_z", "covtype_mm") for k in (256, 1024, 4096)]
EMB3 = [("deep1m", 16384), ("sift1m", 16384), ("spacev1m", 16384)]
R = lambda lo, hi: list(range(lo, hi + 1))
# part -> list of (cells, seeds, start, T, configs)
PARTS = {
    "C1": [(CELLS16, R(10, 29), "random", 2, MAIN)],
    "C2": [(CELLS16, R(0, 9), "kmpp", 2, MAIN)],
    "C3": [(COVTYPE, R(10, 29), "random", 2, MAIN), (COVTYPE, R(0, 9), "kmpp", 2, MAIN)],
    "C4": [(EMB3, R(10, 19), "random", 2, MAIN)],
    "C5": [(CELLS16, R(10, 19), "random", 2, STRIDES), (EMB3, R(10, 12), "random", 2, STRIDES)],
    "C6": [(CELLS16, R(10, 14), "random", 5, MAIN)],
    "C7": [([("deep10m", 65536)], R(10, 11), "random", 2, ["ngp2_full", "pr", "lib"])],
    # equivalence of the C++ implementation with the reference (run with NGP_IMPL=py; compared run by run with C1/C5)
    "EQ": [(CELLS16, R(10, 11), "random", 2, STRIDES)],
    "TIMING": [([("kdd_z", 4096), ("msd_z", 4096), ("geonames", 4096), ("power_z", 1024), ("deep1m", 16384),
                 ("spacev1m", 16384)], R(10, 12), "random", 2, MAIN + STRIDES[1:])],
}
# data placement for this study (host_b is heavily loaded by other users: only the datasets stored only there)
HOST = {"census_z": "host_b", "covtype_z": "host_b", "covtype_mm": "host_b", "sift1m": "host_b",
        "kdd_z": "host_a", "msd_z": "host_a", "power_z": "host_a", "geonames": "host_a", "shuttle_z": "host_a",
        "deep1m": "host_a", "spacev1m": "host_a", "deep10m": "host_a"}
# smoke test: runs that must reproduce stored records bit for bit (ngp5 grid, seed 0), plus the new paths
SMOKE = [("shuttle_z", 256, 0, "random", 2, ["lib", "pr", "ngp2", "ngp2_full"]),
         ("power_z", 256, 0, "random", 2, ["ngp2", "ngp2_full"]),
         ("power_z", 256, 0, "kmpp", 2, ["lib", "ngp2_full", "ngp2_s32"]),
         ("shuttle_z", 256, 10, "random", 5, ["ngp2_full", "h21", "ours"])]


def run_key(cfg, ds, k, seed, start, T):
    return f"cfm|{cfg}|{ds}|k{k}|s{seed}|{start}|T{T}"


def parts_of(cfg, ds, k, seed, start, T, parts):
    return [p for p in parts for cells, seeds, st, TT, cfgs in PARTS[p]
            if (ds, k) in cells and seed in seeds and st == start and TT == T and cfg in cfgs]


def plan(parts, host):
    """Jobs in priority order: each run is assigned to the first part (in the given order) that needs it, and a job
    holds the runs of one part at one (dataset, k, seed, start, T)."""
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


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def code_identity(impl):
    """Hashes of everything that defines a run's computation (checked on resume; stored in every record)."""
    files = ["ablation.py", "confirm_grid.py"] + (["ablation_cpp.py", "ngp_kernels.cpp", "ngp_kernels.so"]
                                                   if impl == "cpp" else [])
    return {f: _sha(os.path.join(HERE, f)) for f in files}


def _run_file(out, key):
    return os.path.join(out, "runs", key.replace("|", "__") + ".json")


def _write_run(out, rec):
    """Durable, atomic per-run record (a stop between runs loses nothing; a stop during the write leaves no file)."""
    path = _run_file(out, rec["key"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "w") as fh:
        fh.write(json.dumps(rec))
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def load_runs(out):
    """All completed runs of an output directory (per-run files; unreadable files are ignored)."""
    recs = {}
    d = os.path.join(out, "runs")
    if os.path.isdir(d):
        for f in os.listdir(d):
            if f.endswith(".json"):
                try:
                    r = json.load(open(os.path.join(d, f)))
                    recs[r["key"]] = r
                except Exception:
                    pass
    return recs


def kmpp_path(kdir, ds, k, seed):
    return os.path.join(kdir, f"{ds.replace(':', '_')}_k{k}_s{seed}.npy")


def run_job(args):
    out, ds, k, seed, start, T, configs, parts, kdir, mode = args
    import numpy as np
    import datasets
    if os.environ.get("NGP_IMPL", "py") == "cpp":     # same rule, runtime-relevant parts in C++ (ablation_cpp.py)
        import ablation_cpp as ablation
    else:
        import ablation
    recs, errs = [], []
    try:
        X, Xte, info = datasets.load(ds, seed)
        n, d = X.shape
        if start == "random":
            rows = np.sort(np.random.RandomState(seed).permutation(n)[:k])          # same init as grid.py
            init = np.ascontiguousarray(X[rows], dtype=np.float32)
        else:
            init = np.ascontiguousarray(np.load(kmpp_path(kdir, ds, k, seed)), dtype=np.float32)
            assert init.shape == (k, d), (init.shape, k, d)
    except Exception:
        return [], [dict(dataset=ds, k=k, seed=seed, start=start, T=T, error=traceback.format_exc())]
    order = list(configs)
    random.Random(f"{ds}|{k}|{seed}|{start}|T{T}").shuffle(order)
    code = code_identity(os.environ.get("NGP_IMPL", "py"))
    for cfg in order:
        key = run_key(cfg, ds, k, seed, start, T)
        try:
            spec = ablation.CONFIGS[cfg]
            removal, donor, split, timing = spec[:4]
            extra = {} if len(spec) <= 4 else (spec[4] if isinstance(spec[4], dict) else {"w": spec[4]})
            ablation.TIMES.clear()
            load0 = os.getloadavg()[0]
            m = ablation.AblationEmeans(d, k, removal, donor, split, timing, nepochs=T, seed=seed, **extra)
            t0 = time.perf_counter()
            C, _ = m.train(X, init_centroids=init)
            secs = time.perf_counter() - t0
            load1 = os.getloadavg()[0]
            rec = dict(key=key, config=cfg, dataset=ds, k=k, seed=seed, init=start, T=T, mode=mode,
                       impl=os.environ.get("NGP_IMPL", "py"),
                       parts=parts_of(cfg, ds, k, seed, start, T, parts),
                       removal=removal, donor=donor, split=split, timing=timing, seconds=secs,
                       times={kk: float(v) for kk, v in ablation.TIMES.items()},
                       n_splits=int(m.n_splits), n_candidates=int(m.n_candidates),
                       n_protected=int(m.n_protected), n_abandoned=int(m.n_abandoned),
                       batch=int(m.recipe["batch_size"]), steps_per_pass=int(m.recipe["steps_per_pass"]),
                       load_before=load0, load_after=load1, code=code)
            rec.update(grid._score(C, X, Xte, info, ds, k))
            rec.update(n=int(n), d=int(d), threads=grid._THREADS, host=os.uname().nodename, pr="5915@98deaca",
                       finished=time.strftime("%Y-%m-%d %H:%M:%S"))
            _write_run(out, rec)
            recs.append(rec)
        except Exception:
            errs.append(dict(key=key, error=traceback.format_exc()))
    return recs, errs


def main():
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from multiprocessing import get_context
    p = argparse.ArgumentParser()
    p.add_argument("--host", required=True); p.add_argument("--out", required=True)
    p.add_argument("--parts", default="C1,C4,C5,C2,C3,C6", help="in priority order")
    p.add_argument("--workers", type=int, default=6); p.add_argument("--threads", type=int, default=4)
    p.add_argument("--kmpp-dir", default="")
    p.add_argument("--split-configs", action="store_true", help="one job per config (results written per run)")
    p.add_argument("--dry-run", action="store_true"); p.add_argument("--smoke", action="store_true")
    p.add_argument("--seed-filter", type=int, default=None, help="only jobs with this seed (C7 lanes)")
    p.add_argument("--skip-file", default="", help="run keys to skip (one per line; C7 runs that hit the cap)")
    p.add_argument("--max-runs", type=int, default=0, help="run at most this many jobs, then exit")
    p.add_argument("--next-key", action="store_true", help="print the key of the next run to do and exit")
    p.add_argument("--allow-code-change", action="store_true",
                   help="resume although finished runs were made with other code (only after a logged amendment)")
    a = p.parse_args()
    parts = a.parts.split(",")
    assert all(x in PARTS for x in parts), parts
    mode = "smoke" if a.smoke else ("timing" if parts == ["TIMING"] else "grid")
    if a.smoke:
        J = [(0, "smoke", ds, k, s, st, T, cfgs) for ds, k, s, st, T, cfgs in SMOKE]
    else:
        J = plan(parts, a.host)
    os.makedirs(a.out, exist_ok=True)
    res_path = os.path.join(a.out, "results.jsonl")
    finished = load_runs(a.out)
    current = code_identity(os.environ.get("NGP_IMPL", "py"))
    stale = sorted(k for k, r in finished.items() if r.get("code") != current)
    if stale and not a.allow_code_change:
        print(f"refusing to resume: {len(stale)} finished runs were made with other code, e.g. {stale[0]}: "
              f"{finished[stale[0]].get('code')} vs {current}", flush=True)
        sys.exit(2)
    done = set(finished)
    if a.skip_file and os.path.exists(a.skip_file):
        done |= {l.strip() for l in open(a.skip_file) if l.strip()}
    if a.seed_filter is not None:
        J = [j for j in J if j[4] == a.seed_filter]
    jobs = []
    for rank, part, ds, k, s, st, T, cfgs in J:
        todo = [c for c in cfgs if run_key(c, ds, k, s, st, T) not in done]
        if a.split_configs:
            jobs += [(rank, (a.out, ds, k, s, st, T, [c], parts, a.kmpp_dir, mode)) for c in todo]
        elif todo:
            jobs.append((rank, (a.out, ds, k, s, st, T, todo, parts, a.kmpp_dir, mode)))
    cost = lambda j: grid.SIZE[j[1]][0] * max(grid.SIZE[j[1]][1], 16) * j[2] * j[5] * len(j[6])
    if mode == "timing":       # seeds in increasing order, so partial timing results cover every cell
        jobs.sort(key=lambda rj: (rj[1][3], -cost(rj[1])))
    else:                      # parts in the given priority order, largest jobs first within a part
        jobs.sort(key=lambda rj: (rj[0], -cost(rj[1])))
    jobs = [j for _, j in jobs]
    if a.next_key:
        print(run_key(jobs[0][6][0], *jobs[0][1:3], jobs[0][3], jobs[0][4], jobs[0][5]) if jobs else "")
        return
    if a.max_runs:
        jobs = jobs[:a.max_runs]
    print(f"{time.strftime('%H:%M:%S')} {len(jobs)} jobs ({sum(len(j[6]) for j in jobs)} runs) on {a.host}, "
          f"parts {parts}, workers {a.workers} x threads {a.threads}", flush=True)
    if a.dry_run:
        from collections import Counter
        c = Counter((j[1], j[4], j[5], len(j[6])) for j in jobs)
        for x in sorted(c):
            print("  ", x, c[x], "jobs")
        print(f"   relative cost {sum(cost(j) for j in jobs):.3e}")
        return
    n_ok = n_err = 0

    def record(recs, errs):
        nonlocal n_ok, n_err
        with open(res_path, "a") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")
        if errs:
            with open(os.path.join(a.out, "errors.jsonl"), "a") as fh:
                for e in errs:
                    fh.write(json.dumps(e) + "\n")
        n_ok += len(recs); n_err += len(errs)
        print(f"{time.strftime('%H:%M:%S')} runs ok {n_ok} errors {n_err}", flush=True)

    if a.workers == 1:         # in this process (TIMING, C7 lanes): a timeout on this process stops the run
        grid._init(a.threads)
        for j in jobs:
            record(*run_job(j))
    else:
        with ProcessPoolExecutor(a.workers, mp_context=get_context("spawn"), initializer=grid._init,
                                 initargs=(a.threads,)) as ex:
            for f in as_completed([ex.submit(run_job, j) for j in jobs]):
                record(*f.result())
    open(os.path.join(a.out, "DONE"), "w").write(f"{n_ok} ok, {n_err} errors\n")
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()
