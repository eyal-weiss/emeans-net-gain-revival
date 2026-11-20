"""Second confirmatory round: medg_pm (primary) and medg_fh (alternative) against PR #5915 and the other rules
(protocol: arxiv/confirm2/PROTOCOL.md, 2026-11-19). Seeds 30-49 had never been used by any run of this project.

Parts (see PARTS), in priority order:
  P   primary:  34 cells (16 development + 18 additional; MNIST and Fashion scaled by 1/255), seeds 30-49, random starts, T = 2
  S5  secondary: the same 34 cells, seeds 30-39, random starts, T = 5
  SK  secondary: the same 34 cells, seeds 30-39, k-means++ starts (sklearn kmeans_plusplus, random_state = seed), T = 2
  EQ2, EQ5  equivalence gate (run with NGP_IMPL=py into their own directory): medg_pm and medg_fh with the reference
      implementation at seeds 30-31 (T = 2; run right after P) and seed 30 (T = 5; run right after S5), compared run by
      run with their C++ twins in P and S5. No confirmatory decision is issued without the gate (confirm2_analyze.py).
Every part runs the 7 configs of CONFIGS7 (EQ2, EQ5: the two variants).

Stopping and resuming. Every finished run is written at once, atomically, to OUT/runs/<key>.json; a stop (SIGTERM to the
process group, stop_r2.sh) loses only the runs in progress. Starting again with the same arguments skips the finished
runs and continues in the same order. Resuming refuses if a finished run was made with other code or another
environment (code, FAISS build, numerical libraries: hashes in every record), on another host, from other data (train
and test hashes) or from another k-means++ start, unless --allow-code-change is given after a logged amendment. Jobs run part by part, seed by seed (largest cells
first within a seed), so a stopped part holds complete seeds except the last few.
k-means++ starts are computed beforehand with --starts-only, in the environment that has scikit-learn 1.9.1 (the PR's
environment has none), and saved atomically under a file lock; an SK job whose start is missing fails (it is not computed
in the run's environment).

usage: python -u confirm2_grid.py --host H --out RUNDIR --parts P,S5,SK [--workers 8 --threads 4] --kmpp-dir DIR/kmpp
       [--datasets a,b,...] [--seeds 100,101] [--dry-run] [--status] [--starts-only] [--max-jobs N]
"""
import argparse, fcntl, hashlib, json, os, random, sys, time, traceback

EXP = os.environ.get("EXPERIMENTS_DIR")
os.environ.setdefault("EMEANS_RR_DATA", os.environ.get("DISTANCE_REVIVAL_DATA", ""))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, EXP)
import grid  # noqa: E402

CONFIGS7 = ["lib", "pr", "ours", "h21", "ngp2_full", "ngp2_full_medg_pm", "ngp2_full_medg_fh"]
VARIANTS = ["ngp2_full_medg_pm", "ngp2_full_medg_fh"]
CELLS16 = [("kdd_z", 256), ("kdd_z", 1024), ("kdd_z", 4096), ("census_z", 256), ("census_z", 1024),
           ("census_z", 4096), ("msd_z", 256), ("msd_z", 1024), ("msd_z", 4096), ("geonames", 256),
           ("geonames", 1024), ("geonames", 4096), ("power_z", 256), ("power_z", 1024), ("shuttle_z", 256),
           ("shuttle_z", 1024)]
NEW18 = [("mnist_255", 256), ("mnist_255", 1024), ("fashion_255", 256), ("fashion_255", 1024),
         ("corel_z", 256), ("corel_z", 1024), ("road3d_z", 256), ("road3d_z", 1024), ("road3d_z", 4096),
         ("codrna_z", 256), ("codrna_z", 1024), ("codrna_z", 4096), ("ijcnn1_z", 256), ("ijcnn1_z", 1024),
         ("sensit_z", 256), ("sensit_z", 1024), ("higgs_z", 256), ("higgs_z", 1024)]
CELLS34 = CELLS16 + NEW18
R = lambda lo, hi: list(range(lo, hi + 1))
# part -> list of (cells, seeds, start, T, configs)
PARTS = {
    "P": [(CELLS34, R(30, 49), "random", 2, CONFIGS7)],
    "S5": [(CELLS34, R(30, 39), "random", 5, CONFIGS7)],
    "SK": [(CELLS34, R(30, 39), "kmpp", 2, CONFIGS7)],
    "EQ2": [(CELLS34, R(30, 31), "random", 2, VARIANTS)],
    "EQ5": [(CELLS34, [30], "random", 5, VARIANTS)],
}
EQ_PARTS = ("EQ2", "EQ5")
# default placement (Census is stored only on host_b; every other dataset is on both hosts); balanced by estimated cost
HOST = {"census_z": "host_b", "msd_z": "host_b", **{ds: "host_b" for ds, _ in NEW18},
        "kdd_z": "host_a", "geonames": "host_a", "power_z": "host_a", "shuttle_z": "host_a"}
SIZE = dict(grid.SIZE, mnist_255=(5e4, 784), fashion_255=(5e4, 784), corel_z=(4.8e4, 16), road3d_z=(4.1e5, 3),
            codrna_z=(3.2e5, 8), ijcnn1_z=(1.2e5, 22), sensit_z=(7.9e4, 100), higgs_z=(7.8e4, 28))
SMOKE_SEEDS_MIN = 100          # smoke tests use seeds >= 100 (or development seeds < 30); never 30-49


def run_key(cfg, ds, k, seed, start, T):
    return f"cfm2|{cfg}|{ds}|k{k}|s{seed}|{start}|T{T}"


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


_IDENTITY = {}


def code_identity(impl):
    """Hashes of everything that defines a run's computation (checked on resume; stored in every record): this
    project's code, the installed FAISS (PR #5915 build: its e_means module and compiled libraries) and the versions of
    Python and NumPy. Computed once per process."""
    if impl in _IDENTITY:
        return _IDENTITY[impl]
    import glob as _glob
    import platform
    import numpy
    import faiss
    files = ["confirm2_grid.py", "ablation.py", "ngp_variants.py", "ngp_variants2.py"] + \
        (["ablation_cpp.py", "ngp_kernels.cpp", "ngp_kernels.so"] if impl == "cpp" else [])
    out = {f: _sha(os.path.join(HERE, f)) for f in files}
    out.update({"grid.py": _sha(os.path.join(EXP, "grid.py")), "datasets.py": _sha(os.path.join(EXP, "datasets.py"))})
    fdir = os.path.dirname(faiss.__file__)
    for f in sorted(_glob.glob(os.path.join(fdir, "contrib", "e_means", "*.py")) + _glob.glob(os.path.join(fdir, "*.so"))):
        out["faiss/" + os.path.relpath(f, fdir)] = _sha(f)
    out.update({"faiss": faiss.__version__, "numpy": numpy.__version__, "python": platform.python_version(),
                "impl": impl})
    _IDENTITY[impl] = out
    return out


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


def finished_keys(out):
    """Keys of the completed runs, from the file names only (no result is read)."""
    d = os.path.join(out, "runs")
    if not os.path.isdir(d):
        return set()
    return {f[:-5].replace("__", "|") for f in os.listdir(d) if f.endswith(".json")}


def load_finished(out):
    """{key: record} of the completed runs (an unreadable file counts as a record with no identity, so it fails the
    resume check instead of being silently rerun)."""
    recs = {}
    d = os.path.join(out, "runs")
    if os.path.isdir(d):
        for f in os.listdir(d):
            if f.endswith(".json"):
                try:
                    r = json.load(open(os.path.join(d, f)))
                    recs[r["key"]] = r
                except Exception:
                    recs[f[:-5].replace("__", "|")] = {}
    return recs


def data_identity(ds):
    """(train sha256 prefix, test sha256 prefix) of a prepared dataset, from its meta.json, after checking that the
    actual files have exactly these hashes (refuses otherwise). Computed once per dataset and process."""
    if ds in _DATA:
        return _DATA[ds]
    import datasets
    d = os.path.join(datasets.data_root(), ds)
    m = json.load(open(os.path.join(d, "meta.json")))
    for part in ("train", "test"):
        h = hashlib.sha256()
        with open(os.path.join(d, f"{part}.npy"), "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                h.update(chunk)
        if h.hexdigest() != m[f"{part}_sha256"]:
            raise SystemExit(f"refusing to run: {ds}/{part}.npy does not match its meta.json hash")
    _DATA[ds] = (str(m["train_sha256"])[:16], str(m["test_sha256"])[:16])
    return _DATA[ds]


_DATA = {}


def start_identity(kdir, ds, k, seed):
    import numpy as np
    path = kmpp_path(kdir, ds, k, seed)
    if not os.path.exists(path):
        return None
    return hashlib.sha256(np.ascontiguousarray(np.load(path), dtype=np.float32).tobytes()).hexdigest()[:16]


def check_finished(recs, impl, host, kdir):
    """Reasons why finished runs may not be combined with runs made now (empty list: resume is safe)."""
    current = code_identity(impl)
    bad, dcache = [], {}
    for key, r in sorted(recs.items()):
        if r.get("code") != current:
            bad.append(f"{key}: other code or environment")
            continue
        if not str(r.get("host", "")).startswith(host):
            bad.append(f"{key}: made on {r.get('host')}")
            continue
        ds = r["dataset"]
        if ds not in dcache:
            dcache[ds] = data_identity(ds)
        if (r.get("data_sha"), r.get("data_test_sha")) != dcache[ds]:
            bad.append(f"{key}: other data")
            continue
        if r["init"] == "kmpp" and r.get("start_sha") != start_identity(kdir, ds, r["k"], r["seed"]):
            bad.append(f"{key}: other k-means++ start")
    return bad


def kmpp_path(kdir, ds, k, seed):
    return os.path.join(kdir, f"{ds.replace(':', '_')}_k{k}_s{seed}.npy")


def ensure_start(kdir, ds, k, seed):
    """The k-means++ start (grid.kmpp_job: sklearn kmeans_plusplus, random_state = seed), computed once under a lock."""
    assert os.path.basename(kdir.rstrip("/")) == "kmpp", kdir
    path = kmpp_path(kdir, ds, k, seed)
    assert path == grid._kmpp_path(os.path.dirname(kdir.rstrip("/")), ds, k, seed)
    if os.path.exists(path):
        return path, 0.0
    os.makedirs(kdir, exist_ok=True)
    with open(path + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if os.path.exists(path):
            return path, 0.0
        return grid.kmpp_job((os.path.dirname(kdir.rstrip("/")), ds, k, seed))


def plan(parts, host, datasets=None, seeds=None):
    """Jobs (rank, part, ds, k, seed, start, T, configs): one per (dataset, k, seed, start, T) of this host."""
    seen, jobs = set(), []
    for rank, p in enumerate(parts):
        for cells, pseeds, start, T, cfgs in PARTS[p]:
            for ds, k in cells:
                if (datasets is None and HOST[ds] != host) or (datasets is not None and ds not in datasets):
                    continue
                for s in (seeds if seeds is not None else pseeds):
                    todo = [c for c in cfgs if (c, ds, k, s, start, T) not in seen]
                    seen.update((c, ds, k, s, start, T) for c in todo)
                    if todo:
                        jobs.append((rank, p, ds, k, s, start, T, todo))
    return jobs


def run_job(args):
    out, ds, k, seed, start, T, configs, kdir, part = args
    import numpy as np
    import datasets
    import ngp_variants2 as V            # every config: lib, pr, ours, h21, ngp2_full (frozen) and the two variants
    impl = os.environ.get("NGP_IMPL", "py")
    recs, errs = [], []
    try:
        X, Xte, info = datasets.load(ds, seed)
        n, d = X.shape
        start_sha = None
        if start == "random":
            rows = np.sort(np.random.RandomState(seed).permutation(n)[:k])          # same init as grid.py
            init = np.ascontiguousarray(X[rows], dtype=np.float32)
        else:
            path = kmpp_path(kdir, ds, k, seed)
            if not os.path.exists(path):
                raise FileNotFoundError(f"k-means++ start missing: {path} (compute it with --starts-only)")
            init = np.ascontiguousarray(np.load(path), dtype=np.float32)
            assert init.shape == (k, d), (init.shape, k, d)
            start_sha = hashlib.sha256(init.tobytes()).hexdigest()[:16]
    except Exception:
        return [], [dict(dataset=ds, k=k, seed=seed, start=start, T=T, error=traceback.format_exc())]
    order = list(configs)
    random.Random(f"{ds}|{k}|{seed}|{start}|T{T}").shuffle(order)
    code = code_identity(impl)
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
                       impl=impl, stat=getattr(m, "stat", None), seconds=secs,
                       times={kk: float(v) for kk, v in V.TIMES.items()}, n_splits=int(m.n_splits),
                       batch=int(m.recipe["batch_size"]), steps_per_pass=int(m.recipe["steps_per_pass"]),
                       alpha0=float(m.recipe["alpha0"]), load_before=load0, load_after=os.getloadavg()[0],
                       start_sha=start_sha, data_sha=str(info["train_sha256"])[:16],
                       data_test_sha=str(info["test_sha256"])[:16], code=code)
            rec.update(grid._score(C, X, Xte, info, ds, k))
            rec.update(n=int(n), d=int(d), threads=grid._THREADS, host=os.uname().nodename, pr="5915@98deaca",
                       finished=time.strftime("%Y-%m-%d %H:%M:%S %Z"))
            _write_run(out, rec)
            recs.append(rec)
        except Exception:
            errs.append(dict(key=key, error=traceback.format_exc()))
    return recs, errs


def starts_job(args):
    kdir, ds, k, s = args
    t0 = time.perf_counter()
    path, secs = ensure_start(kdir, ds, k, s)
    return path, secs, time.perf_counter() - t0


def main():
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from multiprocessing import get_context
    p = argparse.ArgumentParser()
    p.add_argument("--host", required=True); p.add_argument("--out", required=True)
    p.add_argument("--parts", default="P,S5,SK", help="in priority order")
    p.add_argument("--workers", type=int, default=8); p.add_argument("--threads", type=int, default=4)
    p.add_argument("--kmpp-dir", required=True)
    p.add_argument("--datasets", default="", help="run these datasets instead of the host's default placement")
    p.add_argument("--seeds", default="", help="smoke tests only: these seeds (>= 100 or < 30) instead of the part's")
    p.add_argument("--dry-run", action="store_true"); p.add_argument("--status", action="store_true")
    p.add_argument("--starts-only", action="store_true", help="compute the k-means++ starts of the kmpp jobs and exit")
    p.add_argument("--max-jobs", type=int, default=0, help="smoke tests: run at most this many jobs, then exit")
    p.add_argument("--allow-code-change", action="store_true",
                   help="resume although finished runs were made with other code (only after a logged amendment)")
    a = p.parse_args()
    parts = a.parts.split(",")
    assert all(x in PARTS for x in parts), parts
    dsets = set(a.datasets.split(",")) if a.datasets else None
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else None
    if seeds is not None:
        assert all(s >= SMOKE_SEEDS_MIN or s < 30 for s in seeds), "seeds 30-99 are reserved for the protocol"
    J = plan(parts, a.host, dsets, seeds)
    impl = os.environ.get("NGP_IMPL", "py")
    done = finished_keys(a.out)
    if a.status:                 # counts only; no result is read
        from collections import Counter
        tot, fin = Counter(), Counter()
        for _, part, ds, k, s, st, T, cfgs in J:
            for c in cfgs:
                tot[part] += 1
                fin[part] += run_key(c, ds, k, s, st, T) in done
        print(" | ".join(f"{p_} {fin[p_]}/{tot[p_]}" for p_ in parts) + f" | total {sum(fin.values())}/{sum(tot.values())}")
        return
    if not a.starts_only:
        if any(x in EQ_PARTS for x in parts):
            assert all(x in EQ_PARTS for x in parts) and impl == "py", "EQ parts run alone, with the reference implementation"
        else:
            assert impl == "cpp", "the grid parts run with NGP_IMPL=cpp"
    os.makedirs(a.out, exist_ok=True)
    if a.starts_only:
        need = sorted({(ds, k, s) for _, _, ds, k, s, st, _, _ in J if st == "kmpp"
                       and not os.path.exists(kmpp_path(a.kmpp_dir, ds, k, s))},
                      key=lambda x: -SIZE[x[0]][0] * max(SIZE[x[0]][1], 16) * x[1])
        import sklearn
        print(f"{time.strftime('%H:%M:%S')} {len(need)} k-means++ starts to compute on {a.host} "
              f"(scikit-learn {sklearn.__version__})", flush=True)
        with ProcessPoolExecutor(a.workers, mp_context=get_context("spawn"), initializer=grid._init,
                                 initargs=(a.threads,)) as ex:
            futs = [ex.submit(starts_job, (a.kmpp_dir, ds, k, s)) for ds, k, s in need]
            for f in as_completed(futs):
                path, secs, wall = f.result()
                print(f"{time.strftime('%H:%M:%S')} {os.path.basename(path)} {secs:.1f} s", flush=True)
        open(os.path.join(a.out, "STARTS_DONE"), "w").write(f"{len(need)} starts\n")
        return
    for ds in sorted({j[2] for j in J}):           # the data files of this session match their recorded hashes
        data_identity(ds)
    bad = check_finished(load_finished(a.out), impl, a.host, a.kmpp_dir)
    if bad and not a.allow_code_change:
        print(f"refusing to resume: {len(bad)} finished runs differ from what would be run now, e.g. "
              + "; ".join(bad[:3]), flush=True)
        sys.exit(2)
    jobs = []
    for rank, part, ds, k, s, st, T, cfgs in J:
        todo = [c for c in cfgs if run_key(c, ds, k, s, st, T) not in done]
        if todo:
            jobs.append((rank, s, (a.out, ds, k, s, st, T, todo, a.kmpp_dir, part)))
    cost = lambda j: SIZE[j[1]][0] * max(SIZE[j[1]][1], 16) * j[2] * j[5] * len(j[6])
    jobs.sort(key=lambda rsj: (rsj[0], rsj[1], -cost(rsj[2])))      # part, then seed, then largest first
    jobs = [j for _, _, j in jobs]
    if a.max_jobs:
        jobs = jobs[:a.max_jobs]
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S %Z')} {len(jobs)} jobs ({sum(len(j[6]) for j in jobs)} runs) to do on "
          f"{a.host}, parts {parts}, {len(done)} runs already finished; workers {a.workers} x threads {a.threads}",
          flush=True)
    if a.dry_run:
        from collections import Counter
        c = Counter((j[8], j[1]) for j in jobs)
        for x in sorted(c):
            print("  ", x, c[x], "jobs")
        print(f"   relative cost {sum(cost(j) for j in jobs):.3e}")
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
    if not a.max_jobs:
        open(os.path.join(a.out, "DONE"), "w").write(f"{time.strftime('%Y-%m-%d %H:%M:%S %Z')}: "
                                                    f"{n_ok} ok, {n_err} errors in this session\n")
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()
