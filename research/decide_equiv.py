"""Equivalence and timing of the C++ decision (ngp_fast, ngp_decide.cpp) against the round-2 implementation
(ngp_variants2 with NGP_IMPL=cpp: per-row statistics and matching in C++, per-check estimates in NumPy).
2026-11-19 night. Each job trains one key twice in the same process, reference then C++ decision, and records:
  * per check, while the runs are comparable (up to and including the first check whose moves differ): bit identity
    (uint64 views) of the rates, nearest-other distances (delta, also medg_fh's next prior input), costs and gains,
    of rho and of the number of candidates; the largest relative difference over finite entries; mismatches in
    non-finite entries; then whether the accepted moves (donors, freed centres, in order) are identical;
  * final test and training error, splits: identical or not;
  * time: total seconds, and the decision's own time (the 'decide' timer minus the nearest-other-centre search; the
    matching is included for both) for both, plus the number of checks.
Parts: E2 = 34 cells x seeds 30-31, T = 2; E5 = 34 cells x seed 30, T = 5; stats mean (frozen), medg_pm, medg_fh.
Records: OUT/runs/<key>.json, key deq|stat|ds|k|s|random|T. Same host placement as round 2.
usage: NGP_IMPL=cpp python -u decide_equiv.py --host H --out DIR [--parts E2,E5 --workers 7 --threads 4]
"""
import os, sys
import argparse, json, time, traceback  # noqa: E401

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import confirm2_grid as C2  # noqa: E402
import grid  # noqa: E402

STATS = {"mean": "ngp2_full", "medg_pm": "ngp2_full_medg_pm", "medg_fh": "ngp2_full_medg_fh"}
PARTS = {"E2": (C2.CELLS34, [30, 31], 2), "E5": (C2.CELLS34, [30], 5)}


def run_key(stat, ds, k, seed, T):
    return f"deq|{stat}|{ds}|k{k}|s{seed}|random|T{T}"


def logged(base, sink):
    class L(base):
        def _choose_splits(self, rng):
            nc0 = self.n_candidates
            src, tgt = super()._choose_splits(rng)
            est = self._calib_pred is not None             # _choose_splits resets it; the cache may be a previous check's
            cp, cc = (self._calib_pred or {}), ((self._check_cache or {}) if est else {})
            cp_ = lambda key: None if cp.get(key) is None else np.array(cp[key], dtype=np.float64, copy=True)
            cc_ = lambda key: None if cc.get(key) is None else np.array(cc[key], dtype=np.float64, copy=True)
            sink.append(dict(step=int(self._step), src=None if src is None else src.tolist(),
                             tgt=None if tgt is None else tgt.tolist(), gain=cp_("gain"), cost=cp_("cost"),
                             rate=cc_("rate"), delta=cc_("delta"), rho=float(getattr(self, "_rho", float("nan"))),
                             ncand=int(self.n_candidates - nc0), estimated=est))
            return src, tgt
    return L


def compare_arrays(x, y):
    """(bit-identical, max relative difference over entries finite in both, non-finite mismatches)"""
    if x is None or y is None:
        return (x is None and y is None), 0.0, 0
    if x.shape != y.shape:
        return False, float("inf"), -1
    bits = bool(np.array_equal(x.view(np.uint64), y.view(np.uint64)))
    fin = np.isfinite(x) & np.isfinite(y)
    den = np.maximum(np.abs(x[fin]), np.abs(y[fin]))
    rel = float(np.max(np.where(den > 0, np.abs(x[fin] - y[fin]) / np.where(den > 0, den, 1.0), 0.0))) if fin.any() else 0.0
    nonfin = int(np.sum(~fin & ~((np.isnan(x) & np.isnan(y)) | (x == y))))
    return bits, rel, nonfin


def run_job(args):
    out, ds, k, seed, T, stat = args
    import numpy as np
    import datasets
    import ngp_variants2 as V
    import ngp_fast as F
    key = run_key(stat, ds, k, seed, T)
    try:
        X, Xte, info = datasets.load(ds, seed)
        n, d = X.shape
        rows = np.sort(np.random.RandomState(seed).permutation(n)[:k])
        init = np.ascontiguousarray(X[rows], dtype=np.float32)
        spec = V.CONFIGS[STATS[stat]]
        res = {}
        for impl, base in (("ref", V.AblationEmeans), ("cpp", F.AblationEmeans)):
            log = []
            V.TIMES.clear()
            m = logged(base, log)(d, k, *spec[:4], nepochs=T, seed=seed, **spec[4])
            t0 = time.perf_counter()
            C, _ = m.train(X, init_centroids=init)
            secs = time.perf_counter() - t0
            if impl == "cpp":          # every check with estimates went through the C++ decision
                assert m._fast_ok() and getattr(m, "n_fast", 0) == sum(e["estimated"] for e in log) > 0
            tm = {kk: float(v) for kk, v in V.TIMES.items()}
            own = tm.get("decide", 0.0) - tm.get("delta_search", 0.0)        # matching included for both
            sc = grid._score(C, X, Xte, info, ds, k)
            res[impl] = dict(log=log, secs=secs, times=tm, decide_own=own, n_splits=int(m.n_splits), score=sc)
        a, b = res["ref"]["log"], res["cpp"]["log"]
        ARR = ("gain", "cost", "rate", "delta")
        st = {kk: dict(not_bitwise=0, max_rel=0.0, nonfinite=0) for kk in ARR}
        first_diff, compared, rho_bad, ncand_bad, est_bad = None, 0, 0, 0, 0
        for i in range(max(len(a), len(b))):
            if i >= len(a) or i >= len(b) or a[i]["step"] != b[i]["step"]:
                first_diff = i
                break
            compared += 1
            est_bad += a[i]["estimated"] != b[i]["estimated"]
            for kk in ARR:                                 # estimates first, also at the first differing check
                bits, rel, nf = compare_arrays(a[i][kk], b[i][kk])
                st[kk]["not_bitwise"] += not bits
                st[kk]["max_rel"] = max(st[kk]["max_rel"], rel)
                st[kk]["nonfinite"] += nf
            ra, rb = a[i]["rho"], b[i]["rho"]
            rho_bad += not (np.float64(ra).view(np.uint64) == np.float64(rb).view(np.uint64))     # bits, NaN payloads too
            ncand_bad += a[i]["ncand"] != b[i]["ncand"]
            if a[i]["src"] != b[i]["src"] or a[i]["tgt"] != b[i]["tgt"]:
                first_diff = i
                break
        R, Cc = res["ref"], res["cpp"]
        rec = dict(key=key, stat=stat, dataset=ds, k=k, seed=seed, init="random", T=T,
                   checks_ref=len(a), checks_cpp=len(b), checks_compared=compared, first_diff=first_diff,
                   decisions_identical=first_diff is None and len(a) == len(b),
                   moves=sum(len(e["src"]) for e in a if e["src"] is not None),
                   arrays=st, rho_not_bitwise=rho_bad, ncand_mismatch=ncand_bad, estimated_mismatch=est_bad,
                   gain_max_rel=st["gain"]["max_rel"], cost_max_rel=st["cost"]["max_rel"],
                   checks_gain_not_bitwise=st["gain"]["not_bitwise"], checks_cost_not_bitwise=st["cost"]["not_bitwise"],
                   final_identical=all(R["score"][m_] == Cc["score"][m_] for m_ in ("test_mse", "train_mse"))
                   and R["n_splits"] == Cc["n_splits"],
                   ref=dict(secs=R["secs"], decide_own=R["decide_own"], times=R["times"], n_splits=R["n_splits"],
                            test_mse=R["score"]["test_mse"], train_mse=R["score"]["train_mse"]),
                   cpp=dict(secs=Cc["secs"], decide_own=Cc["decide_own"], times=Cc["times"], n_splits=Cc["n_splits"],
                            test_mse=Cc["score"]["test_mse"], train_mse=Cc["score"]["train_mse"]),
                   batch=int(m.recipe["batch_size"]), steps_per_pass=int(m.recipe["steps_per_pass"]),
                   code=dict(ngp_decide=C2._sha(os.path.join(HERE, "ngp_decide.cpp")),
                             so=C2._sha(os.path.join(HERE, "ngp_decide.so")),
                             ngp_fast=C2._sha(os.path.join(HERE, "ngp_fast.py")),
                             runner=C2._sha(os.path.abspath(__file__))),
                   n=int(n), d=int(d), threads=grid._THREADS, host=os.uname().nodename,
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
    p.add_argument("--parts", default="E2,E5"); p.add_argument("--stats", default="medg_pm,medg_fh,mean")
    p.add_argument("--workers", type=int, default=7); p.add_argument("--threads", type=int, default=4)
    p.add_argument("--max-jobs", type=int, default=0)
    a = p.parse_args()
    assert os.environ.get("NGP_IMPL") == "cpp"
    done = C2.finished_keys(a.out)
    jobs = []
    for rank, part in enumerate(a.parts.split(",")):
        cells, seeds, T = PARTS[part]
        for s in seeds:
            for ds, k in cells:
                for st in a.stats.split(","):
                    if C2.HOST[ds] != a.host or run_key(st, ds, k, s, T) in done:
                        continue
                    jobs.append(((rank, s, -C2.SIZE[ds][0] * max(C2.SIZE[ds][1], 16) * k * T), (a.out, ds, k, s, T, st)))
    jobs = [j for _, j in sorted(jobs)]
    if a.max_jobs:
        jobs = jobs[:a.max_jobs]
    print(f"{time.strftime('%H:%M:%S')} {len(jobs)} jobs to do on {a.host} ({len(done)} finished)", flush=True)
    n_ok = n_err = 0
    with ProcessPoolExecutor(a.workers, mp_context=get_context("spawn"), initializer=grid._init,
                             initargs=(a.threads,)) as ex:
        for f in as_completed([ex.submit(run_job, j) for j in jobs]):
            recs, errs = f.result()
            if errs:
                with open(os.path.join(a.out, "errors.jsonl"), "a") as fh:
                    fh.write("".join(json.dumps(e) + "\n" for e in errs))
            n_ok += len(recs); n_err += len(errs)
            r = recs[0] if recs else None
            print(f"{time.strftime('%H:%M:%S')} ok {n_ok} err {n_err}" +
                  (f" {r['key']} dec={r['decisions_identical']} fin={r['final_identical']}" if r else ""), flush=True)
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()
