"""Check that the clean implementation (net_gain_revival) reproduces the research code on the paper's data.

For each of the 34 cells on this host, seed 30, the research class (ngp_stop.AblationEmeans with the paper's
configuration) and the clean class (NumPy and C++ paths) train from the same start; final centres, number of splits
and early-stop step must be identical. Cases: medg_pm with the half-life stop (T = 2 and T = 5), medg_pm without it
(T = 2), medg_fh with it (T = 2). One JSON line per comparison in OUT.
usage (from research/, EXPERIMENTS_DIR=$PWD EMEANS_RR_DATA=... NGP_IMPL=cpp):
  python -u verify_clean.py --host host_a --out ../results/clean_equivalence.jsonl [--workers 7]
"""
import os, sys
import argparse, json, time, traceback  # noqa: E401

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import confirm2_grid as C2  # noqa: E402
import grid  # noqa: E402

CASES = [("medg_pm", "hl", 2), ("medg_pm", None, 2), ("medg_fh", "hl", 2), ("medg_pm", "hl", 5)]


def job(args):
    ds, k, seed = args
    import numpy as np
    import datasets
    import ngp_stop
    from net_gain_revival import NetGainRevivalEmeans
    out = []
    try:
        X, _, _ = datasets.load(ds, seed)
        n, d = X.shape
        rows = np.sort(np.random.RandomState(seed).permutation(n)[:k])
        init = np.ascontiguousarray(X[rows], dtype=np.float32)
        for stat, es, T in CASES:
            spec = ngp_stop.CONFIGS[{"medg_pm": "ngp2_full_medg_pm", "medg_fh": "ngp2_full_medg_fh"}[stat]]
            res = {}
            for impl in ("research", "clean_numpy", "clean_native"):
                if impl == "research":
                    m = ngp_stop.AblationEmeans(d, k, *spec[:4], nepochs=T, seed=seed, early_stop=es, **spec[4])
                else:
                    m = NetGainRevivalEmeans(d, k, nepochs=T, seed=seed, stat=stat, early_stop=es,
                                             use_native=impl == "clean_native")
                t0 = time.perf_counter()
                C, _ = m.train(X, init_centroids=init)
                res[impl] = (np.asarray(C), int(m.n_splits), getattr(m, "es_stop_step", None), time.perf_counter() - t0)
            ref = res["research"]
            for impl in ("clean_numpy", "clean_native"):
                c = res[impl]
                out.append(dict(dataset=ds, k=k, seed=seed, stat=stat, early_stop=es, T=T, impl=impl,
                                identical=bool(np.array_equal(c[0], ref[0]) and c[1] == ref[1] and c[2] == ref[2]),
                                n_splits=c[1], es_stop_step=c[2], seconds=c[3], seconds_research=ref[3]))
    except Exception:
        out.append(dict(dataset=ds, k=k, seed=seed, error=traceback.format_exc()))
    return out


def main():
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from multiprocessing import get_context
    p = argparse.ArgumentParser()
    p.add_argument("--host", required=True); p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=7); p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    jobs = [(ds, k, 30) for ds, k in C2.CELLS34 if C2.HOST[ds] == a.host]
    n_ok = n_bad = n_err = 0
    with open(a.out, "a") as fh, ProcessPoolExecutor(a.workers, mp_context=get_context("spawn"), initializer=grid._init,
                                                     initargs=(a.threads,)) as ex:
        for f in as_completed([ex.submit(job, j) for j in jobs]):
            for r in f.result():
                fh.write(json.dumps(r) + "\n"); fh.flush()
                if "error" in r:
                    n_err += 1; print(r["error"], flush=True)
                elif r["identical"]:
                    n_ok += 1
                else:
                    n_bad += 1; print("DIFFERENT", r, flush=True)
            print(f"{time.strftime('%H:%M:%S')} identical {n_ok} different {n_bad} errors {n_err}", flush=True)
    sys.exit(1 if (n_bad or n_err) else 0)


if __name__ == "__main__":
    main()
