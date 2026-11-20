"""EXPLORATORY measurement for an early end of the move phase (2026-11-19 night; not pre-registered; descriptive only).

Question: how much of the move phase (two-nearest search, margin and axis statistics, revival checks) is spent after the
rule's moves have stopped paying, i.e. what a rule that ends revival early could save. The run is round 2's medg_pm
(NGP_IMPL=cpp, checks every 5th batch) at round-2 keys on round-2 hosts, so every final result must equal the stored
round-2 record exactly (the logging must not change the run). Per revival check it records:
  step, epoch, rows done, frozen flag, accepted moves, the moves' predicted gains and costs (sum of w*gain - cost =
  predicted net gain), candidates, the best donor's gain, the cheapest cost, the last batch's objective and rows, the
  decay product prod(1 - alpha) (for half-life stop rules), and a snapshot of the cumulative component timers.
Parts: M2 = 34 cells x seeds 30-32, T = 2;  M5 = 34 cells x seeds 30-31, T = 5.
Records: OUT/runs/<key>.json, key stm|cfg|ds|k|s|random|T (atomic, resumable).
usage: NGP_IMPL=cpp python -u stopmeasure_grid.py --host H --out DIR [--parts M2,M5 --workers 7 --threads 4]
"""
import os, sys
import argparse, json, time, traceback  # noqa: E401

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import confirm2_grid as C2  # noqa: E402
import grid  # noqa: E402

CFG = "ngp2_full_medg_pm"
PARTS = {"M2": (C2.CELLS34, [30, 31, 32], 2), "M5": (C2.CELLS34, [30, 31], 5)}


def run_key(ds, k, seed, T):
    return f"stm|{CFG}|{ds}|k{k}|s{seed}|random|T{T}"


def make_class(V):
    import numpy as np

    class Logged(V.AblationEmeans):
        def begin(self, global_n, init_centroids):
            super().begin(global_n, init_centroids)
            self.check_log, self._lg_obj, self._lg_rows, self._lg_done = [], None, 0, 0
            self._lg_t0 = time.perf_counter()

        def _apply_statistics(self, ids, sums, counts, objective, alpha, decay, replace_usage=False):
            super()._apply_statistics(ids, sums, counts, objective, alpha, decay, replace_usage)
            self._lg_obj = None if objective is None else float(objective)
            self._lg_rows = int(np.asarray(counts).sum())
            self._lg_done += self._lg_rows

        def _choose_splits(self, rng):
            nc0 = self.n_candidates
            src, tgt = super()._choose_splits(rng)
            cp = self._calib_pred or {}
            gain, cost = cp.get("gain"), cp.get("cost")
            e = dict(step=int(self._step), epoch=int(self._epoch), rows=int(self._lg_done),
                     frozen=bool(self._frozen()), moves=0 if src is None else int(len(src)),
                     cand=int(self.n_candidates - nc0), obj=self._lg_obj, batch_rows=self._lg_rows,
                     decay_prod=float(self._decay_prod), wall=time.perf_counter() - self._lg_t0,
                     times={kk: float(v) for kk, v in V.TIMES.items()})
            if gain is not None and cost is not None:
                cool = self._stale > 0.5
                don = (self._usage > 0) & ~cool
                e["best_gain"] = float(self.w * gain[don].max()) if don.any() else 0.0
                e["min_cost"] = float(cost[~cool].min()) if (~cool).any() else None
                e["sum_cost_all"] = float(cost.sum())
            if src is not None:
                g, c = self.w * gain[src], cost[tgt]
                e.update(gain_sum=float(g.sum()), cost_sum=float(c.sum()), net_sum=float((g - c).sum()),
                         net_max=float((g - c).max()))
            self.check_log.append(e)
            return src, tgt

    return Logged


def run_job(args):
    out, ds, k, seed, T = args
    import numpy as np
    import datasets
    import ngp_variants2 as V
    key = run_key(ds, k, seed, T)
    try:
        assert os.environ.get("NGP_IMPL") == "cpp"
        X, Xte, info = datasets.load(ds, seed)
        n, d = X.shape
        rows = np.sort(np.random.RandomState(seed).permutation(n)[:k])          # the round-2 random start
        init = np.ascontiguousarray(X[rows], dtype=np.float32)
        spec = V.CONFIGS[CFG]
        V.TIMES.clear()
        m = make_class(V)(d, k, *spec[:4], nepochs=T, seed=seed, **spec[4])
        t0 = time.perf_counter()
        C, _ = m.train(X, init_centroids=init)
        secs = time.perf_counter() - t0
        rec = dict(key=key, config=CFG, dataset=ds, k=k, seed=seed, init="random", T=T, seconds=secs,
                   times={kk: float(v) for kk, v in V.TIMES.items()}, n_splits=int(m.n_splits),
                   batch=int(m.recipe["batch_size"]), steps_per_pass=int(m.recipe["steps_per_pass"]),
                   total_steps=int(m.recipe["total_steps"]), checks=m.check_log,
                   runner=C2._sha(os.path.abspath(__file__)), data_sha=str(info["train_sha256"])[:16],
                   code=C2.code_identity("cpp"))
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
    p.add_argument("--parts", default="M2,M5")
    p.add_argument("--workers", type=int, default=7); p.add_argument("--threads", type=int, default=4)
    p.add_argument("--max-jobs", type=int, default=0)
    a = p.parse_args()
    done = C2.finished_keys(a.out)
    jobs = []
    for rank, part in enumerate(a.parts.split(",")):
        cells, seeds, T = PARTS[part]
        for s in seeds:
            for ds, k in cells:
                if C2.HOST[ds] != a.host or run_key(ds, k, s, T) in done:
                    continue
                jobs.append(((rank, s, -C2.SIZE[ds][0] * max(C2.SIZE[ds][1], 16) * k * T), (a.out, ds, k, s, T)))
    jobs = [j for _, j in sorted(jobs)]
    if a.max_jobs:
        jobs = jobs[:a.max_jobs]
    print(f"{time.strftime('%H:%M:%S')} {len(jobs)} runs to do on {a.host} ({len(done)} finished)", flush=True)
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
