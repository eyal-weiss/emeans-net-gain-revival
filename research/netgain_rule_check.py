"""Check that netgain_rule.decide reproduces the CPU classes' decisions exactly (2026-11-19 evening).

Runs the CPU rule (ngp_variants2: frozen "mean", "medg_pm", "medg_fh"; NGP_IMPL=cpp or py) and, at every revival check,
builds the state the rule sees, calls netgain_rule.decide on it and compares the moves with the class's own.
usage: python netgain_rule_check.py OUT.json  [cells as ds:k,...] [seeds 0,1] [T 2,5]
"""
import json, os, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.environ["EXPERIMENTS_DIR"])
out = sys.argv[1]
cells = [(c.split(":")[0], int(c.split(":")[1])) for c in (sys.argv[2] if len(sys.argv) > 2 else
         "shuttle_z:256,shuttle_z:1024,kdd_z:256,corel_z:1024,power_z:256,geonames:1024").split(",")]
seeds = [int(s) for s in (sys.argv[3] if len(sys.argv) > 3 else "0,1").split(",")]
Ts = [int(t) for t in (sys.argv[4] if len(sys.argv) > 4 else "2,5").split(",")]
sys.argv = sys.argv[:1]
import grid  # noqa: E402
grid._init(int(os.environ.get("OMP_NUM_THREADS", "4")))
import datasets  # noqa: E402
import ngp_variants2 as V  # noqa: E402
import netgain_rule as NR  # noqa: E402


class Check(V.AblationEmeans):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.n_checked = self.n_same = 0
        self.mismatch = []

    def _state(self):
        cache = getattr(self, "_check_cache", None)
        return dict(k=self.k, step=self._step, alpha0=self.recipe["alpha0"], total_steps=self.recipe["total_steps"],
                    usage=np.array(self._usage, dtype=np.float64),
                    unorm=self._axis_scale * np.linalg.norm(self._removal_stats[:, :self.d].astype("float64"), axis=1),
                    centroids=np.array(self._centroids, copy=True), readout=np.array(self._readout(), copy=True),
                    m_n=self._m_n.copy(), m_sum=self._m_sum.copy(), m_sq=self._m_sq.copy(), p2=self._p2.copy(),
                    p4=self._p4.copy(), cum_n=self._cum_n.copy(), age=self._age.copy(), stale=self._stale.copy(),
                    decay_prod=float(self._decay_prod),
                    prev_delta=None if cache is None or cache.get("delta") is None else np.array(cache["delta"]))

    def _choose_net_gain(self):
        st = self._state()
        src, tgt = super()._choose_net_gain()
        s2, t2, _ = NR.decide(st, stat=self.stat, w=self.w)
        same = ((src is None and s2 is None) or (src is not None and s2 is not None and np.array_equal(src, s2)
                                                 and np.array_equal(tgt, t2)))
        self.n_checked += 1
        self.n_same += bool(same)
        if not same and len(self.mismatch) < 3:
            self.mismatch.append(dict(step=int(self._step), cls=None if src is None else [len(src), src[:5].tolist()],
                                      fn=None if s2 is None else [len(s2), s2[:5].tolist()]))
        return src, tgt


res = []
for ds, k in cells:
    X, Xte, info = datasets.load(ds, 0)
    n, d = X.shape
    for T in Ts:
        for seed in seeds:
            rows = np.sort(np.random.RandomState(seed).permutation(n)[:k])
            init = np.ascontiguousarray(X[rows], dtype=np.float32)
            for stat in ("mean", "medg_pm", "medg_fh"):
                cfg = "ngp2_full" if stat == "mean" else f"ngp2_full_{stat}"
                spec = V.CONFIGS[cfg]
                t0 = time.perf_counter()
                m = Check(d, k, *spec[:4], nepochs=T, seed=seed, **spec[4])
                C, _ = m.train(X, init_centroids=init)
                r = dict(ds=ds, k=k, T=T, seed=seed, stat=stat, checks=m.n_checked, identical=m.n_same,
                         splits=int(m.n_splits), mismatch=m.mismatch, secs=round(time.perf_counter() - t0, 1))
                res.append(r)
                print(json.dumps(r), flush=True)
json.dump(res, open(out, "w"))
tot, same = sum(r["checks"] for r in res), sum(r["identical"] for r in res)
print(f"TOTAL: {same} of {tot} checks identical in {len(res)} runs", flush=True)
