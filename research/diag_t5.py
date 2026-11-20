"""EXPLORATORY diagnosis of the T = 5 breakdown of ngp2_full (2026-11-19; not part of the confirmatory protocol).

Traces every revival move of one run: for the donor and the freed centre, the evidence the decision rested on (points
since birth, points in the last 5 batches, posterior rate, effective sample size of the gain, raw and posterior gain,
log-scale shrinkage), the population parameters at that check (mu, tau^2 of the log gains, pooled kurtosis), and at the
end where every centre came from (original, donor or child of a split; the donor's evidence at that split) and whether it
is empty. The rule itself is the frozen C++ implementation (ablation_cpp, via ngp_variants); the "mean" variant must reproduce the
confirmatory run exactly (checked against the stored record when one is given).

Variants (stat=...): see ngp_variants.py (mean = the frozen rule; medg, med, pm, medg_pm).
usage: python diag_t5.py OUT.json CONFIG DATASET K SEED T [STAT] [REFERENCE_RECORD.json]
"""
import json, os, sys, time
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.environ["EXPERIMENTS_DIR"])
out_path, cfg, ds, k, seed, T = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
stat = sys.argv[7] if len(sys.argv) > 7 else "mean"
ref_path = sys.argv[8] if len(sys.argv) > 8 else ""
sys.argv = sys.argv[:1]
import grid  # noqa: E402
grid._init(int(os.environ.get("OMP_NUM_THREADS", "4")))
import datasets  # noqa: E402
import ngp_variants2 as V  # noqa: E402  (frozen rule + variants incl. medg_fh; C++ parts with NGP_IMPL=cpp)


class Trace(V.AblationEmeans):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.moves, self.checks = [], []
        self._choice = None

    def begin(self, global_n, init_centroids):
        super().begin(global_n, init_centroids)
        self._ring = np.zeros((5, self.k))                 # points per centre in the last 5 batches
        self._origin = np.zeros(self.k, dtype=np.int64)    # 0 original, 1 donor of a split, 2 child (freed centre)
        self._origin_step = np.full(self.k, -1, dtype=np.int64)
        self._origin_zero = np.zeros(self.k, dtype=bool)   # the split's donor had no point in the 5 batches before
        self._origin_cum0 = np.zeros(self.k, dtype=bool)   # ... and none since its own birth
        self._obj_pass = defaultdict(lambda: [0.0, 0.0])         # pass -> [sum of batch objectives, rows]

    def _apply_statistics(self, ids, sums, counts, objective, alpha, decay, replace_usage=False):
        super()._apply_statistics(ids, sums, counts, objective, alpha, decay, replace_usage)
        self._ring[self._step % 5] = np.asarray(counts, dtype=np.float64)
        o = self._obj_pass[int(self._epoch)]
        o[0] += float(objective)
        o[1] += float(np.asarray(counts).sum())

    def _choose_net_gain(self):
        src, tgt = super()._choose_net_gain()
        P, G = self._calib_pred, self._gd
        rec = dict(step=int(self._step), epoch=int(self._epoch), alpha=G["alpha"] if G else None,
                   mu=G["mu"] if G else None, tau2=G["tau2"] if G else None, kv=G["kv"] if G else None,
                   window=G["window"] if G else None, moves=0 if src is None else len(src),
                   n_zero5=int((self._ring.sum(0) == 0).sum()), n_cum0=int(((self._cum_n == 0) & (self._age >= 5)).sum()),
                   usage0=int((self._usage <= 0).sum()))
        if G is not None:
            pos = np.isfinite(G["pm"])
            rec["gain_mean_over_median_max"] = float(np.exp(np.nanmax(G["pv"]) / 2)) if pos.any() else None
            # the five largest raw gains: posterior, evidence, cooling, chosen?
            used = set() if src is None else set(src.tolist())
            cooling = self._stale > 0.5
            top = np.argsort(-G["g"])[:5]
            post = P["gain"] if P is not None else np.zeros(self.k)
            rec["top_raw"] = [dict(id=int(i), raw=float(G["g"][i]), post=float(post[i]), neff=float(P["rate"][i] * G["window"]),
                                   rank_post=int((post > post[i]).sum()), cooling=bool(cooling[i]), used=int(i) in used,
                                   usage=float(self._usage[i]), norm=float(np.linalg.norm(self._centroids[i])))
                              for i in top]
            elig = ~cooling
            rec["min_cost"] = float(P["cost"][elig].min()) if P is not None and elig.any() else None
            rec["max_post_gain"] = float(post[(self._usage > 0) & elig].max()) if ((self._usage > 0) & elig).any() else None
        self.checks.append(rec)
        if src is not None and P is not None and G is not None:
            rate = P["rate"]
            recent = self._ring.sum(0)
            for d, j in zip(src.tolist(), tgt.tolist()):
                self.moves.append(dict(
                    step=int(self._step), epoch=int(self._epoch),
                    d=d, d_recent=float(recent[d]), d_cum=float(self._cum_n[d]), d_age=float(self._age[d]),
                    d_usage=float(self._usage[d]), d_rate=float(rate[d]), d_neff=float(rate[d] * G["window"]),
                    d_raw=float(G["g"][d]), d_post=float(P["gain"][d]), d_pm=float(G["pm"][d]), d_pv=float(G["pv"][d]),
                    d_stale=float(self._stale[d]), d_origin=int(self._origin[d]), d_origin_zero=bool(self._origin_zero[d]),
                    j=j, j_recent=float(recent[j]), j_cum=float(self._cum_n[j]), j_age=float(self._age[j]),
                    j_usage=float(self._usage[j]), j_rate=float(rate[j]), j_cost=float(P["cost"][j]),
                    j_delta=float(P["delta"][j]), j_origin=int(self._origin[j]),
                    j_mn=float(self._m_n[j]), j_r_raw=float(self._m_sum[j] / max(self._m_n[j], 1.0) / max(P["delta"][j], 1e-300)),
                    j_r_post=float(P["cost"][j] / max(rate[j] * P["delta"][j], 1e-300)),
                    j_cost_pr=float(self._m_sum[j] / max(self._m_n[j], 1.0) * rate[j]),
                    ratio_prior=list(getattr(self, "_ratio_prior", (None, None, None)))))
        return src, tgt

    def _apply_splits(self, source, target):
        recent = self._ring.sum(0)
        zero = recent[source] == 0
        cum0 = self._cum_n[source] == 0
        if not self.removal.startswith("ng"):          # PR and other rules: log the moves here
            for d, j in zip(np.asarray(source).tolist(), np.asarray(target).tolist()):
                self.moves.append(dict(step=int(self._step), epoch=int(self._epoch), d=d, d_recent=float(recent[d]),
                                       d_usage=float(self._usage[d]), d_origin=int(self._origin[d]), j=j,
                                       j_recent=float(recent[j]), j_usage=float(self._usage[j]),
                                       j_origin=int(self._origin[j])))
        n = super()._apply_splits(source, target)
        self._origin[source], self._origin[target] = 1, 2
        self._origin_step[source] = self._origin_step[target] = self._step
        self._origin_zero[source] = self._origin_zero[target] = zero
        self._origin_cum0[source] = self._origin_cum0[target] = cum0
        return n


X, Xte, info = datasets.load(ds, seed)
n, d = X.shape
rows = np.sort(np.random.RandomState(seed).permutation(n)[:k])
init = np.ascontiguousarray(X[rows], dtype=np.float32)
spec = V.CONFIGS[cfg]
extra = {} if len(spec) <= 4 else (spec[4] if isinstance(spec[4], dict) else {"w": spec[4]})
V.TIMES.clear()
if stat != "mean":
    extra = dict(extra, stat=stat)
m = Trace(d, k, *spec[:4], nepochs=T, seed=seed, **extra)
t0 = time.perf_counter()
C, _ = m.train(X, init_centroids=init)
secs = time.perf_counter() - t0
score = grid._score(C, X, Xte, info, ds, k)
# final emptiness by origin
import faiss  # noqa: E402
lab = np.empty(n, dtype=np.int64)
for s in range(0, n, 65536):
    lab[s:s + 65536] = faiss.knn(np.ascontiguousarray(X[s:s + 65536], dtype=np.float32), C, 1)[1].ravel()
cnt = np.bincount(lab, minlength=k)
empty = cnt == 0
by_origin = {}
for o, name in ((0, "original"), (1, "donor"), (2, "child")):
    sel = m._origin == o
    by_origin[name] = dict(centres=int(sel.sum()), empty=int((empty & sel).sum()),
                           from_zero_recent_donor=int((sel & m._origin_zero).sum()),
                           empty_from_zero_recent_donor=int((empty & sel & m._origin_zero).sum()),
                           from_cum0_donor=int((sel & m._origin_cum0).sum()),
                           empty_from_cum0_donor=int((empty & sel & m._origin_cum0).sum()))
splits_per_pass = defaultdict(int)
for mv in m.moves:
    splits_per_pass[mv["epoch"]] += 1
res = dict(config=cfg, stat=stat, dataset=ds, k=k, seed=seed, T=T, seconds=secs, n_splits=int(m.n_splits),
           splits_per_pass=dict(splits_per_pass), by_origin=by_origin,
           train_mse_by_pass={e: v[0] / max(v[1], 1) for e, v in m._obj_pass.items()},
           batch=int(m.recipe["batch_size"]), steps_per_pass=int(m.recipe["steps_per_pass"]),
           alpha0=float(m.recipe["alpha0"]), checks=m.checks, moves=m.moves, **score)
if ref_path and os.path.exists(ref_path):
    ref = json.load(open(ref_path))
    res["reproduces_record"] = (ref["test_mse"] == score["test_mse"] and ref["n_splits"] == int(m.n_splits))
json.dump(res, open(out_path, "w"))
np.save(out_path[:-5] + "_C.npy", C)
print(json.dumps({kk: res[kk] for kk in ("config", "stat", "dataset", "k", "seed", "T", "test_mse", "train_mse", "empty",
                                          "n_splits", "splits_per_pass", "by_origin", "seconds")}
                 | ({"reproduces_record": res["reproduces_record"]} if "reproduces_record" in res else {})), flush=True)
