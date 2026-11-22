"""Checks net_gain_revival.NetGainRevivalEmeans on the PR's head driver (79440bc) against the research class of the
paper's third round (ngp_head.NetGainHead with the guard), and its early stop against a reference built on that class.

For each (dataset, k, seed, T) and freeze in {True, False}:
  1. early_stop=None: the release class with the NumPy computations, and with the C++ kernels, against NetGainHead:
     identical final centres (bit for bit), number of splits and checks;
  2. early_stop="hl": the release class (NumPy and C++) against HeadES below (NetGainHead plus the early stop of
     version 1.0.0, written separately): identical final centres, splits, checks and stop step.
Prints one line per comparison and exits with code 1 on any difference.
usage (head overlay first on the path; the research files and the package importable):
  FAISS_OVERLAY=<faiss_head_79440bc> python verify_head.py --cells shuttle_z:256,corel_z:1024 --seeds 200 --T 2,5
"""
import os
import sys

OVERLAY = os.environ.get("FAISS_OVERLAY", "")
if OVERLAY:
    sys.path.insert(0, OVERLAY)
import argparse  # noqa: E402
import hashlib  # noqa: E402
import math  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from multiprocessing import get_context  # noqa: E402

import numpy as np  # noqa: E402


def _head_es_class():
    import ngp_head

    class HeadES(ngp_head.NetGainHead):
        """NetGainHead with the early stop "hl": a check without a move is quiet; the move phase ends at a quiet check
        once the sum of log(decay) since the last check with a move reaches log(1/2)."""

        def begin(self, global_n, init_centroids):
            super().begin(global_n, init_centroids)
            self._es_stopped, self._es_logq, self.es_stop_step = False, 0.0, None

        def _frozen_ng(self):
            return self._es_stopped or super()._frozen_ng()

        def _apply_statistics(self, ids, sums, counts, objective, alpha, decay, replace_usage=False):
            super()._apply_statistics(ids, sums, counts, objective, alpha, decay, replace_usage)
            if not self._es_stopped:
                g = 0.0 if replace_usage else float(decay)
                self._es_logq += math.log(g) if g > 0.0 else -math.inf

        def _choose_splits(self, rng):
            if self._frozen_ng():
                return None, None
            src, tgt = super()._choose_splits(rng)
            if src is not None:
                self._es_logq = 0.0
            elif self._es_logq <= math.log(0.5):
                self._es_stopped = True
                self.es_stop_step = int(self._step)
            return src, tgt

    return HeadES


def one(args):
    ds, k, seed, T, freeze, threads = args
    os.environ["OMP_NUM_THREADS"] = str(threads)
    import faiss
    faiss.omp_set_num_threads(threads)
    import datasets
    import ngp_head
    from net_gain_revival.emeans import NetGainRevivalEmeans
    X, _, _ = datasets.load(ds, seed)
    n, d = X.shape
    X = np.ascontiguousarray(X, dtype=np.float32)
    init = np.ascontiguousarray(X[np.sort(np.random.RandomState(seed).permutation(n)[:k])])

    def fit(m):
        C, _ = m.train(X, init_centroids=init)
        return (hashlib.sha256(np.ascontiguousarray(C).tobytes()).hexdigest()[:16], int(m.n_splits),
                int(getattr(m, "n_checks", -1)), getattr(m, "es_stop_step", None))

    out = []
    ref = fit(ngp_head.NetGainHead(d, k, nepochs=T, seed=seed, freeze=freeze, guard=True))
    for nat in (False, True):
        got = fit(NetGainRevivalEmeans(d, k, nepochs=T, seed=seed, freeze=freeze, early_stop=None, use_native=nat))
        out.append((f"{ds} k{k} s{seed} T{T} freeze={freeze} no-stop native={nat}", ref[:3], got[:3]))
    ref = fit(_head_es_class()(d, k, nepochs=T, seed=seed, freeze=freeze, guard=True))
    for nat in (False, True):
        got = fit(NetGainRevivalEmeans(d, k, nepochs=T, seed=seed, freeze=freeze, early_stop="hl", use_native=nat))
        out.append((f"{ds} k{k} s{seed} T{T} freeze={freeze} stop=hl native={nat}", ref, got))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cells", required=True)
    p.add_argument("--seeds", default="200")
    p.add_argument("--T", default="2")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    cells = [(c.split(":")[0], int(c.split(":")[1])) for c in a.cells.split(",")]
    jobs = [(ds, k, int(s), int(T), fz, a.threads) for ds, k in cells for s in a.seeds.split(",")
            for T in a.T.split(",") for fz in (False, True)]
    bad = 0
    with ProcessPoolExecutor(a.workers, mp_context=get_context("spawn")) as ex:
        for res in ex.map(one, jobs):
            for name, ref, got in res:
                ok = ref == got
                bad += not ok
                print(("OK   " if ok else "DIFF ") + name + ("" if ok else f"  ref={ref} got={got}") +
                      (f"  splits={got[1]} checks={got[2]} stop={got[3] if len(got) > 3 else '-'}"), flush=True)
    print(f"{'ALL IDENTICAL' if bad == 0 else f'{bad} DIFFERENCES'} ({sum(1 for _ in jobs) * 4} comparisons)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
