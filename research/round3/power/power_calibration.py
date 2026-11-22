"""Round 3, design checks on the EXPLORATORY runs only (before any run at seeds 200-299).

Model: a simulated seed draws, for each of the 34 cells independently, one of that cell's observed log ratios
(exploratory seeds; 5 for the interval runs, 10 for the 5-epoch head-driver runs) and averages them (L_s). This keeps
the heavy-tailed cells (Shuttle k=256 test error, KDD k=256 training error). Two outputs:
  1. power of the registered tests at the planned n (Holm worst case: alpha/12 for I, alpha/8 for F);
  2. calibration of the t-based TOST at the margin: the per-cell values are shifted so that E[L_s] equals log(0.99) or
     log(1.01) exactly, and the rejection rate of the one-sided t-test at alpha/12 (n=100) and alpha/8 (n=40) is
     estimated (nominal 0.00417 and 0.00625).
usage: python power_calibration.py RESULTS_SIDE_DIR [reps]     (RESULTS_SIDE_DIR = results/pr5915/side)
"""
import glob, json, math, os, sys
import numpy as np
from scipy import stats

SIDE = sys.argv[1]
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
LO, HI = math.log(0.99), math.log(1.01)
rng = np.random.default_rng(20261120)


def table(pattern, key, seeds):
    db = {}
    for f in glob.glob(os.path.join(SIDE, "*", pattern)):
        r = json.load(open(f))
        db[key(r)] = r
    return db


def draw(M, n):
    c = M.shape[0]
    return M[np.arange(c)[None, :], rng.integers(0, M.shape[1], size=(n, c))].mean(1)


def boundary_rate(M, n, alpha, reps):
    """Shift so that E[L_s] (mean over cells of each cell's mean) equals the margin; one-sided t rejection rate."""
    out = {}
    for b, alt in ((LO, "greater"), (HI, "less")):
        Mb = M - M.mean() + b
        rej = sum(stats.ttest_1samp(draw(Mb, n), b, alternative=alt).pvalue < alpha for _ in range(reps))
        out[alt] = rej / reps
    return out


iv = table("side_interval/runs/ivl__medg_pm@*.json", lambda r: (r["config"], r["dataset"], r["k"], r["seed"]), None)
cells = sorted({(d, k) for (_, d, k, _) in iv})
print(f"interval: {len(cells)} cells, exploratory seeds 0-4; reps {REPS}")
for c in (1, 2):
    for m in ("test_mse", "train_mse"):
        M = np.array([[math.log(iv[(f"medg_pm@{c}", d, k, s)][m] / iv[("medg_pm@5", d, k, s)][m]) for s in range(5)]
                      for d, k in cells])
        rate = boundary_rate(M, 100, 0.05 / 12, REPS)
        skew = stats.skew(draw(M, 20000))
        print(f"  every {c} vs 5, {m[:-4]}: skewness of L_s {skew:+.2f}; t rejection at the margin, n=100, "
              f"alpha/12 = 0.00417: lower {rate['greater']:.5f}, upper {rate['less']:.5f}")
ph = table("side_prhead/runs/*.json", lambda r: (r["config"], r["dataset"], r["k"], r["seed"], r["T"]), None)
NF, FZ = "ngh79440bc_medg_pm_nofreeze_guard", "ngh79440bc_medg_pm_freeze_guard"
cells = sorted({(d, k) for (c, d, k, s, t) in ph if t == 5 and c == NF})
print(f"head driver, 5 epochs: {len(cells)} cells, exploratory seeds 30-39")
for m in ("test_mse", "train_mse"):
    M = np.array([[math.log(ph[(NF, d, k, s, 5)][m] / ph[(FZ, d, k, s, 5)][m]) for s in range(30, 40)] for d, k in cells])
    rate = boundary_rate(M, 40, 0.05 / 8, REPS)
    print(f"  without vs with the freeze, {m[:-4]}: skewness {stats.skew(draw(M, 20000)):+.2f}; t rejection at the "
          f"margin, n=40, alpha/8 = 0.00625: lower {rate['greater']:.5f}, upper {rate['less']:.5f}")
