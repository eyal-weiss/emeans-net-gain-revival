"""Pre-registered analysis of the third confirmatory round (confirm3/PROTOCOL.md, Section 4).

Inputs (fetched records): ROOT/<host>/r3_gate/runs, ROOT/<host>/r3_interval/runs, ROOT/<host>/r3_prhead/runs, and
the round-2 records ROUND2/<host>/cfm2_grid/runs (gate and data hashes).
usage: python confirm3_analyze.py ROOT ROUND2  > analysis.md
Exit code 1 if the gate or a provenance check fails (no inference is printed then) or if a part is incomplete (its
family is not tested).
"""
import glob, json, math, os, sys
from collections import defaultdict
from functools import lru_cache

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("EXPERIMENTS_DIR", os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import confirm2_grid as C2  # noqa: E402

CELLS34 = C2.CELLS34
ROOT, ROUND2 = sys.argv[1], sys.argv[2]
SEEDS_GATE = [30, 31]
SEEDS_I, SEEDS_F2, SEEDS_F5 = list(range(200, 300)), list(range(200, 260)), list(range(200, 240))
INTERVALS = (1, 2, 5, 10, 20)
NF, FZ, PR = "ngh79440bc_medg_pm_nofreeze_guard", "ngh79440bc_medg_pm_freeze_guard", "pr_head79440bc"
LO, HI = math.log(0.99), math.log(1.01)          # equivalence margin: geometric-mean ratio within [0.99, 1.01]
ALPHA = 0.05
# Registered code (PROTOCOL.md, Section 7): every record's identity must contain these hashes.
REGISTERED = {'interval_grid.py': 'd8661261e5bb586a', 'ngp_interval.py': 'd5d0a9d114bcbcd3', 'prhead_grid.py': 'bab5ed3e080c41d3', 'round3_prhead.py': '31ae07126f2abf6a', 'ngp_head.py': '69dbabd4318f2ce0', 'netgain_rule.py': 'bdb2e0e9f0ee0f14'}
I_FILES = ("interval_grid.py", "ngp_interval.py")
F_FILES = ("prhead_grid.py", "round3_prhead.py", "ngp_head.py", "netgain_rule.py")
HEAD_EMEANS = {"faiss/contrib/e_means/base.py": "e33dc8ad201f9d7d", "faiss/contrib/e_means/e_means.py": "89b9f3f537e613d9"}
SHARED = ("grid.py", "datasets.py", "faiss", "numpy", "python", "faiss/_swigfaiss.so", "faiss/_swigfaiss_avx2.so",
          "confirm2_grid.py", "ablation.py", "ngp_variants.py", "ngp_variants2.py")

ik = lambda c: (lambda ds, k, s: f"ivl|medg_pm@{c}|{ds}|k{k}|s{s}|random|T2")
fk = lambda cfg, T: (lambda ds, k, s: f"prh|{cfg}|{ds}|k{k}|s{s}|random|T{T}")
r2k = lambda ds, k, s: f"cfm2|ngp2_full_medg_pm|{ds}|k{k}|s{s}|random|T2"
short = lambda h: str(h).split(".")[0]


# ------------------------------------------------------------------ statistics
def wilcoxon_p(x, alternative="two-sided"):
    """Wilcoxon signed-rank p (exact without ties and n <= 50, else normal approximation with continuity correction).
    Conservative at degenerate inputs: no nonzero value -> 1; one nonzero value -> its exact p (0.5 one-sided in the
    direction of the alternative, else 1); a non-finite p -> 1."""
    x = np.asarray(x, float)
    x = x[x != 0]
    if len(x) == 0:
        return 1.0
    if len(x) == 1:
        if alternative == "greater":
            return 0.5 if x[0] > 0 else 1.0
        if alternative == "less":
            return 0.5 if x[0] < 0 else 1.0
        return 1.0
    exact = len(np.unique(np.abs(x))) == len(x) and len(x) <= 50
    p = float(stats.wilcoxon(x, zero_method="wilcox", method="exact" if exact else "approx",
                             correction=not exact, alternative=alternative).pvalue)
    return p if math.isfinite(p) else 1.0


def t_p(x, mu, alternative):
    """One-sample t-test p of mean(x) against mu; a degenerate sample (zero spread) gives 0 or 1 by its side."""
    x = np.asarray(x, float)
    if len(x) < 2:
        return 1.0
    if np.ptp(x) == 0:
        v = x[0]
        return 0.0 if (alternative == "greater" and v > mu) or (alternative == "less" and v < mu) else 1.0
    p = float(stats.ttest_1samp(x, mu, alternative=alternative).pvalue)
    return p if math.isfinite(p) else 1.0


def tost_t(L):
    """Equivalence of the mean log ratio with [log 0.99, log 1.01]: two one-sided t-tests; p = the larger."""
    return max(t_p(L, LO, "greater"), t_p(L, HI, "less"))


def tost_w(L):
    """The same with one-sided Wilcoxon tests (pseudomedian; reported as a sensitivity check only)."""
    return max(wilcoxon_p(np.asarray(L) - LO, "greater"), wilcoxon_p(np.asarray(L) - HI, "less"))


def t_ci(L, level):
    L = np.asarray(L, float)
    n = len(L)
    h = stats.t.ppf(0.5 + level / 2, n - 1) * L.std(ddof=1) / math.sqrt(n)
    return float(L.mean() - h), float(L.mean() + h)


def walsh(x):
    x = np.asarray(x, float)
    i, j = np.triu_indices(len(x))
    return np.sort((x[i] + x[j]) / 2.0)


@lru_cache(None)
def signed_rank_cdf(n):
    """P(T <= t) for the signed-rank statistic with n untied nonzero values, t = 0 .. n(n+1)/2."""
    counts = np.zeros(n * (n + 1) // 2 + 1)
    counts[0] = 1
    for r in range(1, n + 1):
        counts[r:] = counts[r:] + counts[:-r].copy() if r < len(counts) else counts[r:]
    return np.cumsum(counts) / counts.sum()


def hl_ci(x, level=0.95):
    """Hodges-Lehmann pseudomedian and its exact distribution-free confidence interval (Walsh averages; symmetric
    location model)."""
    w = walsh(x)
    n = len(x)
    cdf = signed_rank_cdf(n)
    c = int(np.searchsorted(cdf, (1 - level) / 2, side="right"))      # largest c with P(T <= c - 1) <= alpha / 2
    lo, hi = w[max(c - 1, 0)], w[len(w) - max(c, 1)]
    return float(np.median(w)), float(lo), float(hi)


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps, kind="stable")
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[i]))
        adj[i] = run
    return adj


def pct(v):
    return f"{100 * (math.exp(v) - 1):+.2f}%"


def describe(L):
    lo, hi = t_ci(L, 0.95)
    hl, hlo, hhi = hl_ci(L)
    return (f"{pct(L.mean())} [{pct(lo)}, {pct(hi)}] | {pct(hl)} [{pct(hlo)}, {pct(hhi)}] | "
            f"{int((L < 0).sum())}/{len(L)}")


def seedwise(db, ka, kb, m, seeds):
    """L_s = mean over the 34 cells of log(error_a / error_b), per seed; and the number of missing cell-seed pairs."""
    out, missing = [], 0
    for s in seeds:
        v = []
        for ds, k in CELLS34:
            a, b = db.get(ka(ds, k, s)), db.get(kb(ds, k, s))
            if a is None or b is None:
                missing += 1
                continue
            v.append(math.log(a[m] / b[m]))
        if v:
            out.append(float(np.mean(v)))
    return np.array(out), missing


def verdict_diff(a, L, stated):
    """Direction rule: significant after Holm, and the mean and the pseudomedian on the same side."""
    hl = hl_ci(L)[0]
    if a >= ALPHA:
        return "not significant"
    side = "higher" if L.mean() > 0 and hl > 0 else "lower" if L.mean() < 0 and hl < 0 else None
    if side is None:
        return "direction ambiguous"
    if stated is None:
        return f"significant ({side})"
    return f"confirmed ({side})" if side == stated else f"refuted ({side})"


P = print
problems = []
P("# Third confirmatory round: analysis (pre-registered; confirm3/PROTOCOL.md)")
P()

# ------------------------------------------------------------------ round-2 reference (data hashes, gate)
r2 = {}
for host in ("host_a", "host_b"):
    for f in glob.glob(os.path.join(ROUND2, host, "cfm2_grid", "runs", "cfm2__ngp2_full_medg_pm__*__random__T2.json")):
        r = json.load(open(f))
        r2[r["key"]] = r
R2CODE = defaultdict(set)
for r in r2.values():
    R2CODE[short(r["host"])].add(json.dumps(r.get("code"), sort_keys=True))
for host, ids in R2CODE.items():
    if len(ids) != 1:
        problems.append(f"round 2 on {host}: {len(ids)} code identities")
R2CODE = {h: json.loads(sorted(ids)[0]) for h, ids in R2CODE.items()}


def identity_problems(sub, host, code):
    """The identity must be complete and registered: G and I carry round 2's full identity on this host (the rule's
    modules, kernels, grid.py, datasets.py, the FAISS build at 98deaca, versions) plus the registered round-3 files;
    F carries round 2's shared entries, the e_means module of 79440bc and the registered round-3 files."""
    out, ref = [], R2CODE.get(host)
    if ref is None:
        return [f"{sub} on {host}: no round-2 identity"]
    want = dict(ref) if sub != "r3_prhead" else {**{k: ref[k] for k in SHARED}, **HEAD_EMEANS, "impl": "py"}
    want.update({f: REGISTERED[f] for f in (I_FILES if sub != "r3_prhead" else F_FILES)})
    for k, v in want.items():
        if code.get(k) != v:
            out.append(f"{sub} on {host}: identity entry {k} = {code.get(k)}, expected {v}")
    return out


DATA = {}
for r in r2.values():
    h = (r.get("data_sha"), r.get("data_test_sha"))
    if None in h or DATA.setdefault(r["dataset"], h) != h:
        problems.append(f"round-2 data hashes inconsistent for {r['dataset']}")


# ------------------------------------------------------------------ loading with fail-closed checks
def load(sub, expect):
    """All records of a part, checked: one file per key, key = file name = fields, expected configuration/seed/T,
    host = round-2 placement = fetched directory, data hashes = round 2, finite endpoints, a code identity."""
    db = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "*", sub, "runs", "*.json"))):
        hostdir = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(f))))
        r = json.load(open(f))
        key = r.get("key")
        if key in db:
            problems.append(f"{sub}: duplicate key {key}"); continue
        db[key] = r
        prefix = "ivl" if sub != "r3_prhead" else "prh"
        own = f"{prefix}|{r.get('config')}|{r.get('dataset')}|k{r.get('k')}|s{r.get('seed')}|{r.get('init')}|T{r.get('T')}"
        if key != own or os.path.basename(f) != os.path.basename(C2._run_file("", key)):
            problems.append(f"{sub}: key, file name and fields disagree: {f}"); continue
        if not expect(r):
            problems.append(f"{sub}: unexpected run {key}"); continue
        ds = r["dataset"]
        if short(r.get("host")) != C2.HOST.get(ds) or hostdir != C2.HOST.get(ds):
            problems.append(f"{sub}: {key} ran on {r.get('host')} (directory {hostdir}), placement {C2.HOST.get(ds)}")
        if (r.get("data_sha"), r.get("data_test_sha")) != DATA.get(ds):
            problems.append(f"{sub}: {key} other data than round 2")
        if not all(math.isfinite(float(r.get(m, float("nan")))) and float(r[m]) > 0 for m in ("test_mse", "train_mse")):
            problems.append(f"{sub}: {key} non-finite or non-positive error")
        if not isinstance(r.get("code"), dict) or not r["code"]:
            problems.append(f"{sub}: {key} has no code identity")
    return db


cells_ok = lambda r: (r["dataset"], r["k"]) in set(CELLS34) and r["init"] == "random"
gate = load("r3_gate", lambda r: cells_ok(r) and r["config"] == "medg_pm@5" and r["T"] == 2 and r["seed"] in SEEDS_GATE
            and r.get("impl") == "cpp")
iv = load("r3_interval", lambda r: cells_ok(r) and r["config"] in {f"medg_pm@{c}" for c in INTERVALS} and r["T"] == 2
          and r["seed"] in SEEDS_I and r.get("impl") == "cpp")
ph = load("r3_prhead", lambda r: cells_ok(r) and r["config"] in (NF, FZ, PR)
          and ((r["T"] == 2 and r["seed"] in SEEDS_F2) or (r["T"] == 5 and r["seed"] in SEEDS_F5)))

# ------------------------------------------------------------------ gate G
same = 0
for ds, k in CELLS34:
    for s in SEEDS_GATE:
        g, o = gate.get(ik(5)(ds, k, s)), r2.get(r2k(ds, k, s))
        if g is None or o is None:
            problems.append(f"gate: missing {ik(5)(ds, k, s) if g is None else r2k(ds, k, s)}"); continue
        if g["host"] != o["host"]:
            problems.append(f"gate: {ds} k{k} s{s} host {g['host']} vs round 2 {o['host']}")
        elif g["test_mse"] == o["test_mse"] and g["train_mse"] == o["train_mse"]:
            same += 1
        else:
            problems.append(f"gate: {ds} k{k} s{s} differs from round 2")
P(f"## Gate G\n\n{same} of {2 * len(CELLS34)} gate runs equal the round-2 records (expected {2 * len(CELLS34)}).")
P()

# ------------------------------------------------------------------ provenance: one code identity per part and host
ident = defaultdict(set)
for sub, db in (("r3_gate", gate), ("r3_interval", iv), ("r3_prhead", ph)):
    for r in db.values():
        ident[(sub, short(r["host"]))].add(json.dumps(r.get("code"), sort_keys=True))
for (sub, host), ids in sorted(ident.items()):
    P(f"- provenance {sub} on {host}: {len(ids)} code identity")
    if len(ids) != 1:
        problems.append(f"{sub} on {host}: {len(ids)} code identities")
    else:
        problems += identity_problems(sub, host, json.loads(next(iter(ids))))
for host in ("host_a", "host_b"):
    if ident.get(("r3_gate", host)) and ident.get(("r3_interval", host)) and \
            ident[("r3_gate", host)] != ident[("r3_interval", host)]:
        problems.append(f"gate and Part I ran different code on {host}")
P()
if problems:
    P(f"**CHECKS FAILED ({len(problems)}): no inference.**\n")
    for x in problems[:50]:
        P(f"- {x}")
    sys.exit(1)

fail = False
HEAD = ("| | hypothesis | estimate: mean [95% CI] | pseudomedian [95% CI] | seeds lower | p | Holm p | result |\n"
        "|---|---|---|---|---|---|---|---|")

# ------------------------------------------------------------------ Family I
need = {ik(c)(ds, k, s) for c in INTERVALS for ds, k in CELLS34 for s in SEEDS_I}
miss_i = len(need - set(iv))
P(f"## Family I (check interval; 34 cells x seeds 200-299, 2 epochs; {len(need) - miss_i} of {len(need)} runs)")
P()
if miss_i:
    P(f"**Part I incomplete ({miss_i} runs missing): no tests.**")
    fail = True
else:
    rows = []      # (id, name, L, kind, stated, p, extra)
    n_id = 0
    for c in (10, 20):
        for m, name in (("test_mse", "test"), ("train_mse", "train")):
            L, _ = seedwise(iv, ik(c), ik(5), m, SEEDS_I)
            n_id += 1
            rows.append((f"I{n_id}", f"every {c} vs every 5, {name}: higher", L, "diff", "higher", wilcoxon_p(L), ""))
    for c in (1, 2):
        for m, name in (("test_mse", "test"), ("train_mse", "train")):
            L, _ = seedwise(iv, ik(c), ik(5), m, SEEDS_I)
            n_id += 1
            lo90, hi90 = t_ci(L, 0.90)
            rows.append((f"I{n_id}", f"every {c} vs every 5, {name}: equivalent within 1%", L, "equiv", None, tost_t(L),
                         f"unadjusted 90% CI of the mean {pct(lo90)} to {pct(hi90)} (the decision uses the Holm-adjusted p); "
                         f"Wilcoxon TOST p {tost_w(L):.2g} (sensitivity)"))
    for c in (1, 2):
        for m, name in (("test_mse", "test"), ("train_mse", "train")):
            L, _ = seedwise(iv, ik(c), ik(5), m, SEEDS_I)
            n_id += 1
            rows.append((f"I{n_id}", f"every {c} vs every 5, {name}: different", L, "diff", None, wilcoxon_p(L), ""))
    adj = holm([r[5] for r in rows])
    P(HEAD)
    for (hid, name, L, kind, stated, p, extra), a in zip(rows, adj):
        res = ("equivalent within 1%" if a < ALPHA else "equivalence not shown") if kind == "equiv" else \
            verdict_diff(a, L, stated)
        P(f"| {hid} | {name} | {describe(L)} | {p:.2g} | {a:.2g} | {res} |")
    P()
    for hid, name, L, kind, stated, p, extra in rows:
        if extra:
            P(f"- {hid}: {extra}")
    P()
    P("Descriptive (paired over all 3,400 cell-seed runs of each interval; medians):")
    P()
    P("| every | time / every 5 (median ratio) | checks | splits | runs with > k/4 empty centres |\n|---|---|---|---|---|")
    allk = [(ds, k, s) for ds, k in CELLS34 for s in SEEDS_I]
    for c in INTERVALS:
        tr = np.median([iv[ik(c)(ds, k, s)]["seconds"] / iv[ik(5)(ds, k, s)]["seconds"] for ds, k, s in allk])
        P(f"| {c} | {tr:.3f} | {np.median([iv[ik(c)(*x)]['n_checks'] for x in allk]):.0f} | "
          f"{np.median([iv[ik(c)(*x)]['n_splits'] for x in allk]):.0f} | "
          f"{sum(iv[ik(c)(ds, k, s)]['empty'] > k / 4 for ds, k, s in allk)} |")
    P()
    P("Per dataset (geometric-mean ratio to every 5 batches over cells and seeds, test / train):")
    P()
    P("| dataset | " + " | ".join(f"every {c}" for c in (1, 2, 10, 20)) + " |\n|---|---|---|---|---|")
    for dsn in sorted({ds for ds, _ in CELLS34}):
        cl = [(ds, k, s) for ds, k in CELLS34 if ds == dsn for s in SEEDS_I]
        cells = []
        for c in (1, 2, 10, 20):
            vt = [math.log(iv[ik(c)(*x)]["test_mse"] / iv[ik(5)(*x)]["test_mse"]) for x in cl]
            vr = [math.log(iv[ik(c)(*x)]["train_mse"] / iv[ik(5)(*x)]["train_mse"]) for x in cl]
            cells.append(f"{pct(np.mean(vt))} / {pct(np.mean(vr))}")
        P(f"| {dsn} | " + " | ".join(cells) + " |")
    P()

# ------------------------------------------------------------------ Family F
need2 = {fk(c, 2)(ds, k, s) for c in (NF, FZ, PR) for ds, k in CELLS34 for s in SEEDS_F2}
need5 = {fk(c, 5)(ds, k, s) for c in (NF, FZ, PR) for ds, k in CELLS34 for s in SEEDS_F5}
miss2, miss5 = len(need2 - set(ph)), len(need5 - set(ph))
P(f"## Family F (newer driver 79440bc; F2: seeds 200-259, 2 epochs, {len(need2) - miss2} of {len(need2)} runs; "
  f"F5: seeds 200-239, 5 epochs, {len(need5) - miss5} of {len(need5)} runs)")
P()
if miss2 or miss5:
    P(f"**Part F2 or F5 incomplete ({miss2} and {miss5} runs missing): no tests.**")
    fail = True
else:
    rows = []
    for T, seeds in ((2, SEEDS_F2),):
        for other, oname in ((FZ, "with the freeze"), (PR, "the PR's rule")):
            for m, name in (("test_mse", "test"), ("train_mse", "train")):
                L, _ = seedwise(ph, fk(NF, T), fk(other, T), m, seeds)
                rows.append((f"T{T}: without the freeze vs {oname}, {name}: lower", L, "diff", "lower", wilcoxon_p(L), ""))
    for m, name in (("test_mse", "test"), ("train_mse", "train")):
        L, _ = seedwise(ph, fk(NF, 5), fk(FZ, 5), m, SEEDS_F5)
        lo90, hi90 = t_ci(L, 0.90)
        rows.append((f"T5: without the freeze vs with the freeze, {name}: equivalent within 1%", L, "equiv", None,
                     tost_t(L), f"unadjusted 90% CI of the mean {pct(lo90)} to {pct(hi90)} (the decision uses the Holm-adjusted p); "
                         f"Wilcoxon TOST p {tost_w(L):.2g} (sensitivity)"))
    for m, name in (("test_mse", "test"), ("train_mse", "train")):
        L, _ = seedwise(ph, fk(NF, 5), fk(PR, 5), m, SEEDS_F5)
        rows.append((f"T5: without the freeze vs the PR's rule, {name}: lower", L, "diff", "lower", wilcoxon_p(L), ""))
    adj = holm([r[4] for r in rows])
    P(HEAD)
    for i, ((name, L, kind, stated, p, extra), a) in enumerate(zip(rows, adj)):
        res = ("equivalent within 1%" if a < ALPHA else "equivalence not shown") if kind == "equiv" else \
            verdict_diff(a, L, stated)
        P(f"| F{i + 1} | {name} | {describe(L)} | {p:.2g} | {a:.2g} | {res} |")
    P()
    for i, (name, L, kind, stated, p, extra) in enumerate(rows):
        if extra:
            P(f"- F{i + 1}: {extra}")
    P()
    for T, seeds in ((2, SEEDS_F2), (5, SEEDS_F5)):
        allk = [(ds, k, s) for ds, k in CELLS34 for s in seeds]
        P(f"Descriptive, {T} epochs:")
        P()
        for other, oname in ((FZ, "with the freeze"), (PR, "the PR's rule")):
            tr = np.median([ph[fk(NF, T)(*x)]["seconds"] / ph[fk(other, T)(*x)]["seconds"] for x in allk])
            P(f"- running time without the freeze / {oname}: median ratio {tr:.3f}")
        if T == 5:
            for m, name in (("test_mse", "test"), ("train_mse", "train")):
                L, _ = seedwise(ph, fk(NF, 5), fk(FZ, 5), m, SEEDS_F5)
                P(f"- without vs with the freeze, {name}, two-sided Wilcoxon p (unadjusted, descriptive): {wilcoxon_p(L):.2g}")
        mv = sum(ph[fk(NF, T)(*x)].get("n_moves_last_pass", 0) for x in allk)
        sp = sum(ph[fk(NF, T)(*x)]["n_splits"] for x in allk)
        P(f"- moves in the last pass without the freeze: {mv} of {sp} splits")
        emp = {c: sum(ph[fk(c, T)(ds, k, s)]["empty"] > k / 4 for ds, k, s in allk) for c in (NF, FZ, PR)}
        P(f"- runs with more than a quarter of the centres empty: {emp}")
        P()
        P("| dataset | without / with the freeze (test / train) | without the freeze / PR's rule (test / train) |\n"
          "|---|---|---|")
        for dsn in sorted({ds for ds, _ in CELLS34}):
            cl = [(ds, k, s) for ds, k in CELLS34 if ds == dsn for s in seeds]
            cells = []
            for other in (FZ, PR):
                vt = [math.log(ph[fk(NF, T)(*x)]["test_mse"] / ph[fk(other, T)(*x)]["test_mse"]) for x in cl]
                vr = [math.log(ph[fk(NF, T)(*x)]["train_mse"] / ph[fk(other, T)(*x)]["train_mse"]) for x in cl]
                cells.append(f"{pct(np.mean(vt))} / {pct(np.mean(vr))}")
            P(f"| {dsn} | " + " | ".join(cells) + " |")
        P()
sys.exit(1 if fail else 0)
