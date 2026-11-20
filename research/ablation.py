"""Component ablation of FAISS PR #5915 (cost-based revival) and hybrids with the distance test.

Runs on the PR build (env emeans_pr5915, head 98deaca). One subclass of the PR's NumpyEmeans with four switches:
  removal: "mass"       size test N_j < 0.1 mean(N)                         (library)
           "mass_dist"  size test, then the distance test (ours)
           "cost"       R_j <= 0 or R_j < 0.3 median(R), cheapest first, pair guard   (PR)
           "cost_dist"  cost candidates with R_j > 0 must also pass the distance test (hybrid 2.1)
  donor:   "size"       Gumbel draw proportional to N (library)    | "gain"  largest |U_l| first (PR)
  split:   "copy"       c (1 -/+ 1e-4 sign)  (FAISS 1.16)           | "axis"  c -/+ sqrt(2|U|/(pi N)) u (PR)
  timing:  "lib"        checks every 5 batches to the end, last-pass means (library)
           "pr"         no moves unless a full pass follows (except the first pass); last-pass splits keep
                        their running positions (PR)
Distance test (Small Is Not Dead): remove candidate j only if delta_j <= tau * MSE_hat or j had no point for W epochs.
Everything else is the PR's code. Component timers are accumulated in TIMES (seconds, per process).
"""
import time
from collections import defaultdict

import numpy as np
import faiss
from faiss.contrib.e_means import NumpyEmeans
from faiss.contrib.e_means.e_means import _numpy_unused_threshold, _separate_splits

SPLIT_EPSILON = 1e-4          # FAISS 1.16 _SPLIT_EPSILON (Appendix J, Eq. 8)
REMOVAL_THRESHOLD = 0.3       # PR _REMOVAL_THRESHOLD
TIMES = defaultdict(float)


def _timed(name, fn):
    def wrapper(*a, **kw):
        t0 = time.perf_counter()
        try:
            return fn(*a, **kw)
        finally:
            TIMES[name] += time.perf_counter() - t0
    return wrapper


for _name in ("accumulate_removal_stats", "update_removal_axes", "accumulate_cluster_sums"):
    if not hasattr(getattr(faiss, _name), "_timed"):
        _w = _timed(_name, getattr(faiss, _name)); _w._timed = True
        setattr(faiss, _name, _w)

CONFIGS = {
    # attribution: from the library to the PR, one component at a time
    "lib":              ("mass", "size", "copy", "lib"),
    "lib_axis":         ("mass", "size", "axis", "lib"),     # = the PR with unused_threshold=0.1
    "cost":             ("cost", "size", "copy", "lib"),
    "cost_gain":        ("cost", "gain", "copy", "lib"),
    "cost_gain_axis":   ("cost", "gain", "axis", "lib"),     # = the PR without its timing rules
    "pr":               ("cost", "gain", "axis", "pr"),      # = the PR default
    # leave one out of the PR
    "pr_no_gain":       ("cost", "size", "axis", "pr"),
    "pr_no_axis":       ("cost", "gain", "copy", "pr"),
    "pr_no_cost":       ("mass", "gain", "axis", "pr"),
    # ours and hybrids
    "ours":             ("mass_dist", "size", "copy", "lib"),  # = Small Is Not Dead on FAISS 1.16
    "h21":              ("cost_dist", "gain", "axis", "pr"),   # 2.1: PR + distance test as a safeguard
    "h22_gain":         ("mass_dist", "gain", "copy", "lib"),  # 2.2: ours + PR donors
    "h22_gain_axis":    ("mass_dist", "gain", "axis", "lib"),  # 2.2: ours + PR donors + PR split
    "h22_gain_axis_pr": ("mass_dist", "gain", "axis", "pr"),   # 2.2: ... + PR timing
    "lib_gain_axis":    ("mass", "gain", "axis", "lib"),       # control: PR relocation, no distance test
}
# Net-gain check (2026-11-18): free j to split donor d only if cost_j < w * gain_d, gain_d = (2/pi) |U_d|.
#   ng_est:   cost_j = R_j (the PR's estimate of the removal cost; needs the second-nearest search)
#   ng_bound: cost_j = N_j delta_j (the bound of Small Is Not Dead; no second-nearest search)
# PR donors (largest gain first), axis split, pair guard and timing rules are kept.
for _w in (1.0, 1.25, 1.5, 1.75, 2.0):
    CONFIGS[f"ng_est_w{_w:g}"] = ("ng_est", "gain", "axis", "pr", _w)
    CONFIGS[f"ng_bound_w{_w:g}"] = ("ng_bound", "gain", "axis", "pr", _w)
# Cool-down for the net-gain check (bound, w = 1): a centre involved in a split (donor or child) can be neither freed
# nor used as a donor while cooling. "check": skip the next check (10 batches); "half"/"quarter": until the inherited
# weight prod(1 - alpha_t) since the split is <= 1/2 or <= 1/4 (an EMA half-life, so it lengthens as alpha decays).
for _cd in ("none", "check", "half", "quarter"):
    CONFIGS[f"ngc_{_cd}"] = ("ng_bound", "gain", "axis", "pr", {"w": 1.0, "cooldown": _cd})
# Lower w for the candidate rule (bound + half-life cool-down): w < 1 discounts the predicted split gain
for _w in (0.25, 0.5, 0.75, 0.9):
    CONFIGS[f"ngc_half_w{_w:g}"] = ("ng_bound", "gain", "axis", "pr", {"w": _w, "cooldown": "half"})
# Posterior (risk-neutral) net-gain check, 2026-11-18: compare posterior means of cost and gain.
#   cost_j = E[lambda_j | counts since birth] * delta_j   (Gamma-Poisson; prior fitted to all centres by moments)
#   gain_d = E[G_d | G_hat_d] with G_hat_d = (2/pi)|U_d| / (1 - prod(1 - alpha))   (EMA start-up bias removed)
#            (log-normal empirical Bayes; relative variance 2 / n_eff, n_eff = points in the EMA's memory)
# Same matching, pair guard, axis split, timing rules and half-life cool-down as ngc_half.
CONFIGS["ngp"] = ("ng_post", "gain", "axis", "pr", {"w": 1.0, "cooldown": "half", "matcher": "v2"})
# Same rule with an approximate nearest-other-centre search for delta (overhead test): IVF over the centres with
# round(sqrt(k)) coarse lists whose centroids are centres sampled at each check (no training), probing 1/8 of the lists.
# An approximate search can only return a farther neighbour, so delta and the cost can only be overestimated.
CONFIGS["ngp_ivf"] = ("ng_post", "gain", "axis", "pr", {"w": 1.0, "cooldown": "half", "matcher": "v2", "knn": "ivf"})
# Posterior net-gain with a calibrated removal cost and a data-driven gain noise (2026-11-18, "ngp2"):
#   cost_j = E[lambda_j] * r_hat_j * delta_j, where r_j = (mean sampled margin d2 - d1 of j's rows since birth) / delta_j
#            is shrunk toward the population ratio rho (normal empirical Bayes, pooled within-cluster variance);
#            margins come from a two-nearest search that replaces the usual search on 1 row in margin_stride (8);
#   gain noise: relative variance (kappa - 1) / n_eff with kappa the pooled kurtosis of the projections on the split
#            axis (sampled rows), instead of the Gaussian 2 / n_eff.
CONFIGS["ngp2"] = ("ng_post2", "gain", "axis", "pr", {"w": 1.0, "cooldown": "half", "matcher": "v2", "margin_stride": 8})
CONFIGS["ngp2_ivf"] = ("ng_post2", "gain", "axis", "pr",
                       {"w": 1.0, "cooldown": "half", "matcher": "v2", "margin_stride": 8, "knn": "ivf"})
# option 3 candidates (2026-11-19): ngp2 with the two-nearest search on every row (no sampling parameter), and with the
# no-fallback-donor rule
CONFIGS["ngp2_full"] = ("ng_post2", "gain", "axis", "pr", {"w": 1.0, "cooldown": "half", "matcher": "v2", "margin_stride": 1})
# confirmatory study (2026-11-19, protocol_ngp.md): the frozen rule is ngp2_full (margin_stride 1); sampling-rate
# ablation = the same rule with the two-nearest search on 1 row in s (s = 8 is ngp2)
for _s in (2, 4, 16, 32):
    CONFIGS[f"ngp2_s{_s}"] = ("ng_post2", "gain", "axis", "pr",
                             {"w": 1.0, "cooldown": "half", "matcher": "v2", "margin_stride": _s})
CONFIGS["ngp2_full_nf"] = ("ng_post2", "gain", "axis", "pr",
                           {"w": 1.0, "cooldown": "half", "matcher": "v2", "margin_stride": 1, "no_fallback_donor": True})
# Base rule after the ngp3 experiments (2026-11-19), "ngp4" = ngp2 with: the two-nearest search on every row while moves
# are allowed (no sampling parameter), no fallback donor, and Smyth-moderated variances on the margin ratio r = m/delta.
# Gain as in ngp2 (the PR's predicted Gaussian gain, start-up corrected, log-scale EB with noise (kappa-1)/n_eff).
_NGP4 = {"w": 1.0, "cooldown": "half", "matcher": "v2", "margin_stride": 1, "no_fallback_donor": True, "moderate": True,
         "gain_noise": "compound"}
# gain_noise="compound" (Codex review of note rev 2): the predicted gain is a sum over a Poisson number of points, so its
# relative variance is kappa / n_eff (compound Poisson), not (kappa - 1) / n_eff as in ngp2 (kept there for identity).
CONFIGS["ngp4"] = ("ng_post2", "gain", "axis", "pr", dict(_NGP4))
# attribution of ngp4's KDD regression (2026-11-19): ngp2 on every row plus one ngp4 change at a time
CONFIGS["ngp2_full_cmp"] = ("ng_post2", "gain", "axis", "pr", {"w": 1.0, "cooldown": "half", "matcher": "v2",
                                                               "margin_stride": 1, "gain_noise": "compound"})
CONFIGS["ngp2_full_mod"] = ("ng_post2", "gain", "axis", "pr", {"w": 1.0, "cooldown": "half", "matcher": "v2",
                                                               "margin_stride": 1, "moderate": True})
# "Correct the estimates" (2026-11-19), on the parameter-free base ngp2_full:
#   ngp5m : gain noise MEASURED from the batch-to-batch variation of the gain's increments (b_t = the batch's scatter
#           along the split axis): relative variance = CV^2(b) / effective number of batches of the EMA, per cluster,
#           Smyth-moderated across clusters; restarts at a split (a newborn's gain is inherited, not measured).
#   ngp5t : Tweedie's formula for the log gains (nonparametric marginal by binned KDE) with the base noise.
#   ngp5mt: both.
_NGP5 = {"w": 1.0, "cooldown": "half", "matcher": "v2", "margin_stride": 1}
CONFIGS["ngp5m"] = ("ng_post2", "gain", "axis", "pr", dict(_NGP5, gain_noise="measured"))
CONFIGS["ngp5t"] = ("ng_post2", "gain", "axis", "pr", dict(_NGP5, gain_eb="tweedie"))
CONFIGS["ngp5mt"] = ("ng_post2", "gain", "axis", "pr", dict(_NGP5, gain_noise="measured", gain_eb="tweedie"))
# Event-triggered revival checks (2026-11-19), on ngp2_full: "every" = a full check after every batch (reference);
# "event" = after every batch an O(k) bound max_d G~_d - min_j E[lambda_j] r~_j delta_LB_j, full check only if > 0;
# delta_LB from the last exact delta and the centres' displacements (triangle inequality), split centres exact.
CONFIGS["ngp2b"] = ("ng_post2", "gain", "axis", "pr", dict(_NGP5, trigger="every"))
CONFIGS["ngp2e"] = ("ng_post2", "gain", "axis", "pr", dict(_NGP5, trigger="event"))
# Adaptive confirmation wait (2026-11-19, quick test): propose-then-confirm on ngp2_full, but a pair is decided only once
# its donor has received as many fresh points as the proposal's estimate rested on (posterior rate x EMA memory
# (2 - alpha)/alpha batches), or at the last check before moves freeze; unready pairs stay pending (centres reserved),
# new proposals continue from the other centres.
CONFIGS["ngp2cw"] = ("ng_post2", "gain", "axis", "pr", dict(_NGP5, confirm=True, confirm_wait="evidence"))
CONFIGS["ngp4_ivf"] = ("ng_post2", "gain", "axis", "pr", dict(_NGP4, knn="ivf"))
# Propose, then confirm (2026-11-19): moves chosen by ngp4 at a check are executed only at the NEXT check, and only if
# the combined move measured on the points that arrived in between pays: gain = exact reduction on the donor's fresh
# points (each choosing between the two children, as placed now, and its fallback unless that is the candidate);
# cost = posterior rate * per-point change measured on the candidate's fresh points (fallback or children), as a
# ratio to delta shrunk toward the population ratio (isolation prior when the candidate gets few or no fresh points).
CONFIGS["ngp4c"] = ("ng_post2", "gain", "axis", "pr", dict(_NGP4, confirm=True))
# Option 1 (2026-11-19): confirm on the value AFTER ONE LOCAL LLOYD STEP, measured on the fresh points: each child
# recentres on the fresh points it captures (+ n |m - pos|^2) and each fallback centre l absorbing n moved points
# recentres (+ n^2/(N_l + n) |m - c_l|^2, Ward's correction), each term debiased for the in-sample fit of the mean
# (minus its expected value under no shift: trace(cov) resp. n/(N_l + n) trace(cov)); groups with < 2 points: no term.
CONFIGS["ngp4ca"] = ("ng_post2", "gain", "axis", "pr", dict(_NGP4, confirm=True, confirm_adapt=True))
# The simplified posterior net-gain rule of the working note (arxiv/netgain_note/note.tex), "ngp3": while moves are
# allowed, the two-nearest search runs on every row (the PR's own path); per centre since birth: N, sums of margins,
# squared margins, |p| and p^2 (p = projection on the centre's split axis).
#   cost_j = E[lambda_j] * r~_j * delta_j,  r_j = mean margin / delta_j            (eq. 7 of the note)
#   gain_d = E[lambda_d] * s_d^2 * (2 q~_d - 1),  q_d = mean |p| / s_d              (eq. 8; s_d = the split offset used)
# r and q shrunk by normal-normal empirical Bayes with Smyth-moderated variances; rates Gamma-Poisson.
# Corrected after the Codex review of the note and the first smoke tests (2026-11-19): the per-point quantities (mean
# margin, mean |p| and their variances) use the algorithm's own EMA memory (step alpha_t, reset at splits), with the
# EMA's effective number of points; rates keep the since-birth Gamma-Poisson posterior; variances are moderated on the
# dimensionless ratios r and q; a donor may not be the freed centre's nearest other centre (its main fallback).
_NGP3 = {"w": 1.0, "cooldown": "half", "matcher": "v2", "memory": "ema", "no_fallback_donor": True}
CONFIGS["ngp3"] = ("ng_post3", "gain", "axis", "pr", dict(_NGP3))
CONFIGS["ngp3_ivf"] = ("ng_post3", "gain", "axis", "pr", dict(_NGP3, knn="ivf"))
CONFIGS["ngp3b"] = ("ng_post3", "gain", "axis", "pr", dict(_NGP3, memory="birth"))      # ablation: since-birth memory
# Option 1 (2026-11-19): every rate is a Gamma-Poisson posterior on DISCOUNTED evidence with the algorithm's own
# forgetting factor (1 - alpha_t): N' <- (1 - alpha) N' + n, T' <- (1 - alpha) T' + 1; at a split donor and child each
# get half of the donor's N' with its exposure T' (as e-means halves its count). Used on both sides of the decision.
CONFIGS["ngp3f"] = ("ng_post3", "gain", "axis", "pr", dict(_NGP3, rate_mode="discount"))
# diagnostic (2026-11-19): donor gains use the current EMA rate (usage / (1 - prod(1 - alpha))) instead of since birth
CONFIGS["ngp3_gr"] = ("ng_post3", "gain", "axis", "pr", dict(_NGP3, gain_rate="ema"))
# ngc_half with the corrected matcher (Codex review round 1): isolates the matcher fix from the posterior rule
CONFIGS["ngc_half_m2"] = ("ng_bound", "gain", "axis", "pr", {"w": 1.0, "cooldown": "half", "matcher": "v2"})


class AblationEmeans(NumpyEmeans):
    def __init__(self, d, k, removal, donor, split, timing, w=1.0, cooldown="none", matcher="v1", knn="exact",
                 margin_stride=8, memory="birth", no_fallback_donor=False, gain_rate="birth", rate_mode="birth",
                 moderate=False, gain_noise="fixed_n", confirm=False, confirm_adapt=False, gain_eb="normal",
                 trigger="periodic", confirm_wait="next", tau_near=4.0, abandon_passes=2.0, **kw):
        assert removal in ("mass", "mass_dist", "cost", "cost_dist", "ng_est", "ng_bound", "ng_post", "ng_post2",
                           "ng_post3")
        self.margin_stride = int(margin_stride)
        assert memory in ("birth", "ema")
        self.memory = memory        # ng_post3: memory of the per-point statistics
        self.no_fallback_donor = bool(no_fallback_donor)   # forbid pairing j with its nearest other centre as donor
        assert gain_rate in ("birth", "ema") and rate_mode in ("birth", "discount")
        self.gain_rate = gain_rate
        self.rate_mode = rate_mode
        self.moderate = bool(moderate)     # ng_post2: Smyth-moderated per-cluster variances of r (else pooled)
        assert gain_noise in ("fixed_n", "compound", "measured") and gain_eb in ("normal", "tweedie")
        self.gain_eb = gain_eb
        assert trigger in ("periodic", "every", "event") and confirm_wait in ("next", "evidence")
        self.confirm_wait = confirm_wait
        self.trigger = trigger
        self.n_full_checks = self.n_bound_evals = 0
        self.gain_noise = gain_noise       # relative variance of the gain: (kappa-1)/n (fixed_n) or kappa/n (compound)
        self.confirm = bool(confirm)       # propose at a check, confirm on fresh points at the next check
        self.confirm_adapt = bool(confirm_adapt)   # confirm on the value after one local Lloyd step
        self.n_proposed = self.n_confirmed = 0
        self.confirm_log = []
        assert donor in ("size", "gain") and split in ("copy", "axis") and timing in ("lib", "pr")
        assert cooldown in ("none", "check", "half", "quarter") and matcher in ("v1", "v2")
        self.matcher = matcher
        assert knn in ("exact", "ivf")
        self.knn = knn          # nearest-other-centre search for delta (net-gain rules); "ivf" = approximate
        self.removal, self.donor, self.split, self.timing, self.w = removal, donor, split, timing, float(w)
        self.cooldown = cooldown
        self.n_quick_target = self.n_quick_donor = 0     # moves involving a centre with mostly inherited statistics
        self.tau_near, self.abandon_passes = float(tau_near), abandon_passes
        needs_r = removal.startswith("cost") or removal in ("ng_est", "ng_post3")   # second-nearest search
        super().__init__(d, k, unused_threshold=None if needs_r else 0.1, **kw)
        self.n_candidates = self.n_protected = self.n_abandoned = 0

    # ------------------------------------------------------------------ gates
    def _keeps_removal_stats(self):
        return (self.removal.startswith("cost") or self.removal.startswith("ng") or self.donor == "gain"
                or self.split == "axis")

    def _frozen(self):
        if self.timing != "pr":
            return False
        rows = self.recipe["epoch_rows"]
        done = sum(rows[:self._epoch]) + min(rows[self._epoch], self._epoch_rows + self.recipe["batch_size"])
        return len(rows) > 1 and done >= self._n and sum(rows) - done < self._n

    def begin(self, global_n, init_centroids):
        super().begin(global_n, init_centroids)
        self._moved = np.zeros(self.k, dtype=bool) if self.timing == "pr" else None
        self._collect = self.removal.endswith("_dist")
        self._dist = np.zeros(self.k, dtype=np.float64)       # EMA of per-cluster distortion D_j
        self._batch_dist = np.zeros(self.k, dtype=np.float64)
        self._last_hit = np.zeros(self.k, dtype=np.int64)
        self._stale = np.zeros(self.k, dtype=np.float64)          # inherited weight since the last split
        self._split_step = np.full(self.k, -10**9, dtype=np.int64)
        self._decay_prod = 1.0                                     # prod(1 - alpha_t) since the start (EMA bias)
        self._cum_n = np.zeros(self.k, dtype=np.float64)          # points assigned since birth (decisions only)
        self._age = np.zeros(self.k, dtype=np.float64)            # batches since birth
        self._dn = np.zeros(self.k, dtype=np.float64)             # discounted counts N' (rate_mode="discount")
        self._dt = np.zeros(self.k, dtype=np.float64)             # discounted exposure T'
        self._dw2 = np.zeros(self.k, dtype=np.float64)            # discounted sum of squared weights (variance)
        self._measure_all = self.removal == "ng_post3"
        if self._measure_all:         # sums over all rows assigned since birth, while moves are allowed
            self._m_n = np.zeros(self.k, dtype=np.float64)        # rows measured
            self._m_sum = np.zeros(self.k, dtype=np.float64)      # sum of margins d2 - d1
            self._m_sq = np.zeros(self.k, dtype=np.float64)       # sum of squared margins
            self._pabs = np.zeros(self.k, dtype=np.float64)       # sum of |p|
            self._p2 = np.zeros(self.k, dtype=np.float64)         # sum of p^2
            self._ema_batch = np.zeros((5, self.k), dtype=np.float64)   # this batch: n, sum m, sum m^2, sum |p|, sum p^2
            self._ema = np.zeros((5, self.k), dtype=np.float64)         # EMAs of the five (step alpha_t)
            self._ema_w2 = np.zeros(self.k, dtype=np.float64)           # EMA weights squared (effective sample size)
        self._pend = None             # pending proposals (confirm=True)
        self._inc = np.zeros((4, self.k), dtype=np.float64)   # measured gain noise: EMA of b, of b^2, sum w, sum w^2
        self._delta_ref = None        # event trigger: exact delta and positions at the last full check
        self._pos_ref = None
        self._new_since = set()       # centres moved by splits since then
        self._r_ref = None            # ratio estimates at the last full check
        self._sample_margins = self.removal == "ng_post2"
        if self._sample_margins:      # sums over the sampled rows of each cluster since its birth
            self._m_n = np.zeros(self.k, dtype=np.float64)        # sampled rows
            self._m_sum = np.zeros(self.k, dtype=np.float64)      # sum of margins d2 - d1
            self._m_sq = np.zeros(self.k, dtype=np.float64)       # sum of squared margins
            self._p2 = np.zeros(self.k, dtype=np.float64)         # sum of p^2, p = projection on the split axis
            self._p4 = np.zeros(self.k, dtype=np.float64)         # sum of p^4
            self._margin_rng = np.random.RandomState((self.seed * 7919 + 17) % (2 ** 31))

    # ------------------------------------------------------------------ statistics
    def _assign_rows(self, x, centroids, second_out=None):
        if (getattr(self, "_sample_margins", False) and second_out is None and self._centroids is not None
                and len(x) > 0 and len(centroids) >= 2 and not self._frozen()):
            distance, labels = self._assign_rows_sampled(x, centroids)
        else:
            t0 = time.perf_counter()
            distance, labels = super()._assign_rows(x, centroids, second_out)
            TIMES["search_top2" if second_out is not None else "search_top1"] += time.perf_counter() - t0
            if getattr(self, "_measure_all", False) and second_out is not None and len(x) > 0:
                self._measure_rows(x, centroids, distance, labels, second_out)
        if self._collect:
            t0 = time.perf_counter()
            self._batch_dist += np.bincount(labels, weights=np.maximum(distance, 0), minlength=self.k)
            TIMES["dist_stats"] += time.perf_counter() - t0
        return distance, labels

    def _assign_rows_sampled(self, x, centroids):
        """Nearest-centre assignment where 1 row in margin_stride gets a two-nearest search instead: its margin
        d2 - d1 and its projection on its cluster's split axis are accumulated (decisions only). Labels follow the
        parent's tie rule (ties keep the lowest label)."""
        n = len(x)
        off = int(self._margin_rng.randint(self.margin_stride))
        smask = np.zeros(n, dtype=bool)
        smask[off::self.margin_stride] = True
        sidx, ridx = np.flatnonzero(smask), np.flatnonzero(~smask)
        distance = np.empty(n, dtype=np.float32)
        labels = np.empty(n, dtype=np.int64)
        t0 = time.perf_counter()
        if len(ridx):
            d1, l1 = NumpyEmeans._assign_rows(self, np.ascontiguousarray(x[ridx]), centroids)
            distance[ridx], labels[ridx] = d1, l1
        TIMES["search_top1"] += time.perf_counter() - t0
        if len(sidx) == 0:
            return distance, labels
        t0 = time.perf_counter()
        xs = np.ascontiguousarray(x[sidx])
        D, I = faiss.knn(xs, centroids, 2)
        lab = np.where(D[:, 0] == D[:, 1], I.min(axis=1), I[:, 0])
        distance[sidx], labels[sidx] = D[:, 0], lab
        TIMES["search_sampled_top2"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        other = np.where(I[:, 0] == lab, D[:, 1], D[:, 0]).astype(np.float64)
        margin = np.maximum(other - D[:, 0].astype(np.float64), 0.0)
        margin = np.where(np.isfinite(margin), margin, 0.0)
        if self._pend is not None:
            second = np.where(I[:, 0] == lab, I[:, 1], I[:, 0])
            self._accumulate_pending(xs, lab, D[:, 0].astype(np.float64), other, second)
        k = self.k
        self._m_n += np.bincount(lab, minlength=k)
        self._m_sum += np.bincount(lab, weights=margin, minlength=k)
        self._m_sq += np.bincount(lab, weights=margin * margin, minlength=k)
        y = xs.astype(np.float64) - centroids[lab].astype(np.float64)
        pr = np.einsum("ij,ij->i", y, self._axes[lab].astype(np.float64))
        self._p2 += np.bincount(lab, weights=pr * pr, minlength=k)
        self._p4 += np.bincount(lab, weights=pr ** 4, minlength=k)
        TIMES["margin_stats"] += time.perf_counter() - t0
        return distance, labels

    def _measure_rows(self, x, centroids, distance, labels, second):
        """ng_post3: accumulate, per assigned centre, margins d2 - d1 and |p|, p^2 on its split axis (all rows)."""
        t0 = time.perf_counter()
        k = self.k
        margin = np.asarray(second, dtype=np.float64) - np.asarray(distance, dtype=np.float64)
        margin = np.where(np.isfinite(margin), np.maximum(margin, 0.0), 0.0)
        cnt = np.bincount(labels, minlength=k).astype(np.float64)
        sm = np.bincount(labels, weights=margin, minlength=k)
        sq = np.bincount(labels, weights=margin * margin, minlength=k)
        y = np.asarray(x, dtype=np.float64) - centroids[labels].astype(np.float64)
        pr = np.einsum("ij,ij->i", y, self._axes[labels].astype(np.float64))
        sa = np.bincount(labels, weights=np.abs(pr), minlength=k)
        s2 = np.bincount(labels, weights=pr * pr, minlength=k)
        self._m_n += cnt; self._m_sum += sm; self._m_sq += sq; self._pabs += sa; self._p2 += s2
        self._ema_batch += np.stack([cnt, sm, sq, sa, s2])
        TIMES["margin_stats"] += time.perf_counter() - t0

    def _accumulate_pending(self, xs, lab, d1, other, second):
        """Fresh evidence for pending pairs: for each point of a pending candidate j or donor d, its exact error after
        the combined move (min over its fallback -- unless the fallback is the pair's other centre -- and the two
        children of d placed as the split would place them now) against its current error d1."""
        P = self._pend
        pidx = P["pair_of"][lab]
        sel = np.flatnonzero(pidx >= 0)
        if len(sel) == 0:
            return
        pi, role = pidx[sel], P["role_of"][lab[sel]]                  # role 0: candidate j, 1: donor d
        dc, cc = self._split_positions(P["d"])                        # (P, dim) each, current positions
        x = xs[sel].astype(np.float64)
        e1 = ((x - dc[pi].astype(np.float64)) ** 2).sum(1)
        e2 = ((x - cc[pi].astype(np.float64)) ** 2).sum(1)
        partner = np.where(role == 0, P["d"][pi], P["j"][pi])
        fb = np.where(second[sel] == partner, np.inf, other[sel])     # fallback lost if it is the partner
        after = np.minimum(np.minimum(e1, e2), fb)
        change = after - d1[sel]                                      # > 0: error increases
        if self.confirm_adapt:
            which = np.where(e1 <= np.minimum(e2, fb), 0, np.where(e2 <= fb, 1, 2))   # donor-child, child, fallback
            sq = (x * x).sum(1)
            for c in (0, 1):                                           # children: count, sum, sum of squared norms
                m = which == c
                if m.any():
                    np.add.at(P["ch_n"][:, c], pi[m], 1.0)
                    np.add.at(P["ch_sum"][:, c, :], pi[m], x[m])
                    np.add.at(P["ch_sq"][:, c], pi[m], sq[m])
            m = which == 2                                             # fallbacks: per (pair, fallback centre)
            if m.any():
                keys = pi[m].astype(np.int64) * self.k + second[sel][m].astype(np.int64)
                uk, inv = np.unique(keys, return_inverse=True)
                cnt = np.bincount(inv, minlength=len(uk)).astype(np.float64)
                sx = np.zeros((len(uk), x.shape[1]))
                np.add.at(sx, inv, x[m])
                sqs = np.bincount(inv, weights=sq[m], minlength=len(uk))
                for t, key in enumerate(uk.tolist()):
                    acc = P["fb"].setdefault(key, [0.0, np.zeros(x.shape[1]), 0.0])
                    acc[0] += cnt[t]; acc[1] += sx[t]; acc[2] += sqs[t]
        cand, don = role == 0, role == 1
        npair = len(P["j"])
        P["cost_sum"] += np.bincount(pi[cand], weights=change[cand], minlength=npair)
        P["cost_sq"] += np.bincount(pi[cand], weights=change[cand] ** 2, minlength=npair)
        P["cost_n"] += np.bincount(pi[cand], minlength=npair)
        P["gain_sum"] += np.bincount(pi[don], weights=-change[don], minlength=npair)
        P["gain_n"] += np.bincount(pi[don], minlength=npair)

    def _finish_step(self, expected_n):
        """As the parent's, but the revival check runs after every batch ("every") or when the bound says a move may
        pay ("event"); "periodic" keeps the library's every five batches."""
        if getattr(self, "trigger", "periodic") == "periodic":
            return super()._finish_step(expected_n)
        if self.trigger == "every" or self._bound_positive():
            self.n_full_checks += 1
            self._revive_unused(self._split_rng)
        self._invalidate_readout()
        self._step += 1
        self._epoch_steps += 1
        self._epoch_rows += expected_n
        self._pending_stats = None
        self._batch_accumulation = self._batch_objective = None
        self._batch_centroids = None

    def _bound_positive(self):
        """Upper bound on the best move's estimated net gain: max over eligible donors of G~ minus min over eligible
        candidates of E[lambda] r~ delta_LB (O(k) plus distances to the centres moved since the last full check)."""
        if self._frozen():
            return False
        if self._delta_ref is None:
            return True                                     # no exact delta yet: do a full check
        self.n_bound_evals += 1
        usage = self._usage.astype("float64")
        cooling = self._stale > 0.5
        rate = self._posterior_rates()
        raw = (2.0 / np.pi) * self._axis_scale * np.linalg.norm(self._removal_stats[:, :self.d].astype("float64"), axis=1)
        gain = self._posterior_gains(raw, rate)
        donors = (usage > 0) & ~cooling
        if not donors.any():
            return False
        gmax = float(gain[donors].max())
        C = self._centroids.astype(np.float64)
        disp = np.sqrt(((C - self._pos_ref) ** 2).sum(1))
        new = np.array(sorted(self._new_since), dtype=np.int64)
        old = np.ones(self.k, dtype=bool)
        old[new] = False
        d_old = np.where(old, disp, 0.0)
        top = np.argsort(d_old)[-2:]
        mx1, mx2 = d_old[top[-1]], d_old[top[0]] if len(top) > 1 else 0.0
        other = np.where(np.arange(self.k) == top[-1], mx2, mx1)          # largest displacement of another centre
        lb = np.maximum(np.sqrt(np.maximum(self._delta_ref, 0.0)) - disp - other, 0.0) ** 2
        if len(new):
            dn = faiss.knn(np.ascontiguousarray(C.astype(np.float32)), np.ascontiguousarray(C[new].astype(np.float32)),
                           min(2, len(new)))[0]
            dn = np.where(np.isin(np.arange(self.k), new)[:, None] & (dn <= 0), np.inf, dn)   # not to itself
            lb = np.minimum(lb, dn.min(1).astype(np.float64))
        cand = ~cooling
        if not cand.any():
            return False
        cost_lb = rate * np.maximum(self._r_ref, 0.0) * lb
        return gmax > float(cost_lb[cand].min())

    def _accumulate_core(self, x, index=None, labels_out=None):
        if self._collect and self._pending_stats is None:
            self._batch_dist[:] = 0.0
        return super()._accumulate_core(x, index, labels_out=labels_out)

    def _apply_statistics(self, ids, sums, counts, objective, alpha, decay, replace_usage=False):
        t0 = time.perf_counter()
        inc = None
        if getattr(self, "gain_noise", "") == "measured" and self._removal_stats is not None and not self._frozen():
            # this batch's scatter along each cluster's split axis: b = <sum p y, u> = sum p^2 (read before the parent
            # folds and zeroes the batch rows)
            inc = np.einsum("ij,ij->i", self._batch_removal[:, :self.d].astype(np.float64),
                            self._axes.astype(np.float64))
        super()._apply_statistics(ids, sums, counts, objective, alpha, decay, replace_usage)
        if inc is not None:
            g = 0.0 if replace_usage else decay
            a = 1.0 if replace_usage else alpha
            E = self._inc
            E[0] = g * E[0] + a * inc
            E[1] = g * E[1] + a * inc * inc
            E[2] = g * E[2] + a
            E[3] = g * g * E[3] + a * a
        TIMES["ema_incl_axes"] += time.perf_counter() - t0
        self._stale *= (0.0 if replace_usage else decay)
        self._decay_prod *= (0.0 if replace_usage else decay)
        self._cum_n += np.asarray(counts, dtype=np.float64)
        self._age += 1.0
        if getattr(self, "_pend", None) is not None and not self._frozen():
            self._pend["batches"] += 1
            if self.confirm_adapt:
                self._pend["win_n"] += np.asarray(counts, dtype=np.float64)
        g = 0.0 if replace_usage else decay                        # forgetting factor 1 - alpha_t
        self._dn = g * self._dn + np.asarray(counts, dtype=np.float64)
        self._dt = g * self._dt + 1.0
        self._dw2 = g * g * self._dw2 + 1.0
        if getattr(self, "_measure_all", False):
            # per-point statistics with the algorithm's memory: EMA with this batch's step (all centres decay;
            # sums of squared weights give the effective number of points behind each EMA mean)
            a = 0.0 if replace_usage else decay
            self._ema *= a
            self._ema += (1.0 - a) * self._ema_batch
            self._ema_w2 = a * a * self._ema_w2 + (1.0 - a) ** 2 * self._ema_batch[0]
            self._ema_batch[:] = 0.0
        if self._collect:
            t0 = time.perf_counter()
            touched = counts > 0
            self._last_hit[touched] = self._step
            if replace_usage:
                self._dist[touched] = alpha * self._batch_dist[touched]
            else:
                self._dist *= decay
                self._dist[touched] += alpha * self._batch_dist[touched]
            TIMES["dist_stats"] += time.perf_counter() - t0

    # ------------------------------------------------------------------ revival
    def _dist_ok(self, ids):
        """Distance test for candidates ids: True = may be removed."""
        C = self._centroids
        delta = faiss.knn(np.ascontiguousarray(C[ids]), C, 2)[0][:, 1].astype(np.float64)
        m = self._dist.sum() / max(float(self._usage.sum(dtype=np.float64)), 1e-300)
        ok = delta <= self.tau_near * m
        if self.abandon_passes is not None:
            window = self.abandon_passes * self.recipe["steps_per_pass"]
            abandoned = (self._step - self._last_hit[ids]) >= window
            self.n_abandoned += int((abandoned & ~ok).sum())
            ok |= abandoned
        self.n_candidates += len(ids)
        self.n_protected += int((~ok).sum())
        return ok

    def _size_donors(self, rng, donors_mask, n):
        usage = self._usage
        uniform = np.maximum(rng.random_sample(self.k), np.finfo("float64").tiny)
        keys = np.full(self.k, -np.inf)
        keys[donors_mask] = np.log(usage[donors_mask]) - np.log(-np.log(uniform[donors_mask]))
        return np.argsort(keys)[-n:][::-1]

    def _gain_order(self, donors):
        axis_norm = self._axis_scale * np.linalg.norm(self._removal_stats[:, :self.d].astype("float64"), axis=1)
        return donors[np.argsort(-axis_norm[donors].astype("float32"), kind="stable")]

    def _choose_splits(self, rng):
        t0 = time.perf_counter()
        self._calib_pred = None
        try:
            source, target = self._choose(rng)
        finally:
            TIMES["decide"] += time.perf_counter() - t0
        if getattr(self, "calibrate", False) and self._calib_pred is not None:
            self._calibrate(source, target)
        return source, target

    # ------------------------------------------------------------------ calibration (diagnostic only)
    def _split_children(self, d):
        """The two positions a split of donor d would use now: exactly those of _apply_splits."""
        donor_c, child_c = self._split_positions(np.array([d], dtype="int64"))
        return donor_c[0].astype("float64"), child_c[0].astype("float64")

    def _calibrate(self, source, target):
        """Predicted cost/gain at this check vs their exact values on the full train and test sets
        (current centres, first order: removal = points to their second-nearest centre; split = the
        donor's points to the nearer of the two children placed as the rule would place them)."""
        from faiss.contrib.e_means.base import _scheduled_alpha
        pred = self._calib_pred
        k, C = self.k, np.ascontiguousarray(self._centroids)
        alpha = float(_scheduled_alpha(self.recipe["alpha0"], self._step, self.recipe["total_steps"]))
        B = int(self.recipe["batch_size"])
        moves = [] if source is None else list(zip(np.asarray(target).tolist(), np.asarray(source).tolist()))
        sample = self.calib_rng.choice(k, min(64, k), replace=False).tolist()
        donors = sorted({d for _, d in moves} | set(sample))
        truth = {}
        for name, Xs in (("tr", self.calib_X), ("te", self.calib_Xte)):
            D, I = faiss.knn(Xs, C, 2)
            d1 = np.maximum(D[:, 0].astype("float64"), 0.0)
            margin = np.maximum(D[:, 1].astype("float64") - d1, 0.0)
            lab = I[:, 0]
            m = np.bincount(lab, minlength=k)
            cost = np.bincount(lab, weights=margin, minlength=k)
            order = np.argsort(lab, kind="stable")
            starts = np.searchsorted(lab[order], np.arange(k + 1))
            gain = {}
            for d in donors:
                idx = order[starts[d]:starts[d + 1]]
                if len(idx) == 0:
                    gain[d] = 0.0
                    continue
                c1, c2 = self._split_children(d)
                x = Xs[idx].astype("float64")
                new = np.minimum(((x - c1) ** 2).sum(1), ((x - c2) ** 2).sum(1))
                gain[d] = float(d1[idx].sum() - new.sum())
            truth[name] = dict(n=len(Xs), m=m, cost=cost, gain=gain, mse=float(d1.mean()))
        tr, te = truth["tr"], truth["te"]
        base = dict(step=int(self._step), epoch=int(self._epoch), alpha=alpha, B=B, n_tr=tr["n"], n_te=te["n"],
                    ema_weight=1.0 - self._decay_prod, pred_unbiased=bool(pred.get("unbiased", False)),
                    mse_tr=tr["mse"], mse_te=te["mse"])

        def centre(j):
            return dict(N=float(self._usage[j]), stale=float(self._stale[j]), m_tr=int(tr["m"][j]),
                        cum_n=float(self._cum_n[j]), age=float(self._age[j]),
                        m_te=int(te["m"][j]), cost_pred=float(pred["cost"][j]), cost_tr=float(tr["cost"][j]),
                        cost_te=float(te["cost"][j]),
                        delta=None if pred["delta"] is None else float(pred["delta"][j]))

        def donor(d):
            return dict(N=float(self._usage[d]), stale=float(self._stale[d]), m_tr=int(tr["m"][d]),
                        cum_n=float(self._cum_n[d]), age=float(self._age[d]),
                        rate_pred=None if pred.get("rate") is None else float(pred["rate"][d]),
                        m_te=int(te["m"][d]), gain_pred=float(pred["gain"][d]), gain_tr=tr["gain"][d],
                        gain_te=te["gain"][d])

        for j, d in moves:
            self.calib_rows.append(dict(base, kind="move", j=int(j), d=int(d), cand=centre(j), donor=donor(d)))
        for c in sample:
            self.calib_rows.append(dict(base, kind="sample", j=int(c), cand=centre(c), donor=donor(c)))

    def _choose(self, rng):
        if self._frozen():
            return None, None
        usage = self._usage
        if self.removal.startswith("ng"):
            if self.confirm and self.confirm_wait == "evidence":
                return self._choose_confirm_wait()
            if self.confirm:
                return self._choose_confirm()
            return self._choose_net_gain()
        if self.removal.startswith("mass"):
            unused = usage < _numpy_unused_threshold(usage, self.k, 0.1)
            evict = np.flatnonzero(unused)
            if self.removal == "mass_dist" and len(evict):
                evict = evict[self._dist_ok(evict)]
            donors_mask = (~unused) & (usage > 0)
            n = min(len(evict), int(donors_mask.sum()))
            if n == 0:
                return None, None
            if self.donor == "size":
                return self._size_donors(rng, donors_mask, n), evict[:n]
            return self._gain_order(np.flatnonzero(donors_mask))[:n], evict[:n]
        # cost-based candidates (PR _choose_removal_splits)
        removal = self._removal_stats[:, self.d].astype("float64")
        cand = np.flatnonzero((removal <= 0) | (removal < REMOVAL_THRESHOLD * np.median(removal)))
        if len(cand) == 0:
            return None, None
        cand = cand[np.argsort(removal[cand], kind="stable")]
        if self.removal == "cost_dist":
            free = removal[cand] <= 0                     # no row: removal costs nothing
            ok = np.ones(len(cand), dtype=bool)
            if (~free).any():
                ok[~free] = self._dist_ok(cand[~free])
            cand = cand[ok]
            if len(cand) == 0:
                return None, None
        kept, kept_ids, protected = [], set(), set()
        centroids = self._readout()
        for j, nearest in zip(cand, self._nearest_other(cand)):
            back = nearest in kept_ids and not (centroids[j] == centroids[nearest]).all()
            if j not in protected and not back:
                kept.append(j)
                kept_ids.add(j)
                protected.add(nearest)
        kept = np.array(kept, dtype="int64")
        donors_mask = usage > 0
        donors_mask[kept] = False
        n = min(len(kept), int(donors_mask.sum()))
        if n == 0:
            return None, None
        if self.donor == "gain":
            return self._gain_order(np.flatnonzero(donors_mask))[:n], kept[:n]
        return self._size_donors(rng, donors_mask, n), kept[:n]

    def _ivf_knn2(self, C):
        """Approximate 2-NN of every centre among the centres: IVF with round(sqrt(k)) lists whose coarse
        centroids are centres sampled at this check (no training), probing max(2, nlist // 8) lists."""
        k, d = C.shape
        nlist = max(1, int(round(np.sqrt(k))))
        rng = np.random.RandomState((self.seed * 1000003 + self._step) % (2 ** 31))
        coarse = np.ascontiguousarray(C[np.sort(rng.choice(k, nlist, replace=False))])
        quantizer = faiss.IndexFlatL2(d)
        quantizer.add(coarse)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
        index.is_trained = True
        index.add(C)
        index.nprobe = max(2, nlist // 8) if nlist >= 2 else 1
        D, I = index.search(C, 2)
        return D, I

    def _posterior_rates_discounted(self):
        """Gamma-Poisson posterior means from discounted evidence (N', T'): (a + N') / (b + T'). Prior by moments
        over centres with T' >= 1: mean m of N'/T', between-centre variance = var(N'/T') - mean(rate * W2' / T'^2)
        (the Poisson part for discounted weights)."""
        n, t, w2 = self._dn, self._dt, self._dw2
        seen = t >= 1.0
        if seen.sum() < 2:
            return np.where(t > 0, n / np.maximum(t, 1e-12), 0.0)
        r = n[seen] / t[seen]
        m = float(r.mean())
        if not m > 0:
            return np.zeros(self.k)
        v_lam = float(r.var()) - float(np.mean(r * w2[seen] / t[seen] ** 2))
        v_lam = max(v_lam, 1e-6 * m * m)
        a, b = m * m / v_lam, m / v_lam
        return (a + n) / (b + t)

    def _posterior_rates(self):
        """Posterior mean of each centre's expected points per batch (Gamma-Poisson), from the points assigned
        since its birth (or since the start) over the batches since then. The Gamma prior is fitted to all centres
        by moments, removing the Poisson part of the spread of the raw rates. A centre born at this check (age 0)
        gets the prior mean."""
        age, cum = self._age, self._cum_n
        seen = age >= 1
        if seen.sum() < 2:
            return np.where(age > 0, cum / np.maximum(age, 1.0), 0.0)
        r = cum[seen] / age[seen]
        m = float(r.mean())
        if not m > 0:
            return np.zeros(self.k)
        v_lam = float(r.var()) - float(np.mean(r / age[seen]))          # between-centre variance of the rates
        v_lam = max(v_lam, 1e-6 * m * m)
        a, b = m * m / v_lam, m / v_lam                                   # Gamma(shape a, rate b)
        return (a + cum) / (b + age)

    def _gain_rel_var(self):
        """Relative variance per point of a variance-type estimate: kappa - 1 (2 for Gaussian data). For ng_post2,
        kappa is the pooled kurtosis of the projections on the split axes of the sampled rows (clusters with >= 2
        sampled rows and positive spread, weighted by their sampled rows); otherwise the Gaussian value."""
        gauss = 3.0 if getattr(self, "gain_noise", "fixed_n") == "compound" else 2.0   # Gaussian kappa or kappa-1
        if not getattr(self, "_sample_margins", False):
            return gauss
        ok = (self._m_n >= 2) & (self._p2 > 0)
        if not ok.any():
            return gauss
        n = self._m_n[ok]
        kappa = (self._p4[ok] / n) / (self._p2[ok] / n) ** 2
        kbar = float(np.average(kappa, weights=n))
        if getattr(self, "gain_noise", "fixed_n") == "compound":
            return max(kbar, 1e-6)                   # compound Poisson: Var(sum p^2) / E^2 = kappa / n
        return max(kbar - 1.0, 1e-6)

    def _margin_ratios(self, delta):
        """Shrunk ratio r_hat_j = (mean sampled margin of j) / delta_j: normal-normal empirical Bayes toward the
        population ratio rho (mean of the observed ratios weighted by sampled rows), with the pooled within-cluster
        variance of margin / delta and the between-cluster variance by moments. Clusters without samples (or
        without a finite positive delta) get rho. Returns (r_hat, rho)."""
        k = self.k
        ok = (self._m_n > 0) & np.isfinite(delta) & (delta > 0)
        r_hat = np.ones(k)
        if not ok.any():
            return r_hat, 1.0                       # no sampled evidence yet: the bound itself (ratio 1)
        n, dl = self._m_n[ok], delta[ok]
        r = (self._m_sum[ok] / n) / dl
        rho = float(np.average(r, weights=n))
        if ok.sum() < 2:                            # one cluster: no between-cluster variance; all get rho
            return np.full(k, max(rho, 0.0)), rho
        many = n >= 2
        if self.moderate:
            # per-cluster sample variances of m / delta, moderated across clusters (Smyth); prior value for n < 2
            s2r = np.zeros(len(n))
            s2r[many] = np.maximum(self._m_sq[ok][many] - self._m_sum[ok][many] ** 2 / n[many], 0.0) / (
                (n[many] - 1.0) * dl[many] ** 2)
            mod, s0, _ = self._moderated_variances(np.where(many, s2r, 0.0), np.where(many, n - 1.0, 0.0))
            sig = np.where(many, mod, s0)
            sig = np.where(np.isfinite(sig) & (sig > 0), sig, max(float(r.var()), 1e-12))
            noise = sig / n
            tau2 = max(float(r.var()) - float(np.mean(noise)), 1e-6 * rho * rho, 1e-12)
        else:
            if many.any():
                within = (self._m_sq[ok][many] - self._m_sum[ok][many] ** 2 / n[many]) / dl[many] ** 2
                sigma2 = max(float(within.sum()) / float((n[many] - 1).sum()), 1e-12)
            else:
                sigma2 = max(float(r.var()), 1e-12)
            tau2 = max(float(r.var()) - float(np.mean(sigma2 / n)), 1e-6 * rho * rho, 1e-12)
            noise = sigma2 / n
        r_hat[:] = rho
        r_hat[ok] = (tau2 * r + noise * rho) / (tau2 + noise)
        self._ratio_prior = (rho, tau2, float(np.median(noise * n)) if len(n) else 0.0)
        return np.maximum(r_hat, 0.0), rho

    @staticmethod
    def _moderated_variances(s2, df):
        """Smyth's (2004) moderated variances: s~^2 = (d0 s0^2 + df s^2) / (d0 + df), with d0 and s0^2 estimated from
        the spread of log s^2 across units (method of moments with digamma/trigamma). Units with df <= 0 or s2 <= 0 do
        not enter the fit and get s0^2. Returns (moderated variances, s0^2, d0)."""
        from scipy.special import digamma, polygamma
        out = np.zeros(len(s2))
        ok = (df > 0) & (s2 > 0) & np.isfinite(s2)
        if ok.sum() < 3:
            s0 = float(np.mean(s2[ok])) if ok.any() else 0.0
            out[:] = s0
            return out, s0, np.inf
        dg = df[ok] / 2.0
        e = np.log(s2[ok]) - digamma(dg) + np.log(dg)
        ebar = float(e.mean())
        target = float(e.var(ddof=1)) - float(np.mean(polygamma(1, dg)))
        if target > 0:
            # invert the trigamma function (Smyth's Newton iteration)
            if target > 1e7:
                y = 1.0 / np.sqrt(target)
            elif target < 1e-6:
                y = 1.0 / target
            else:
                y = 0.5 + 1.0 / target
                for _ in range(50):
                    tri = float(polygamma(1, y))
                    dif = tri * (1.0 - tri / target) / float(polygamma(2, y))
                    y += dif
                    if -dif / y < 1e-8:
                        break
            d0 = 2.0 * y
            s0 = float(np.exp(ebar + digamma(d0 / 2.0) - np.log(d0 / 2.0)))
            out[:] = s0
            out[ok] = (d0 * s0 + df[ok] * s2[ok]) / (d0 + df[ok])
        else:
            d0, s0 = np.inf, float(np.exp(min(ebar, 700.0)))
            out[:] = s0
        return out, s0, d0

    @staticmethod
    def _eb_normal(theta_hat, v, ok, weights):
        """Normal-normal empirical Bayes (eq. 5 of the note): posterior means of theta for units with ok, the prior
        mean theta_bar (weighted by `weights`) for the others; tau^2 by moments, floored."""
        post = np.zeros(len(theta_hat))
        ok = ok & np.isfinite(theta_hat) & np.isfinite(v) & (v >= 0)
        if not ok.any():
            return post, 0.0
        th, vv, w = theta_hat[ok], v[ok], weights[ok]
        tbar = float(np.average(th, weights=w))
        if ok.sum() < 2:
            post[:] = tbar
            return post, tbar
        tau2 = max(float(th.var()) - float(np.mean(vv)), 1e-6 * tbar * tbar, 1e-12)
        post[:] = tbar
        post[ok] = (tau2 * th + vv * tbar) / (tau2 + vv)
        return post, tbar

    def _split_offsets(self):
        """s_d for every centre, exactly as _split_positions computes it (the PR's sqrt(2 |U| / (pi N)))."""
        norm = self._axis_scale * np.linalg.norm(self._removal_stats[:, :self.d].astype("float64"), axis=1)
        mass = self._usage.astype("float64")
        valid = (norm > 0) & (mass > 0) & np.isfinite(norm) & np.isfinite(mass)
        return np.sqrt(2.0 * np.where(valid, norm / np.where(valid, mass, 1.0), 0.0) / np.pi)

    def _stats3(self, which):
        """(n, mean, sample variance) of the margins (which=0) or of |p| (which=1) per centre: over all points since
        birth (memory="birth"), or EMA-weighted with the effective number of points (sum w)^2 / sum w^2 (memory="ema")."""
        if self.memory == "birth":
            n = self._m_n
            s1, s2 = (self._m_sum, self._m_sq) if which == 0 else (self._pabs, self._p2)
            mean = np.where(n > 0, s1 / np.maximum(n, 1.0), 0.0)
            var = np.where(n >= 2, (s2 - n * mean * mean) / np.maximum(n - 1.0, 1.0), 0.0)
            return n, mean, np.maximum(var, 0.0)
        en, e1, e2 = self._ema[0], self._ema[1 + 2 * which], self._ema[2 + 2 * which]
        pos = en > 0
        mean = np.where(pos, e1 / np.where(pos, en, 1.0), 0.0)
        var = np.where(pos, e2 / np.where(pos, en, 1.0) - mean * mean, 0.0)
        n = np.where(pos & (self._ema_w2 > 0), en * en / np.where(self._ema_w2 > 0, self._ema_w2, 1.0), 0.0)
        # own variance only from an effective size of 2 (else the prior variance is used); unbiased for effective n
        var = np.where(n >= 2.0, np.maximum(var, 0.0) * n / np.maximum(n - 1.0, 1.0), 0.0)
        bad = ~(np.isfinite(mean) & np.isfinite(var) & np.isfinite(n))
        n, mean, var = np.where(bad, 0.0, n), np.where(bad, 0.0, mean), np.where(bad, 0.0, var)
        return n, mean, var

    def _cost_ratios3(self, delta):
        """r~_j: shrunk ratio of the mean margin to delta_j (eq. 7 of the note). Returns (r~, r_bar)."""
        n, mean, s2 = self._stats3(0)
        ok = (n > 0) & np.isfinite(delta) & (delta > 0)
        dl = np.where(ok, delta, 1.0)
        r_hat = mean / dl
        s2r = s2 / (dl * dl)                                   # variance of margin / delta (dimensionless)
        own = ok & (n >= 2)
        mod, s0, _ = self._moderated_variances(np.where(own, s2r, 0.0), np.where(own, n - 1.0, 0.0))
        mod = np.where(own, mod, s0)
        v = mod / np.maximum(n, 1.0)
        r_tilde, r_bar = self._eb_normal(r_hat, v, ok, n)
        return np.maximum(r_tilde, 0.0), r_bar

    def _posterior_gains3(self, rate):
        """gain_d = E[lambda_d] * s_d^2 * (2 q~_d - 1), q_d = mean |p| / s_d (eq. 8 of the note)."""
        n, mean, s2 = self._stats3(1)                       # mean |p| and variance of |p|
        s = self._split_offsets()
        ok = (n > 0) & (s > 0)
        sd = np.where(ok, s, 1.0)
        q_hat = mean / sd
        s2q = s2 / (sd * sd)                                   # variance of |p| / s (dimensionless)
        own = ok & (n >= 2)
        mod, s0, _ = self._moderated_variances(np.where(own, s2q, 0.0), np.where(own, n - 1.0, 0.0))
        mod = np.where(own, mod, s0)
        v = mod / np.maximum(n, 1.0)
        q_tilde, self._q_bar = self._eb_normal(q_hat, v, ok, n)
        return np.where(s > 0, rate * s * s * (2.0 * q_tilde - 1.0), 0.0)

    def _measured_gain_var(self, fallback):
        """Relative variance of each gain estimate from the batch-to-batch variation of its increments: CV^2(b) per
        batch over the effective number of batches n_b = (sum w)^2 / sum w^2 of the EMA, with CV^2 moderated across
        clusters (Smyth; df = n_b - 1). Clusters without increments keep the fallback value."""
        E1, E2, W, W2 = self._inc
        ok = (W > 0) & (W2 > 0) & (E1 > 0)
        out = np.array(fallback, dtype=np.float64, copy=True)
        if ok.sum() < 3:
            return out
        nb = np.where(ok, W * W / np.where(ok, W2, 1.0), 0.0)
        m = np.where(ok, E1 / np.where(ok, W, 1.0), 0.0)
        q = np.where(ok, E2 / np.where(ok, W, 1.0), 0.0)
        cv2 = np.where(ok & (m > 0), np.maximum(q - m * m, 0.0) / np.where(m > 0, m * m, 1.0), 0.0)
        cv2 = cv2 * np.where(nb > 1, nb / np.maximum(nb - 1.0, 1e-12), 1.0)   # unbiased for the effective size
        own = ok & (nb >= 2) & (cv2 > 0)
        mod, s0, _ = self._moderated_variances(np.where(own, cv2, 0.0), np.where(own, nb - 1.0, 0.0))
        mod = np.where(own, mod, s0)
        out[ok] = mod[ok] / np.maximum(nb[ok], 1.0)
        return out

    @staticmethod
    def _tweedie(theta, v):
        """Tweedie's formula (Efron 2011) for estimates theta_i ~ N(mu_i, v_i): E[mu|theta] = theta + v (log f)'(theta),
        Var = v (1 + v (log f)''(theta)), with f the marginal density of all theta, estimated by a binned Gaussian KDE
        (Silverman's bandwidth). Returns (posterior means, posterior variances)."""
        n = len(theta)
        sd = float(theta.std())
        iqr = float(np.subtract(*np.percentile(theta, [75, 25])))
        h = 0.9 * min(sd, iqr / 1.34 if iqr > 0 else sd) * n ** (-0.2)
        if not h > 0:
            return theta, v
        lo, hi = float(theta.min()) - 4 * h, float(theta.max()) + 4 * h
        M = 1024
        grid = np.linspace(lo, hi, M)
        dx = grid[1] - grid[0]
        counts = np.bincount(np.clip(((theta - lo) / dx).round().astype(int), 0, M - 1), minlength=M).astype(float)
        u = np.arange(-int(4 * h / dx) - 1, int(4 * h / dx) + 2) * dx
        ker = np.exp(-0.5 * (u / h) ** 2) / (np.sqrt(2 * np.pi) * h)
        f = np.convolve(counts, ker, mode="same") / n
        f1 = np.convolve(counts, -u / h ** 2 * ker, mode="same") / n
        f2 = np.convolve(counts, (u ** 2 / h ** 4 - 1 / h ** 2) * ker, mode="same") / n
        fi, f1i, f2i = (np.interp(theta, grid, a) for a in (f, f1, f2))
        fi = np.maximum(fi, 1e-300)
        l1 = f1i / fi
        l2 = f2i / fi - l1 * l1
        th = theta + v * l1
        vp = np.maximum(v * (1.0 + v * l2), 0.0)
        return th, vp

    def _posterior_gains(self, raw, rate):
        """Posterior mean of each centre's split gain (per batch). The raw estimate (2/pi)|U| is divided by the
        EMA weight 1 - prod(1 - alpha) (start-up bias), then shrunk on the log scale toward the mean of all
        centres (normal-normal empirical Bayes) with the relative variance 2 / n_eff of a variance-type estimate,
        n_eff = expected points in the EMA's memory = rate * min((2 - alpha) / alpha, batches so far).
        Zero-gain policy: a centre with zero raw gain (no rows, or no spread) cannot gain from a split, keeps gain 0
        (it can never be a donor, since costs are >= 0) and is left out of the prior; with fewer than two positive
        gains there is no population to shrink toward and the raw (bias-corrected) gains are used."""
        from faiss.contrib.e_means.base import _scheduled_alpha
        g = raw / max(1.0 - self._decay_prod, 1e-12)
        alpha = float(_scheduled_alpha(self.recipe["alpha0"], self._step, self.recipe["total_steps"]))
        alpha = min(max(alpha, 1e-6), 1.0)
        window = min((2.0 - alpha) / alpha, float(self._step + 1))
        s2 = self._gain_rel_var() / np.maximum(rate * window, 1e-3)
        if getattr(self, "gain_noise", "") == "measured":
            s2 = self._measured_gain_var(s2)
        out = np.zeros(self.k)
        pos = g > 0
        if pos.sum() < 2:
            out[pos] = g[pos]
            return out
        lg, s2p = np.log(g[pos]), s2[pos]
        if getattr(self, "gain_eb", "normal") == "tweedie" and pos.sum() >= 10:
            th, vp = self._tweedie(lg, s2p)
            out[pos] = np.exp(th + vp / 2.0)
            return out
        mu = float(lg.mean())
        tau2 = max(float(lg.var()) - float(s2p.mean()), 1e-6)            # between-centre variance of log gains
        post_mean = (tau2 * lg + s2p * mu) / (tau2 + s2p)
        post_var = tau2 * s2p / (tau2 + s2p)
        out[pos] = np.exp(post_mean + post_var / 2.0)
        return out

    def _choose_net_gain(self):
        """Greedy matching: cheapest removal with largest-gain donor, accepted while cost < w * gain."""
        usage = self._usage.astype("float64")
        gain = (2.0 / np.pi) * self._axis_scale * np.linalg.norm(
            self._removal_stats[:, :self.d].astype("float64"), axis=1)
        if self.cooldown == "none":
            cooling = np.zeros(self.k, dtype=bool)
        elif self.cooldown == "check":
            cooling = (self._step - self._split_step) <= 5          # split at the previous check
        else:
            cooling = self._stale > (0.5 if self.cooldown == "half" else 0.25)
        rate = None
        if self.removal in ("ng_post", "ng_post2"):
            rate = self._posterior_rates()
            gain = self._posterior_gains(gain, rate)
        elif self.removal == "ng_post3":
            rate = self._posterior_rates_discounted() if self.rate_mode == "discount" else self._posterior_rates()
            grate = rate
            if self.gain_rate == "ema":
                grate = self._usage.astype("float64") / max(1.0 - self._decay_prod, 1e-12)
            gain = self._posterior_gains3(grate)
        delta = nn_other = None
        if self.removal == "ng_est":
            cost = self._removal_stats[:, self.d].astype("float64")
        else:
            C = np.ascontiguousarray(self._centroids)
            t0 = time.perf_counter()
            if self.knn == "ivf":
                D, I = self._ivf_knn2(C)
            else:
                D, I = faiss.knn(C, C, 2)
            TIMES["delta_search"] += time.perf_counter() - t0
            ids = np.arange(self.k)
            # nearest other centre: the first result unless it is the centre itself (a duplicate may come first)
            nn_other = np.where(I[:, 0] == ids, I[:, 1], I[:, 0])
            delta = np.where(I[:, 0] == ids, D[:, 1], D[:, 0]).astype("float64")
            if self.removal not in ("ng_post", "ng_post2", "ng_post3"):
                delta = D[:, 1].astype("float64")       # earlier rules: second result, as in the runs so far
            missing = (nn_other < 0) | ~np.isfinite(delta)
            delta = np.where(missing, np.finfo("float64").max / 4, delta)   # no neighbour found: never cheap
            nn_other = np.where(nn_other < 0, ids, nn_other)
            cost = (usage if rate is None else rate) * np.maximum(delta, 0.0)
            if self.removal == "ng_post2":
                ratio, self._rho = self._margin_ratios(delta)
                cost = cost * ratio
                if self.trigger == "event":
                    self._delta_ref = np.array(delta, dtype=np.float64, copy=True)
                    self._pos_ref = np.array(self._centroids, dtype=np.float64, copy=True)
                    self._r_ref = np.array(ratio, dtype=np.float64, copy=True)
                    self._new_since = set()
            elif self.removal == "ng_post3":
                ratio, self._rho = self._cost_ratios3(delta)
                cost = cost * ratio
        self._calib_pred = dict(cost=cost, gain=gain, delta=delta, unbiased=rate is not None, rate=rate)
        self._check_cache = dict(rate=rate, delta=delta)
        donors = np.flatnonzero((usage > 0) & ~cooling)
        if len(donors) == 0:
            return None, None
        donors = donors[np.argsort(-gain[donors], kind="stable")]
        cand = np.flatnonzero((cost < self.w * gain[donors[0]]) & ~cooling)     # nothing else can pass
        if len(cand) == 0:
            return None, None
        cand = cand[np.argsort(cost[cand], kind="stable")]
        self.n_candidates += len(cand)
        centroids = self._readout()
        if self.removal in ("ng_post", "ng_post2", "ng_post3") and nn_other is not None and self.knn == "exact":
            nearest = dict(zip(cand.tolist(), nn_other[cand].tolist()))      # reuse the (exact) delta search
        else:   # exact nearest for the pair guard (also with knn="ivf": an approximate neighbour would weaken it)
            nearest = dict(zip(cand.tolist(), self._nearest_other(cand).tolist()))
        if self.matcher == "v2":
            return self._match_v2(cand, donors, cost, gain, nearest, centroids,
                                  forbid_nearest=self.no_fallback_donor)
        kept, source, kept_ids, protected, used = [], [], set(), set(), set()
        pos = 0
        for j in cand.tolist():
            while pos < len(donors) and (donors[pos] in kept_ids or donors[pos] == j):
                pos += 1
            if pos == len(donors):
                break
            d = int(donors[pos])
            if not cost[j] < self.w * gain[d]:
                break                                               # costs rise, gains fall: done
            nb = nearest[j]
            back = nb in kept_ids and not (centroids[j] == centroids[nb]).all()
            if j in protected or back or j in used:
                continue
            kept.append(j); kept_ids.add(j); protected.add(nb)
            source.append(d); used.add(d)
            pos += 1
        if not kept:
            return None, None
        return np.array(source, dtype="int64"), np.array(kept, dtype="int64")

    def _choose_confirm(self):
        """Confirm last check's proposals on fresh points, execute the confirmed ones, propose new ones."""
        self._ratio_prior = None
        self._check_cache = None
        src, tgt = self._choose_net_gain()                  # this check's estimates and proposals
        acc_s, acc_t = [], []
        P = self._pend
        self._pend = None
        if P is not None and P["batches"] > 0 and self._check_cache is not None and self._check_cache["rate"] is not None:
            rate, delta = self._check_cache["rate"], self._check_cache["delta"]
            prior = getattr(self, "_ratio_prior", None)
            rho, tau2, s0 = prior if prior is not None else (1.0, 1.0, 1.0)
            F = float(P["batches"])
            fb_by_pair = {}
            if self.confirm_adapt:
                for key, val in P["fb"].items():
                    fb_by_pair.setdefault(key // self.k, []).append((key % self.k, val))
            for i in range(len(P["j"])):
                j, d = int(P["j"][i]), int(P["d"][i])
                gain = P["gain_sum"][i] / F                      # exact, per batch, on fresh points
                dl = float(delta[j]) if np.isfinite(delta[j]) and delta[j] > 0 else 0.0
                n = float(P["cost_n"][i])
                if dl > 0:
                    if n > 0:
                        mean = P["cost_sum"][i] / n
                        var = (P["cost_sq"][i] - n * mean * mean) / (n - 1.0) if n >= 2 else s0 * dl * dl
                        v = max(var, 0.0) / (n * dl * dl)
                        r = (tau2 * (mean / dl) + v * rho) / (tau2 + v)
                    else:
                        r = rho                                  # no fresh point: the isolation prior
                    cost = float(rate[j]) * max(r, 0.0) * dl
                else:
                    cost = float(rate[j]) * (P["cost_sum"][i] / n if n > 0 else 0.0)
                rec_v = 0.0
                if self.confirm_adapt:
                    rec_v = self._recentring_value(P, i, d, fb_by_pair.get(i, ())) / F
                    gain += rec_v
                self.confirm_log.append(dict(step=int(self._step), F=F, n_d=float(P["gain_n"][i]), n_j=n,
                                             gain_imm=float(P["gain_sum"][i] / F), recentre=rec_v, cost=float(cost),
                                             pred_gain=float(P["pred_gain"][i]), pred_cost=float(P["pred_cost"][i]),
                                             accept=bool(gain > self.w * cost)))
                if gain > self.w * cost:
                    acc_s.append(d); acc_t.append(j)
            self.n_confirmed += len(acc_s)
        if src is not None:                                  # new proposals, excluding centres moved now
            busy = set(acc_s) | set(acc_t)
            keep = [(int(a), int(b)) for a, b in zip(src, tgt) if int(a) not in busy and int(b) not in busy]
            if keep:
                k = self.k
                pd_, pj_ = np.array([a for a, _ in keep], dtype=np.int64), np.array([b for _, b in keep], dtype=np.int64)
                pair_of = np.full(k, -1, dtype=np.int64)
                role_of = np.zeros(k, dtype=np.int64)
                pair_of[pj_] = np.arange(len(keep)); role_of[pj_] = 0
                pair_of[pd_] = np.arange(len(keep)); role_of[pd_] = 1
                z = lambda: np.zeros(len(keep))
                self._pend = dict(j=pj_, d=pd_, pair_of=pair_of, role_of=role_of, batches=0,
                                  cost_sum=z(), cost_sq=z(), cost_n=z(), gain_sum=z(), gain_n=z(),
                                  pred_gain=np.array([float(self._calib_pred["gain"][a]) for a, _ in keep]),
                                  pred_cost=np.array([float(self._calib_pred["cost"][b]) for _, b in keep]))
                if self.confirm_adapt:
                    dim = self._centroids.shape[1]
                    self._pend.update(ch_n=np.zeros((len(keep), 2)), ch_sum=np.zeros((len(keep), 2, dim)),
                                      ch_sq=np.zeros((len(keep), 2)), fb={}, win_n=np.zeros(k))
                self.n_proposed += len(keep)
        if acc_s:
            return np.array(acc_s, dtype=np.int64), np.array(acc_t, dtype=np.int64)
        return None, None

    def _last_check_before_freeze(self):
        """True if moves will be frozen at the next periodic check (5 batches from now)."""
        if self.timing != "pr":
            return False
        rows = self.recipe["epoch_rows"]
        B = self.recipe["batch_size"]
        done = sum(rows[:self._epoch]) + min(rows[self._epoch], self._epoch_rows + 6 * B)
        return len(rows) > 1 and done >= self._n and sum(rows) - done < self._n

    def _choose_confirm_wait(self):
        """Pipelined propose-then-confirm with an evidence-based wait (see the ngp2cw config)."""
        from faiss.contrib.e_means.base import _scheduled_alpha
        src, tgt = self._choose_net_gain()
        P = self._pend
        acc_s, acc_t = [], []
        keep_idx = []
        cache = self._check_cache
        if P is not None and len(P["j"]) and cache is not None and cache["rate"] is not None:
            rate, delta = cache["rate"], cache["delta"]
            prior = getattr(self, "_ratio_prior", None)
            rho, tau2, s0 = prior if prior is not None else (1.0, 1.0, 1.0)
            alpha = min(max(float(_scheduled_alpha(self.recipe["alpha0"], self._step, self.recipe["total_steps"])),
                            1e-6), 1.0)
            memory = min((2.0 - alpha) / alpha, float(self._step + 1))
            force = self._last_check_before_freeze()
            busy = set()
            for i in range(len(P["j"])):
                j, d = int(P["j"][i]), int(P["d"][i])
                F = float(P["batches"] - P["born"][i])
                target = float(rate[d]) * memory
                if not force and (F <= 0 or P["gain_n"][i] < target):
                    keep_idx.append(i)
                    continue
                gain = P["gain_sum"][i] / max(F, 1.0)
                dl = float(delta[j]) if np.isfinite(delta[j]) and delta[j] > 0 else 0.0
                n = float(P["cost_n"][i])
                if dl > 0:
                    if n > 0:
                        mean = P["cost_sum"][i] / n
                        var = (P["cost_sq"][i] - n * mean * mean) / (n - 1.0) if n >= 2 else s0 * dl * dl
                        v = max(var, 0.0) / (n * dl * dl)
                        r = (tau2 * (mean / dl) + v * rho) / (tau2 + v)
                    else:
                        r = rho
                    cost = float(rate[j]) * max(r, 0.0) * dl
                else:
                    cost = float(rate[j]) * (P["cost_sum"][i] / n if n > 0 else 0.0)
                if gain > self.w * cost and j not in busy and d not in busy:
                    acc_s.append(d); acc_t.append(j); busy.update((j, d))
            self.n_confirmed += len(acc_s)
        # carry the unready pairs, add new proposals from free centres
        reserved = set(acc_s) | set(acc_t)
        carried = []
        if P is not None:
            for i in keep_idx:
                carried.append(i)
                reserved.update((int(P["j"][i]), int(P["d"][i])))
        new = []
        if src is not None:
            for a, b in zip(src, tgt):
                a, b = int(a), int(b)
                if a not in reserved and b not in reserved:
                    new.append((a, b)); reserved.update((a, b))
        if carried or new:
            k = self.k
            nb = 0 if P is None else P["batches"]
            fields = ("cost_sum", "cost_sq", "cost_n", "gain_sum", "gain_n")
            Q = {f: np.concatenate([P[f][carried] if P is not None else np.zeros(0), np.zeros(len(new))]) for f in fields}
            Q["j"] = np.concatenate([P["j"][carried] if P is not None else np.zeros(0, np.int64),
                                     np.array([b for _, b in new], dtype=np.int64)]).astype(np.int64)
            Q["d"] = np.concatenate([P["d"][carried] if P is not None else np.zeros(0, np.int64),
                                     np.array([a for a, _ in new], dtype=np.int64)]).astype(np.int64)
            Q["born"] = np.concatenate([P["born"][carried] if P is not None else np.zeros(0), np.full(len(new), nb)])
            Q["batches"] = nb
            Q["pair_of"] = np.full(k, -1, dtype=np.int64)
            Q["role_of"] = np.zeros(k, dtype=np.int64)
            Q["pair_of"][Q["j"]] = np.arange(len(Q["j"])); Q["role_of"][Q["j"]] = 0
            Q["pair_of"][Q["d"]] = np.arange(len(Q["d"])); Q["role_of"][Q["d"]] = 1
            self._pend = Q
            self.n_proposed += len(new)
        else:
            self._pend = None
        if acc_s:
            return np.array(acc_s, dtype=np.int64), np.array(acc_t, dtype=np.int64)
        return None, None

    def _recentring_value(self, P, i, d, fallbacks):
        """Reduction of the error from one local Lloyd step after the move of pending pair i (sum over the fresh
        window): children move to the mean of the fresh points they capture, fallback centres recentre on the points
        they absorb (Ward). Each term is debiased for fitting the mean on the same points."""
        dc, cc = self._split_positions(np.array([d], dtype=np.int64))
        pos = [dc[0].astype(np.float64), cc[0].astype(np.float64)]
        total = 0.0
        for c in (0, 1):
            n = P["ch_n"][i, c]
            if n < 2:
                continue
            m = P["ch_sum"][i, c] / n
            tr = max((P["ch_sq"][i, c] - n * float(m @ m)) / (n - 1.0), 0.0)        # trace of the covariance
            total += max(n * float((m - pos[c]) @ (m - pos[c])) - tr, 0.0)
        for l, (n, sx, sq) in fallbacks:
            if n < 2:
                continue
            Nl = float(P["win_n"][l])
            m = sx / n
            tr = max((sq - n * float(m @ m)) / (n - 1.0), 0.0)
            cl = self._centroids[l].astype(np.float64)
            w = n / (Nl + n)
            total += max(n * w * float((m - cl) @ (m - cl)) - w * tr, 0.0)
        return total

    def _match_v2(self, cand, donors, cost, gain, nearest, centroids, forbid_nearest=False):
        """Corrected greedy matching (Codex review round 1). A candidate that is also the next donor is skipped
        as a donor only for itself: the donor pointer is not consumed unless the move is accepted, and the
        monotone stop (costs rise, gains fall) applies only when no donor was skipped for this candidate."""
        kept, source, kept_ids, protected, used = [], [], set(), set(), set()
        pos = 0
        for j in cand.tolist():
            while pos < len(donors) and (int(donors[pos]) in kept_ids or int(donors[pos]) in used):
                pos += 1
            q = pos
            nb_j = nearest[j] if forbid_nearest else -1      # the freed centre's main fallback may not be split
            while q < len(donors) and (int(donors[q]) in kept_ids or int(donors[q]) in used or int(donors[q]) == j
                                       or int(donors[q]) == nb_j):
                q += 1
            if q == len(donors):
                if q == pos:
                    break
                continue
            d = int(donors[q])
            if not cost[j] < self.w * gain[d]:
                if q == pos:
                    break                                           # costs rise, gains fall: done
                continue                                            # j itself was the better donor; go on
            nb = nearest[j]
            back = nb in kept_ids and not (centroids[j] == centroids[nb]).all()
            if j in protected or back or j in used:
                continue
            kept.append(j); kept_ids.add(j); protected.add(nb)
            source.append(d); used.add(d)
        if not kept:
            return None, None
        return np.array(source, dtype="int64"), np.array(kept, dtype="int64")

    def _split_positions(self, source):
        """Donor and child positions of a split (shared by _apply_splits and the calibration, so that the
        calibration evaluates exactly the positions the split will use)."""
        if self.split == "axis":
            centroids = self._centroids[source].astype("float64")
            norm = self._axis_scale * np.linalg.norm(self._removal_stats[source, :-1].astype("float64"), axis=1)
            mass = self._usage[source].astype("float64")
            valid = (norm > 0) & (mass > 0) & np.isfinite(norm) & np.isfinite(mass)
            spread = np.sqrt(2.0 * np.where(valid, norm / np.where(valid, mass, 1.0), 0.0) / np.pi)
            offset = spread[:, None] * self._axes[source].astype("float64")
            donor_centroids = (centroids - offset).astype("float32")
            child_centroids = (centroids + offset).astype("float32")
        else:
            centroids = self._centroids[source]
            sign = np.ones(self.d, dtype="float32")
            sign[1::2] = -1.0
            offset = SPLIT_EPSILON * sign
            donor_centroids = centroids * (1.0 - offset)
            child_centroids = centroids * (1.0 + offset)
        _separate_splits(donor_centroids, child_centroids)
        return donor_centroids, child_centroids

    def _apply_splits(self, source, target):
        t0 = time.perf_counter()
        donor_centroids, child_centroids = self._split_positions(source)
        half_usage = self._usage[source] / 2.0
        self._usage[source] = half_usage
        self._centroids[source] = donor_centroids
        self._usage[target] = half_usage
        self._centroids[target] = child_centroids
        if self._removal_stats is not None:
            self._removal_stats[source] /= 2
            self._removal_stats[target] = self._removal_stats[source]
            self._axes[target] = self._axes[source]
        if self._pass_sums is not None and self._moved is not None:
            self._moved[source] = self._moved[target] = True
        self.n_quick_target += int((self._stale[target] > 0.5).sum())
        self.n_quick_donor += int((self._stale[source] > 0.5).sum())
        self._stale[source] = self._stale[target] = 1.0
        self._split_step[source] = self._split_step[target] = self._step
        self._cum_n[source] = self._cum_n[target] = 0.0           # both regions changed: evidence restarts
        if getattr(self, "trigger", "periodic") == "event":
            self._new_since.update(np.asarray(source).tolist())
            self._new_since.update(np.asarray(target).tolist())
        self._inc[:, source] = self._inc[:, target] = 0.0
        half_dn = self._dn[source] / 2.0                            # discounted evidence: halved, exposure kept
        self._dn[source] = half_dn
        self._dn[target] = half_dn
        self._dt[target] = self._dt[source]
        self._dw2[target] = self._dw2[source]
        if getattr(self, "_sample_margins", False):
            for arr in (self._m_n, self._m_sum, self._m_sq, self._p2, self._p4):
                arr[source] = arr[target] = 0.0
        if getattr(self, "_measure_all", False):
            for arr in (self._m_n, self._m_sum, self._m_sq, self._pabs, self._p2):
                arr[source] = arr[target] = 0.0
            self._ema[:, source] = self._ema[:, target] = 0.0
            self._ema_w2[source] = self._ema_w2[target] = 0.0
        self._age[source] = self._age[target] = 0.0
        if self._collect:
            half = self._dist[source] / 2.0
            self._dist[source] = half
            self._dist[target] = half
            self._last_hit[target] = self._step
        self.n_splits += len(source)
        self._invalidate_readout()
        TIMES["apply_splits"] += time.perf_counter() - t0
        return len(source)
