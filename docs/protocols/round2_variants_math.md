# The two variants of `ngp2_full` tested in the second confirmatory round: mathematical specification

Written 2026-11-19, before any run of the second round. It is the reference against which the code
(`code/experiments/pr5915/ngp_variants.py`, `ngp_variants2.py`) is reviewed. The frozen rule `ngp2_full` is specified in
a working note (rev 2b, with a paragraph on boundary cases added before this round; not released, the paper's Appendix A gives the complete specification) and `protocol_ngp.md` Section 1.
The variants change **only the gain of splitting a donor** (Section 2 below). The removal cost, the matching, the
cool-down, the trigger and everything else are those of `ngp2_full`.

## 1. What the frozen rule does with the gain (for reference)

At a revival check (every 5 batches while moves are allowed), for every centre d:

- **Raw gain.** U_d is the EMA of the per-batch split statistic on the split axis; the raw gain is (2/π)·|U_d|, divided
  by the EMA's start-up weight 1 − Π(1 − α) (floored at 10⁻¹²): g_d.
- **Noise variance on the log scale.** s²_d = (κ̄ − 1) / max(λ̂_d · W, 10⁻³), where
  - κ̄ is the pooled kurtosis of the projections on the split axes (clusters with ≥ 2 sampled rows and positive spread,
    weighted by their rows); κ̄ − 1 is floored at 10⁻⁶; it is 2 (Gaussian) when no cluster qualifies;
  - λ̂_d is the posterior mean of d's points per batch (Gamma–Poisson, moment-fitted prior; `_posterior_rates`);
  - W = min((2 − α_t)/α_t, t + 1) is the EMA's memory in batches, with α_t clipped to [10⁻⁶, 1] and t the batch index.
- **Population.** Only centres with g_d > 0 enter (set P, m = |P|). A centre with g_d = 0 keeps gain 0. If m < 2, the
  gains are the raw g_d (no shrinkage).
- **Prior on y_d = log g_d (method of moments, unweighted).** μ = mean(y), τ² = max(var(y) − mean(s²), 10⁻⁶) over P.
- **Posterior of y_d.** Normal–normal: mean m_d = (τ² y_d + s²_d μ)/(τ² + s²_d), variance v_d = τ² s²_d/(τ² + s²_d).
- **Gain used.** The posterior mean of the gain, G_d = exp(m_d + v_d/2).

The removal cost of centre j is C_j = λ̂_j · r̃_j · δ_j (posterior mean rate × shrunk margin ratio × squared distance to
the nearest other centre). A move (remove j, split d) is accepted, greedily, while C_j < w · G_d (w = 1).

## 2. The variants

### 2.1 `ngp2_full_medg_pm` (primary candidate): posterior median, Paule–Mandel prior

Two changes to Section 1, nothing else:

1. **Prior by Paule–Mandel** instead of moments. For t ≥ 0 let w_d(t) = 1/(t + s²_d), μ(t) = Σ w_d y_d / Σ w_d and
   Q(t) = Σ_{d∈P} w_d(t) (y_d − μ(t))². Q is continuous and decreasing in t.
   - If Q(0) ≤ m − 1: τ² = 0 and μ = μ(0).
   - Otherwise τ² is the root of Q(t) = m − 1, found by bisection on log t: lower end log t = −30; upper end
     h = log(max(4 var(y), 10⁻¹²)) + 1, raised by 2 until Q(eʰ) ≤ m − 1 or h ≥ 60; 100 halvings; τ² = exp(midpoint of
     the final bracket); μ = μ(τ²).
   - The bracket's lower end is e⁻³⁰ ≈ 9.4·10⁻¹⁴: if the root lies below it, the bisection returns about e⁻³⁰ instead
     of the root (Codex's example: y = (0, 10⁻⁷), s² = (10⁻¹⁸, 10⁻¹⁶)). The floor below makes τ² the same in both cases;
     only μ = μ(t) is computed at the returned t, and it differs from μ(root) only when some s²_d is not much larger than
     e⁻³⁰.
   - Then τ² = max(τ², 10⁻⁶) (the same floor as the frozen rule).
   - Why: the moment estimator weights all centres equally, and mean(s²) is dominated by centres with almost no evidence
     (n_eff ≪ 1), which drives τ² to its floor (KDD 256, seed 13) or, with decayed near-zero gains, inflates it
     (T = 5). Paule–Mandel weights each centre by its precision, so centres without evidence barely count.
2. **Posterior median** instead of the posterior mean: G_d = exp(m_d) (the median of the log-normal posterior of the
   gain), with m_d as in Section 1 using this (μ, τ²).
   - Why: exp(m + v/2) credits a centre without evidence (s²_d large, so v_d ≈ τ²) with exp(μ + τ²/2), which is huge
     when τ² is large; this made empty centres the best donors at T = 5 (runaway). The median does not grow with v_d.
   - Interpretation: exp(m_d) is the value the gain exceeds with posterior probability ½. The cost is still a posterior
     mean, so the decision C_j < G_d is "the gain is more likely than not to exceed the estimated cost"; it is not a
     statement about the joint posterior of gain and cost.

Boundary cases as in the frozen rule: g_d = 0 gives G_d = 0; m < 2 gives G_d = g_d.

### 2.2 `ngp2_full_medg_fh` (reported alternative): posterior median, Fay–Herriot prior with covariates

The prior mean of y_d depends on two well-measured properties of the centre instead of being one global μ:

- y_d = x_dᵀβ + u_d + e_d, u_d ~ N(0, τ²), e_d ~ N(0, s²_d), over d ∈ P.
- x_d = (1, log max(λ̂_d, 10⁻⁶), log max(δ⁻_d, 10⁻¹²)), where δ⁻_d is d's squared distance to the nearest other centre
  **at the previous check** (the current check's δ is computed after the gains). At the first check, or if no previous
  δ exists, x_d = (1, log max(λ̂_d, 10⁻⁶)). Note that δ⁻ is stale for centres moved since the previous check (a split's
  donor and freed centre, and EMA drift); it is used as a covariate only.
- **Estimation.** For t ≥ 0, β(t) is the weighted least-squares fit with weights w_d(t) = 1/(t + s²_d), computed as the
  least-squares solution of (XᵀWX + 10⁻¹² I) β = XᵀW y; Q(t) = Σ w_d(t) (y_d − x_dᵀβ(t))².
  - If Q(0) ≤ m − p (p = number of columns): τ² = 0, β = β(0).
  - Otherwise τ² solves Q(t) = m − p by the same bisection as in 2.1 but with 60 halvings; β = β(τ²). The same
    lower-end caveat as in 2.1 applies.
  - τ² = max(τ², 10⁻⁶).
  - Rank policy: p is the number of columns of X whatever its rank, and the 10⁻¹² ridge makes β unique when X is rank
    deficient (for example, all centres with the same rate). With a rank-deficient X this is a regularized working
    estimator, not the classical one, which would use m − rank(X) (Codex's example: four rows, three identical columns,
    s² = 1, y = (−1, −1, 1, 1) gives τ² = 3 here and 1/3 with the rank).
- **Posterior median.** G_d = exp((τ² y_d + s²_d x_dᵀβ)/(τ² + s²_d)).
- **Boundary cases.** g_d = 0 gives G_d = 0. If m < 4 the gains are the raw g_d (no shrinkage).

## 3. Implementation notes (what the review should check)

- Both variants are implemented by overriding `_posterior_gains(raw, rate)` (`ngp_variants.AblationEmeans`,
  `ngp_variants2.AblationEmeans`); `raw` is (2/π)·|U_d| as computed by the caller (`_choose_net_gain`, reference or C++
  version) and `rate` is λ̂ from `_posterior_rates()`.
- `ngp_variants.AblationEmeans._posterior_gains` first calls the frozen computation (its result is discarded for the
  variants when m ≥ 2, and kept when m < 2, which equals g_d on P and 0 elsewhere).
- g, α_t, W and s² must be computed exactly as in the frozen rule (`ablation.py`, `_posterior_gains` and
  `_gain_rel_var` with `gain_noise="fixed_n"`, `gain_eb="normal"`).
- δ⁻ is read from `self._check_cache["delta"]`, which `_choose_net_gain` sets after computing the gains at each check;
  at the first check `_check_cache` does not exist yet.
- The diagnostic dict `self._gd` has no effect on the rule.
