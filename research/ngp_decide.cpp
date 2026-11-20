// The whole per-check revival decision of the posterior net-gain rule in C++ (2026-11-19 night; proof of concept).
// It replaces the vectorized NumPy estimates of ablation_cpp.AblationEmeans._choose_net_gain together with
// ngp_variants / ngp_variants2 _posterior_gains (stats "mean" = the frozen ngp2_full, "medg_pm", "medg_fh"),
// ablation._posterior_rates, _gain_rel_var and _margin_ratios (moderate = False), for the configuration of every
// confirmatory run (ng_post2, matcher v2, exact delta search, cool-down "half", gain_eb "normal", gain_noise
// "fixed_n", periodic checks). The nearest-other-centre search stays in FAISS (its result D, I is an input).
//
// Exactness (NumPy 2.5.5 + OpenBLAS 0.3.34 of the emeans_pr5915 env; checked on the servers, 2026-11-19).
//  * Sums: NumPy's add.reduce on a contiguous float64 array = 0.0 + pairwise_sum over the whole array (8 accumulators,
//    blocks of 128, halving split); NumPy 2.5.5 does not cut 1-D reductions into buffer chunks (verified up to 300,000
//    elements; NumPy 1.26 did cut them at 8,192, so this exactness is tied to the NumPy version). Means, variances,
//    np.average and the row norms therefore round exactly as in the reference.
//  * Element-wise operations follow the reference's order; -ffp-contract=off (no fused multiply-adds); Python max/min
//    tie rules are reproduced; np.maximum returns the second operand on a tie and propagates a NaN first operand
//    (NumPy's optimized x86 loops, SIMD and scalar helper alike; this only ever decides the sign of a zero).
//  * exp and log are the C library's. NumPy uses the same functions on CPUs without AVX-512 (bit-identical there) and
//    its own SIMD exp/log on AVX-512 CPUs (last-bit differences in a few percent of inputs; decisions unaffected in
//    the tests).
//  * medg_fh: the products X'WX, X'Wy, X beta and the least-squares solve call the same OpenBLAS / LAPACK routines,
//    with the same arguments, as NumPy's matmul and np.linalg.lstsq (cblas_dgemm RowMajor/Trans/NoTrans,
//    cblas_dgemv, dgelsd with a workspace query and rcond = eps * p): bit-identical in the same process.
// Diagnostics of the reference (_gd, _ratio_prior) are not produced.
// Build (in the env): g++ -O3 -std=c++17 -ffp-contract=off -fopenmp -fPIC -shared -o ngp_decide.so ngp_decide.cpp
//   -L$CONDA_PREFIX/lib -Wl,-rpath,$CONDA_PREFIX/lib -l:libcblas.so.3 -l:liblapack.so.3
#include "ngp_kernels.cpp"   // ngp_margin_stats and ngp_match_v2, unchanged (the matching is called below)

#include <cfloat>
#include <numeric>

extern "C" {   // OpenBLAS (CBLAS) and LAPACK, LP64: the libraries NumPy itself calls in this env
void cblas_dgemm(int order, int ta, int tb, int m, int n, int k, double alpha, const double* a, int lda,
                 const double* b, int ldb, double beta, double* c, int ldc);
void cblas_dgemv(int order, int trans, int m, int n, double alpha, const double* a, int lda, const double* x, int incx,
                 double beta, double* y, int incy);
void dgelsd_(const int* m, const int* n, const int* nrhs, double* a, const int* lda, double* b, const int* ldb,
             double* s, const double* rcond, int* rank, double* work, const int* lwork, int* iwork, int* info);
}

namespace {

double pairwise(const double* a, int64_t n) {          // NumPy's pairwise_sum (numpy/_core/src/umath/loops_utils.h)
    if (n < 8) {
        double r = 0.0;
        for (int64_t i = 0; i < n; i++) r += a[i];
        return r;
    }
    if (n <= 128) {
        double r[8];
        for (int j = 0; j < 8; j++) r[j] = a[j];
        int64_t i;
        for (i = 8; i < n - (n % 8); i += 8)
            for (int j = 0; j < 8; j++) r[j] += a[i + j];
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; i++) res += a[i];
        return res;
    }
    int64_t n2 = n / 2;
    n2 -= n2 % 8;
    return pairwise(a, n2) + pairwise(a + n2, n - n2);
}
double npsum(const std::vector<double>& v) { return 0.0 + pairwise(v.data(), (int64_t)v.size()); }
double npmean(const std::vector<double>& v) { return npsum(v) / (double)v.size(); }
double npvar(const std::vector<double>& v) {           // np.var, ddof 0: sum((x - sum(x)/n)^2) / n
    const double mu = npsum(v) / (double)v.size();
    std::vector<double> t(v.size());
    for (size_t i = 0; i < v.size(); i++) { const double e = v[i] - mu; t[i] = e * e; }
    return npsum(t) / (double)v.size();
}
double npaverage(const std::vector<double>& a, const std::vector<double>& w) {   // np.average(a, weights=w)
    std::vector<double> t(a.size());
    for (size_t i = 0; i < a.size(); i++) t[i] = a[i] * w[i];
    const double scl = npsum(w);
    return npsum(t) / scl;
}
inline double pymax(double a, double b) { return (b > a) ? b : a; }          // Python max(a, b)
inline double pymin(double a, double b) { return (b < a) ? b : a; }          // Python min(a, b)
inline double npmaximum(double a, double b) { return std::isnan(a) ? a : (a > b ? a : b); }   // np.maximum (SIMD)

// ablation._posterior_rates
void posterior_rates(int64_t k, const double* age, const double* cum, double* rate) {
    std::vector<double> r;
    std::vector<double> ra;
    for (int64_t i = 0; i < k; i++)
        if (age[i] >= 1) { r.push_back(cum[i] / age[i]); ra.push_back(age[i]); }
    if (r.size() < 2) {
        for (int64_t i = 0; i < k; i++) rate[i] = (age[i] > 0) ? cum[i] / npmaximum(age[i], 1.0) : 0.0;
        return;
    }
    const double m = npmean(r);
    if (!(m > 0)) {
        for (int64_t i = 0; i < k; i++) rate[i] = 0.0;
        return;
    }
    std::vector<double> q(r.size());
    for (size_t i = 0; i < r.size(); i++) q[i] = r[i] / ra[i];
    double v = npvar(r) - npmean(q);
    v = pymax(v, 1e-6 * m * m);
    const double a = m * m / v, b = m / v;
    for (int64_t i = 0; i < k; i++) rate[i] = (a + cum[i]) / (b + age[i]);
}

// ablation._gain_rel_var (ng_post2: sampled margins on; gain_noise "fixed_n": kappa - 1, Gaussian 2)
double gain_rel_var(int64_t k, const double* m_n, const double* p2, const double* p4) {
    std::vector<double> kap, n;
    for (int64_t i = 0; i < k; i++)
        if (m_n[i] >= 2 && p2[i] > 0) {
            const double a = p4[i] / m_n[i], b = p2[i] / m_n[i];
            kap.push_back(a / (b * b));
            n.push_back(m_n[i]);
        }
    if (kap.empty()) return 2.0;
    return pymax(npaverage(kap, n) - 1.0, 1e-6);
}

// ngp_variants.paule_mandel
void paule_mandel(const std::vector<double>& y, const std::vector<double>& v, int iters, double& mu, double& tau2) {
    const size_t m = y.size();
    std::vector<double> w(m), t1(m);
    auto q = [&](double t, double& mu_out) {
        for (size_t i = 0; i < m; i++) w[i] = 1.0 / (t + v[i]);
        for (size_t i = 0; i < m; i++) t1[i] = w[i] * y[i];
        const double mu_ = npsum(t1) / npsum(w);
        for (size_t i = 0; i < m; i++) { const double e = y[i] - mu_; t1[i] = w[i] * (e * e); }
        mu_out = mu_;
        return npsum(t1);
    };
    const double df = (double)m - 1.0;
    double mu0;
    const double q0 = q(0.0, mu0);
    if (q0 <= df) { mu = mu0; tau2 = 0.0; return; }
    double lo = -30.0, hi = std::log(pymax(npvar(y) * 4.0, 1e-12)) + 1.0, tmp;
    while (q(std::exp(hi), tmp) > df && hi < 60.0) hi += 2.0;
    for (int it = 0; it < iters; it++) {
        const double mid = 0.5 * (lo + hi);
        if (q(std::exp(mid), tmp) > df) lo = mid; else hi = mid;
    }
    tau2 = std::exp(0.5 * (lo + hi));
    q(tau2, mu);
}

// np.linalg.lstsq(A, b, rcond=None) for a p x p A (row-major) and one right-hand side, as NumPy calls LAPACK:
// column-major copies, workspace query, dgelsd with rcond = eps * max(m, n). Returns LAPACK's info (0 = success).
int lstsq_gelsd(int p, const double* A, const double* b, double* x) {
    std::vector<double> af(p * p), bf(p), s(p);
    for (int r = 0; r < p; r++)
        for (int c = 0; c < p; c++) af[c * p + r] = A[r * p + c];
    for (int r = 0; r < p; r++) bf[r] = b[r];
    const int one = 1, mlw = -1;
    const double rcond = DBL_EPSILON * p;
    int rank = 0, info = 0, iq = 0;
    double wq = 0.0;
    dgelsd_(&p, &p, &one, af.data(), &p, bf.data(), &p, s.data(), &rcond, &rank, &wq, &mlw, &iq, &info);
    if (info != 0) return info;
    const int lwork = (int)wq;
    std::vector<double> work(std::max(lwork, 1));
    std::vector<int> iwork(std::max(iq, 1));
    dgelsd_(&p, &p, &one, af.data(), &p, bf.data(), &p, s.data(), &rcond, &rank, work.data(), &lwork, iwork.data(),
            &info);
    for (int r = 0; r < p; r++) x[r] = bf[r];
    return info;
}

// ngp_variants2.fay_herriot: (beta, tau^2) of y = X beta + u + e, WLS beta with weights 1 / (tau^2 + v),
// tau^2 by the Paule-Mandel equation sum w (y - X beta)^2 = m - p (bisection on log tau^2, as the reference).
// X: m x p row-major (np.column_stack). The products are NumPy's BLAS calls: Xw.T @ X -> dgemm (RowMajor, Trans,
// NoTrans), Xw.T @ y -> dgemv (RowMajor, Trans), X @ beta -> dgemv (RowMajor, NoTrans). Returns 0, or LAPACK's info.
int fay_herriot(const std::vector<double>& y, const std::vector<double>& v, const std::vector<double>& X, int p,
                int iters, double* beta, double& tau2) {
    const int m = (int)y.size();
    std::vector<double> w(m), Xw((size_t)m * p), pred(m), t(m);
    int status = 0;
    auto fit = [&](double tt, double* bt) {
        for (int i = 0; i < m; i++) w[i] = 1.0 / (tt + v[i]);
        for (int i = 0; i < m; i++)
            for (int a = 0; a < p; a++) Xw[(size_t)i * p + a] = X[(size_t)i * p + a] * w[i];
        double A[9], b[3];
        cblas_dgemm(101, 112, 111, p, p, m, 1.0, Xw.data(), p, X.data(), p, 0.0, A, p);
        for (int a = 0; a < p; a++)                        // + 1e-12 * np.eye(p), element by element
            for (int c = 0; c < p; c++) A[a * p + c] = A[a * p + c] + 1e-12 * (a == c ? 1.0 : 0.0);
        cblas_dgemv(101, 112, m, p, 1.0, Xw.data(), p, y.data(), 1, 0.0, b, 1);
        const int info = lstsq_gelsd(p, A, b, bt);
        if (info != 0 && status == 0) status = info;
        cblas_dgemv(101, 111, m, p, 1.0, X.data(), p, bt, 1, 0.0, pred.data(), 1);
        for (int i = 0; i < m; i++) { const double r = y[i] - pred[i]; t[i] = w[i] * r * r; }
        return npsum(t);
    };
    const double df = (double)m - (double)p;
    double b0[3], tmp[3];
    const double q0 = fit(0.0, b0);
    if (q0 <= df) { for (int a = 0; a < p; a++) beta[a] = b0[a]; tau2 = 0.0; return status; }
    double lo = -30.0, hi = std::log(pymax(npvar(y) * 4.0, 1e-12)) + 1.0;
    while (fit(std::exp(hi), tmp) > df && hi < 60.0) hi += 2.0;
    for (int it = 0; it < iters; it++) {
        const double mid = 0.5 * (lo + hi);
        if (fit(std::exp(mid), tmp) > df) lo = mid; else hi = mid;
    }
    tau2 = std::exp(0.5 * (lo + hi));
    fit(tau2, beta);
    return status;
}

// ablation._posterior_gains (stat 0, the frozen rule), ngp_variants (stat 1, medg_pm), ngp_variants2 (stat 2, medg_fh)
int posterior_gains(int64_t k, const double* raw, const double* rate, double decay_prod, double alpha, double step,
                    double kv, int stat, const double* prev_delta, double* out) {
    const double den = pymax(1.0 - decay_prod, 1e-12);
    const double window = pymin((2.0 - alpha) / alpha, step + 1.0);
    std::vector<double> g(k), s2(k);
    for (int64_t i = 0; i < k; i++) {
        g[i] = raw[i] / den;
        s2[i] = kv / npmaximum(rate[i] * window, 1e-3);
        out[i] = 0.0;
    }
    std::vector<int64_t> pos;
    for (int64_t i = 0; i < k; i++) if (g[i] > 0) pos.push_back(i);
    const size_t np_ = pos.size();
    if (np_ < (stat == 2 ? 4u : 2u)) {
        for (int64_t i : pos) out[i] = g[i];
        return 0;
    }
    std::vector<double> lg(np_), s2p(np_);
    for (size_t i = 0; i < np_; i++) { lg[i] = std::log(g[pos[i]]); s2p[i] = s2[pos[i]]; }
    if (stat == 2) {
        const int p = prev_delta ? 3 : 2;
        std::vector<double> X(np_ * p);                    // np.column_stack(cols): row-major, m x p
        for (size_t i = 0; i < np_; i++) {
            X[i * p] = 1.0;
            X[i * p + 1] = std::log(npmaximum(rate[pos[i]], 1e-6));
            if (p == 3) X[i * p + 2] = std::log(npmaximum(prev_delta[pos[i]], 1e-12));
        }
        double beta[3], tau2;
        const int st = fay_herriot(lg, s2p, X, p, 60, beta, tau2);
        if (st != 0) return st;
        tau2 = pymax(tau2, 1e-6);
        std::vector<double> pred(np_);
        cblas_dgemv(101, 111, (int)np_, p, 1.0, X.data(), p, beta, 1, 0.0, pred.data(), 1);   // X @ beta
        for (size_t i = 0; i < np_; i++)
            out[pos[i]] = std::exp((tau2 * lg[i] + s2p[i] * pred[i]) / (tau2 + s2p[i]));
        return 0;
    }
    double mu, tau2;
    if (stat == 1) {
        paule_mandel(lg, s2p, 100, mu, tau2);
        tau2 = pymax(tau2, 1e-6);
    } else {
        mu = npmean(lg);
        tau2 = pymax(npvar(lg) - npmean(s2p), 1e-6);
    }
    for (size_t i = 0; i < np_; i++) {
        const double pm = (tau2 * lg[i] + s2p[i] * mu) / (tau2 + s2p[i]);
        if (stat == 1) {
            out[pos[i]] = std::exp(pm);
        } else {
            const double pv = tau2 * s2p[i] / (tau2 + s2p[i]);
            out[pos[i]] = std::exp(pm + pv / 2.0);
        }
    }
    return 0;
}

// ablation._margin_ratios with moderate = False; returns rho
double margin_ratios(int64_t k, const double* delta, const double* m_n, const double* m_sum, const double* m_sq,
                     double* r_hat) {
    std::vector<int64_t> ok;
    for (int64_t i = 0; i < k; i++) {
        r_hat[i] = 1.0;
        if (m_n[i] > 0 && std::isfinite(delta[i]) && delta[i] > 0) ok.push_back(i);
    }
    if (ok.empty()) return 1.0;
    const size_t m = ok.size();
    std::vector<double> n(m), r(m);
    for (size_t i = 0; i < m; i++) {
        n[i] = m_n[ok[i]];
        r[i] = (m_sum[ok[i]] / n[i]) / delta[ok[i]];
    }
    const double rho = npaverage(r, n);
    if (m < 2) {
        const double v = pymax(rho, 0.0);
        for (int64_t i = 0; i < k; i++) r_hat[i] = v;
        return rho;
    }
    double sigma2;
    std::vector<double> within, nm1;
    for (size_t i = 0; i < m; i++)
        if (n[i] >= 2) {
            const int64_t j = ok[i];
            const double dl = delta[j];
            within.push_back((m_sq[j] - m_sum[j] * m_sum[j] / n[i]) / (dl * dl));
            nm1.push_back(n[i] - 1.0);
        }
    if (!within.empty())
        sigma2 = pymax(npsum(within) / npsum(nm1), 1e-12);
    else
        sigma2 = pymax(npvar(r), 1e-12);
    std::vector<double> sn(m);
    for (size_t i = 0; i < m; i++) sn[i] = sigma2 / n[i];
    double tau2 = npvar(r) - npmean(sn);
    tau2 = pymax(pymax(tau2, 1e-6 * rho * rho), 1e-12);          // Python max(a, b, c)
    for (int64_t i = 0; i < k; i++) r_hat[i] = rho;
    for (size_t i = 0; i < m; i++) r_hat[ok[i]] = (tau2 * r[i] + sn[i] * rho) / (tau2 + sn[i]);
    for (int64_t i = 0; i < k; i++) r_hat[i] = npmaximum(r_hat[i], 0.0);
    return rho;
}

}  // namespace

extern "C" {

// One revival check. Inputs: k, d; usage (k, float64); removal_stats (k x ncol float32; the first d columns are U /
// axis_scale), gscale = (2/pi) * axis_scale (computed by the caller as the reference does); stale, age, cum (k);
// decay_prod; alpha (the clamped scheduled step) and step; m_n, m_sum, m_sq, p2, p4 (k); D, I (k x 2: faiss.knn of
// the centroids against themselves); stat (0 mean, 1 medg_pm, 2 medg_fh); prev_delta (k or NULL; medg_fh); w;
// readout (k x d float32, the pair guard's duplicate test); forbid_nearest.
// Outputs: rate, gain, delta, cost (k); rho; n_cand (candidates passed to the matching); src, kept (capacity k).
// Returns the number of accepted pairs, -1 on a bad id, or -2 if LAPACK's dgelsd fails (medg_fh).
int64_t ngp_decide(int64_t k, int64_t d, const double* usage, const float* removal_stats, int64_t ncol, double gscale,
                   const double* stale, const double* age, const double* cum, double decay_prod, double alpha,
                   double step, const double* m_n, const double* m_sum, const double* m_sq, const double* p2,
                   const double* p4, const float* D, const int64_t* I, int stat, const double* prev_delta, double w,
                   const float* readout, int forbid_nearest, double* rate, double* gain, double* delta, double* cost,
                   double* rho_out, int64_t* n_cand, int64_t* src, int64_t* kept) {
    *n_cand = 0;
    std::vector<double> raw(k), row(d);
    for (int64_t i = 0; i < k; i++) {                        // (2/pi) * axis_scale * ||U_i|| (np.linalg.norm, axis 1)
        const float* rs = removal_stats + i * ncol;
        for (int64_t j = 0; j < d; j++) { const double x = (double)rs[j]; row[j] = x * x; }
        raw[i] = gscale * std::sqrt(0.0 + pairwise(row.data(), d));
    }
    posterior_rates(k, age, cum, rate);
    const double kv = gain_rel_var(k, m_n, p2, p4);
    if (posterior_gains(k, raw.data(), rate, decay_prod, alpha, step, kv, stat, prev_delta, gain) != 0)
        return -2;                                       // LAPACK failure (NumPy would raise LinAlgError)
    std::vector<int64_t> nn(k);
    for (int64_t i = 0; i < k; i++) {
        const bool self0 = I[2 * i] == i;
        int64_t o = self0 ? I[2 * i + 1] : I[2 * i];
        double dl = (double)(self0 ? D[2 * i + 1] : D[2 * i]);
        if (o < 0 || !std::isfinite(dl)) dl = DBL_MAX / 4;
        if (o < 0) o = i;
        if (o >= k) return -1;
        nn[i] = o;
        delta[i] = dl;
    }
    std::vector<double> ratio(k);
    *rho_out = margin_ratios(k, delta, m_n, m_sum, m_sq, ratio.data());
    for (int64_t i = 0; i < k; i++) cost[i] = (rate[i] * npmaximum(delta[i], 0.0)) * ratio[i];
    std::vector<int64_t> donors;
    for (int64_t i = 0; i < k; i++) if (usage[i] > 0 && !(stale[i] > 0.5)) donors.push_back(i);
    if (donors.empty()) return 0;
    std::stable_sort(donors.begin(), donors.end(), [&](int64_t a, int64_t b) { return -gain[a] < -gain[b]; });
    const double top = w * gain[donors[0]];
    std::vector<int64_t> cand;
    for (int64_t i = 0; i < k; i++) if (cost[i] < top && !(stale[i] > 0.5)) cand.push_back(i);
    if (cand.empty()) return 0;
    std::stable_sort(cand.begin(), cand.end(), [&](int64_t a, int64_t b) { return cost[a] < cost[b]; });
    *n_cand = (int64_t)cand.size();
    return ngp_match_v2((int64_t)cand.size(), cand.data(), (int64_t)donors.size(), donors.data(), cost, gain, w,
                        nn.data(), k, d, readout, forbid_nearest, src, kept);
}

}  // extern "C"
