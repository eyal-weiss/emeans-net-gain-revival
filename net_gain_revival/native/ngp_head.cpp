// C++ kernels of net_gain_revival for the e-means driver of PR #5915 at its head (79440bc): the v1.0.0 kernels
// (ngp_decide.cpp, which includes ngp_kernels.cpp; both unchanged) plus the per-point sums from the driver's own
// two-nearest search. Build: native/build.sh (-ffp-contract=off: no fused multiply-adds).
#include "ngp_decide.cpp"

extern "C" {

// Per-point sums from the driver's assignment, as NetGainRevivalEmeans._assign_rows computes them in NumPy:
//   margin = max(second - dist, 0) in float64, 0 if not finite;  p = <x - c_label, axis_label> (float64);
//   m_n += count, m_sum += sum margin, m_sq += sum margin^2, p2 += sum p^2, p4 += sum p^4   (per label).
// x: n x d float32; dist, second: n float32 (nearest and second-nearest squared distances); labels: n int64 (the
// driver's labels, ties to the lowest index); C, A: k x d float32 (the centroids the rows were assigned to, the axes).
// Row values are computed in parallel; the sums are accumulated in row order into per-batch buffers that are then
// added to the totals (the order of numpy.bincount followed by +=), so the result does not depend on the thread
// count. Returns 0, or -1 if a label is out of range (nothing is added then).
int ngp_margin_stats_labeled(int64_t n, int64_t d, int64_t k, const float* x, const float* dist, const float* second,
                             const int64_t* labels, const float* C, const float* A, double* m_n, double* m_sum,
                             double* m_sq, double* p2, double* p4, int nthreads) {
    std::vector<double> margin(n), proj(n);
    int bad = 0;
#pragma omp parallel for num_threads(nthreads) schedule(static) reduction(| : bad)
    for (int64_t i = 0; i < n; i++) {
        const int64_t lab = labels[i];
        if (lab < 0 || lab >= k) {
            bad = 1;
            margin[i] = proj[i] = 0.0;
            continue;
        }
        double m = (double)second[i] - (double)dist[i];
        if (!std::isfinite(m)) m = 0.0;
        else if (m < 0.0) m = 0.0;
        margin[i] = m;
        const float* xi = x + i * d;
        const float* c = C + lab * d;
        const float* a = A + lab * d;
        double s = 0.0;
        for (int64_t j = 0; j < d; j++) s += ((double)xi[j] - (double)c[j]) * (double)a[j];
        proj[i] = s;
    }
    if (bad) return -1;
    std::vector<double> bn(k, 0.0), bs(k, 0.0), bq(k, 0.0), b2(k, 0.0), b4(k, 0.0);
    for (int64_t i = 0; i < n; i++) {
        const int64_t l = labels[i];
        const double m = margin[i], p = proj[i];
        bn[l] += 1.0;
        bs[l] += m;
        bq[l] += m * m;
        b2[l] += p * p;
        b4[l] += std::pow(p, 4.0);
    }
    for (int64_t l = 0; l < k; l++) {
        m_n[l] += bn[l];
        m_sum[l] += bs[l];
        m_sq[l] += bq[l];
        p2[l] += b2[l];
        p4[l] += b4[l];
    }
    return 0;
}

}  // extern "C"
