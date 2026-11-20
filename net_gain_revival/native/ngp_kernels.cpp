// C++ kernels for the posterior net-gain rule (ngp2 / ngp2_full; protocol_ngp.md section 2).
// Build: g++ -O3 -ffp-contract=off -fopenmp -fPIC -shared -o ngp_kernels.so ngp_kernels.cpp
// (-ffp-contract=off: no fused multiply-adds, so each product and sum is rounded as in the NumPy reference.)
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

extern "C" {

// Per-row statistics of the two-nearest search, as in AblationEmeans._assign_rows_sampled (reference):
//   label  = lowest index on a distance tie, else the nearest;  dist = nearest distance;
//   margin = max(second - nearest, 0), 0 if not finite;  p = <x - c_label, axis_label> (float64);
//   m_n += count, m_sum += sum margin, m_sq += sum margin^2, p2 += sum p^2, p4 += sum p^4   (per label).
// x: n x d float32, D: n x 2 float32, I: n x 2 int64 (faiss.knn(x, C, 2)), C, A: k x d float32 (centroids, axes).
// Row values are computed in parallel; the sums are accumulated in row order into per-batch buffers that are then
// added to the totals (the order of numpy.bincount followed by +=), so the result does not depend on the thread
// count. Returns 0, or -1 if a label is out of range.
int ngp_margin_stats(int64_t n, int64_t d, int64_t k, const float* x, const float* D, const int64_t* I,
                     const float* C, const float* A, int64_t* labels, float* dist, double* m_n, double* m_sum,
                     double* m_sq, double* p2, double* p4, int nthreads) {
    std::vector<double> margin(n), proj(n);
    int bad = 0;
#pragma omp parallel for num_threads(nthreads) schedule(static) reduction(| : bad)
    for (int64_t i = 0; i < n; i++) {
        const int64_t i0 = I[2 * i], i1 = I[2 * i + 1];
        const float d0 = D[2 * i], d1 = D[2 * i + 1];
        const int64_t lab = (d0 == d1) ? std::min(i0, i1) : i0;
        labels[i] = lab;
        dist[i] = d0;
        if (lab < 0 || lab >= k) {
            bad = 1;
            margin[i] = proj[i] = 0.0;
            continue;
        }
        const double other = (i0 == lab) ? (double)d1 : (double)d0;
        double m = other - (double)d0;
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

// Greedy matching (AblationEmeans._match_v2, reference): candidates in increasing cost, donors in decreasing gain.
// cand: m ids, donors: nd ids, cost / gain: k float64, nearest: k int64 (nearest other centre of each id; only the
// candidates' entries are read), centroids: k x d float32 (duplicate test of the pair guard).
// Writes the accepted (donor, freed centre) pairs to source_out / kept_out (capacity m) and returns their number,
// or -1 if an id is out of range.
int64_t ngp_match_v2(int64_t m, const int64_t* cand, int64_t nd, const int64_t* donors, const double* cost,
                     const double* gain, double w, const int64_t* nearest, int64_t k, int64_t d,
                     const float* centroids, int forbid_nearest, int64_t* source_out, int64_t* kept_out) {
    std::vector<char> kept(k, 0), prot(k, 0), used(k, 0);
    for (int64_t t = 0; t < m; t++)
        if (cand[t] < 0 || cand[t] >= k || nearest[cand[t]] < 0 || nearest[cand[t]] >= k) return -1;
    for (int64_t t = 0; t < nd; t++)
        if (donors[t] < 0 || donors[t] >= k) return -1;
    int64_t pos = 0, out = 0;
    for (int64_t t = 0; t < m; t++) {
        const int64_t j = cand[t];
        while (pos < nd && (kept[donors[pos]] || used[donors[pos]])) pos++;
        int64_t q = pos;
        const int64_t nb_j = forbid_nearest ? nearest[j] : -1;
        while (q < nd && (kept[donors[q]] || used[donors[q]] || donors[q] == j || donors[q] == nb_j)) q++;
        if (q == nd) {
            if (q == pos) break;
            continue;
        }
        const int64_t dn = donors[q];
        if (!(cost[j] < w * gain[dn])) {
            if (q == pos) break;  // costs rise, gains fall: done
            continue;             // j itself was the better donor; go on
        }
        const int64_t nb = nearest[j];
        bool back = false;
        if (kept[nb]) {
            bool same = true;
            const float* cj = centroids + j * d;
            const float* cn = centroids + nb * d;
            for (int64_t e = 0; e < d; e++)
                if (!(cj[e] == cn[e])) { same = false; break; }
            back = !same;
        }
        if (prot[j] || back || used[j]) continue;
        kept[j] = 1;
        prot[nb] = 1;
        used[dn] = 1;
        source_out[out] = dn;
        kept_out[out] = j;
        out++;
    }
    return out;
}

}  // extern "C"
