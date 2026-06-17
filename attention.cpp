#include "attention.h"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

// ---------------- naive attention ----------------
std::vector<double> naive_attention(const double* __restrict__ Q,
                                   const double* __restrict__ K,
                                   const double* __restrict__ V,
                                   size_t N, size_t d, double scale)
{
    std::vector<double> S(N * N, 0.f);
    std::vector<double> O(N * d, 0.f);

    // S = Q (Nxd) * K^T (dxN) = NxN
    block_dgemm(false, true,
                N, N, d, scale,
                Q, d, K, d, 0.f, S.data(), N);
    // naive_dgemm(false, false,
    //             N, N, d, scale,
    //             Q, d, K, N, 0.f, S.data(), N);

    // P = softmax_row(S)
    for (size_t i = 0; i < N; ++i) {
        double m = row_max(&S[i*N], N);
        double sumExp = 0.f;
        for (size_t j = 0; j < N; ++j) {
            S[i*N + j] = std::exp(S[i*N + j] - m); // P_{ij} = \exp(x_j - m(x))
            sumExp += S[i*N + j]; // l(x) = \sum_j \exp(x_j - m(x))
        }
        for (size_t j = 0; j < N; ++j) S[i*N + j] /= sumExp; // P_{ij} = f(x) / l(x)
    }

    // O (Nxd) = P (NxN) * V (Nxd)
    block_dgemm(false, false,
                N, d, N, 1.f,
                S.data(), N, V, d, 0.f, O.data(), d);
    // naive_dgemm(false, false,
    //             N, d, N, 1.f,
    //             S.data(), N, V, d, 0.f, O.data(), d);
    return O;
}

// --------------- FlashAttention-v2 ---------------
std::vector<double> flash_attention_v2(const double* __restrict__ Q,
                                      const double* __restrict__ K,
                                      const double* __restrict__ V,
                                      size_t N, size_t d, size_t M_bytes, double scale)
{
    const size_t M = M_bytes / sizeof(double);
    auto [Br, Bc] = compute_block_sizes_from_M(N, M, d);

    // outputs and running stats
    std::vector<double> O(N * d, 0.f);
    std::vector<double> m(N, -std::numeric_limits<double>::infinity());
    std::vector<double> l(N, 0.f);

    // preallocate buffers with max sizes
    std::vector<double> Sij(Br * Bc, 0.f);
    std::vector<double> PV(Br * d, 0.f);

    // compute Tr and Tc by ceil division
    const size_t Tr = (N + Br - 1) / Br;
    const size_t Tc = (N + Bc - 1) / Bc;

    for (size_t i = 0; i < Tr; ++i) {
        const size_t i0 = i * Br;
        const size_t br = std::min(Br, N - i0);
        std::fill(PV.begin(), PV.begin() + br * d, 0.f);

        for (size_t j = 0; j < Tc; ++j) {
            const size_t j0 = j * Bc;
            const size_t bc = std::min(Bc, N - j0);

            // 1) Sij = Qi (br x d) * Kj^T (d x bc)  -> (br x bc)
            // Row-major leading dims: Qi->d, Kj->d, Sij->bc
            block_dgemm(false, true,
                        br, bc, d, scale,
                        Q + i0*d, d,
                        K + j0*d, d,
                        0.f, Sij.data(), bc);
            // dgemm(false, true,
            //       br, bc, d, scale,
            //       Q + i0*d, d,
            //       K + j0*d, d,
            //       0.f, Sij.data(), bc);

            // 2) stable exp, rowwise max/sum
            for (size_t r = 0; r < br; ++r) {
                const size_t row = i0 + r;
                double* srow = Sij.data() + r * bc;
                double* PVrow = PV.data() + r * d;

                const double m_old = m[row];
                m[row] = std::max(m_old, row_max(srow, bc));
                const double a = std::exp(m_old - m[row]);
                l[row] *= a;
                for (size_t c = 0; c < bc; ++c) {
                    srow[c] = std::exp(srow[c] - m[row]);
                    l[row] += srow[c];
                }
                for (size_t k = 0; k < d; ++k) {
                    PVrow[k] *= a;
                }
            }

            // 3) PV = Ptilde (br x bc) * Vj (bc x d) -> (br x d)  [ONE GEMM]
            block_dgemm(false, false,
                        br, d, bc, 1.f,
                        Sij.data(), bc,
                        V + j0*d, d,
                        1.f, PV.data(), d);
            // dgemm(false, false,
            //       br, d, bc, 1.f,
            //       Sij.data(), bc,
            //       V + j0*d, d,
            //       1.f, PV.data(), d);
        }
        for (size_t r = 0; r < br; ++r) {
            const size_t row = i0 + r;
            double* Orow = O.data() + row * d;
            double* PVrow = PV.data() + r * d;
            const double inv_l = 1.f / l[row];
            for (size_t k = 0; k < d; ++k) {
                Orow[k] += inv_l * PVrow[k];
            }
        }
    }
    return O;
}
