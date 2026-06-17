#include "utils.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <type_traits>

// A templated implementation of numpy.allclose() for float/double
template <typename T>
bool allclose(const std::vector<T>& a,
              const std::vector<T>& b)
{
    static_assert(std::is_floating_point_v<T>, "allclose<T>: T must be float or double");
    if (a.size() != b.size()) return false;

    // Use relaxed tolerances for float
    T rtol;
    T atol;
    if constexpr (std::is_same_v<T, float>) {
        rtol = 1e-3f;
        atol = 1e-5f;
    } else {
        rtol = 1e-8;
        atol = 1e-10;
    }

    for (size_t i = 0; i < a.size(); i++) {
        T diff = std::fabs(a[i] - b[i]);
        T tol = atol + rtol * std::fabs(b[i]);
        if (diff > tol) return false;
    }
    return true;
}

template bool allclose<float>(const std::vector<float>&,
                              const std::vector<float>&);

template bool allclose<double>(const std::vector<double>&,
                               const std::vector<double>&);

double row_max(const double* row, size_t len) {
    double m = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < len; ++i) m = std::max(m, row[i]);
    return m;
}

template <typename T>
void random_matrix(std::vector<T>& A) {
    std::mt19937 rng(42);
    T low = T(-1);
    T high = T(1);
    std::uniform_real_distribution<T> dist(low, high);
    for (auto& x : A) x = dist(rng);
}

// Explicit instantiations
template void random_matrix<float>(std::vector<float>&);
template void random_matrix<double>(std::vector<double>&);

std::pair<size_t,size_t> compute_block_sizes_from_M(size_t N, size_t M_doubles, size_t d) {
    // compute base by floor division
    size_t base = std::max(size_t(1), M_doubles / (size_t(5) * d));  // Use 5 as the factor to avoid overwriting
    size_t Br = std::min(N, std::min(base, d));
    size_t Bc = std::min(N, base);
    // std::cout << "\tbase=" << base << ", Br=" << Br << ", Bc=" << Bc << std::endl;
    return {Br, Bc};
}

// Inner block multiplication kernel
template <typename T>
inline void do_block(bool transA, bool transB,
                     int i0, int imax,
                     int j0, int jmax,
                     int k0, int kmax,
                     int lda, int ldb, int ldc,
                     T alpha,
                     const T* A, const T* B,
                     T* C)
{
    for (int i = i0; i < imax; i++) {
        T* Crow = C + i * ldc;
        for (int k = k0; k < kmax; k++) {
            T aik = alpha * (
                !transA ? A[i * lda + k] : A[k * lda + i]);
            for (int j = j0; j < jmax; j++) {
                T bkj = !transB ? B[k * ldb + j] : B[j * ldb + k];
                Crow[j] += aik * bkj;
            }
        }
    }
}

// Naive GEMM (no blocking): uses 3 nested loops
template <typename T>
void naive_gemm(bool transA, bool transB,
                int M, int N, int K,
                T alpha,
                const T* A, int lda,
                const T* B, int ldb,
                T beta,
                T* C, int ldc)
{
    static_assert(std::is_floating_point_v<T>, "naive_gemm<T>: T must be float or double");

    // Scale C by beta
    if (beta == T(0)) {
        for (int i = 0; i < M; i++)
            std::fill(C + i * ldc, C + i * ldc + N, T(0));  // fill C with zero
    } else if (beta != T(1)) {
        for (int i = 0; i < M; i++) {
            T* Crow = C + i * ldc;
            for (int j = 0; j < N; j++)
                Crow[j] *= beta;  // C = beta * C
        }
    }

    // Call do_block once for the entire matrix (no blocking)
    // C += A @ B
    do_block<T>(transA, transB,
                0, M, 0, N, 0, K,
                lda, ldb, ldc,
                alpha, A, B, C);
}

// Type-specific wrappers (BLAS-style)
void naive_sgemm(bool transA, bool transB,
                 int M, int N, int K,
                 float alpha,
                 const float* A, int lda,
                 const float* B, int ldb,
                 float beta,
                 float* C, int ldc)
{
    naive_gemm<float>(transA, transB, M, N, K,
                      alpha, A, lda, B, ldb, beta, C, ldc);
}

void naive_dgemm(bool transA, bool transB,
                 int M, int N, int K,
                 double alpha,
                 const double* A, int lda,
                 const double* B, int ldb,
                 double beta,
                 double* C, int ldc)
{
    naive_gemm<double>(transA, transB, M, N, K,
                       alpha, A, lda, B, ldb, beta, C, ldc);
}

// --------- MODIFIED BLOCK GEMM WITH PACKING ----------
template <typename T>
void block_gemm(bool transA, bool transB,
                int M, int N, int K,
                T alpha,
                const T* A, int lda,
                const T* B, int ldb,
                T beta,
                T* C, int ldc)
{
    static_assert(std::is_floating_point_v<T>, "T must be float/double");
    // Scale C
    if (beta == T(0)) {
        for (int i = 0; i < M; i++)
            std::fill(C + i * ldc, C + i * ldc + N, T(0));
    } else if (beta != T(1)) {
        for (int i = 0; i < M; i++) {
            T* Crow = C + i * ldc;
            for (int j = 0; j < N; j++) Crow[j] *= beta;
        }
    }

    // L2-sized tiles (tune as needed)
    constexpr int Mb = 64;
    constexpr int Nb = 64;
    constexpr int Kb = 192;

    std::vector<T> Abuf(Mb * Kb);
    std::vector<T> Bbuf(Kb * Nb);

    for (int k0 = 0; k0 < K; k0 += Kb) {
        const int kmax = std::min(k0 + Kb, K);
        const int kr   = kmax - k0;

        for (int j0 = 0; j0 < N; j0 += Nb) {
            const int jmax = std::min(j0 + Nb, N);
            const int nr   = jmax - j0;

            // Pack B panel (kr x nr) into Bbuf row-major
            if (!transB) {
                for (int k = 0; k < kr; ++k) {
                    const T* Bk = B + (k0 + k) * ldb + j0;
                    std::memcpy(Bbuf.data() + k * nr, Bk, sizeof(T) * nr);
                }
            } else {
                for (int k = 0; k < kr; ++k) {
                    for (int j = 0; j < nr; ++j) {
                        Bbuf[k * nr + j] = B[(j0 + j) * ldb + (k0 + k)];
                    }
                }
            }

            for (int i0 = 0; i0 < M; i0 += Mb) {
                const int imax = std::min(i0 + Mb, M);
                const int mr   = imax - i0;

                // Pack A block (mr x kr) into Abuf row-major
                if (!transA) {
                    for (int i = 0; i < mr; ++i) {
                        const T* Ai = A + (i0 + i) * lda + k0;
                        std::memcpy(Abuf.data() + i * kr, Ai, sizeof(T) * kr);
                    }
                } else {
                    for (int i = 0; i < mr; ++i) {
                        for (int k = 0; k < kr; ++k) {
                            Abuf[i * kr + k] = A[(k0 + k) * lda + (i0 + i)];
                        }
                    }
                }

                // Compute C[i0:imax, j0:jmax] += Abuf(mr x kr) * Bbuf(kr x nr)
                do_block<T>(
                    /*transA=*/false, /*transB=*/false,
                    /*i0=*/0, /*imax=*/mr,
                    /*j0=*/0, /*jmax=*/nr,
                    /*k0=*/0, /*kmax=*/kr,
                    /*lda=*/kr, /*ldb=*/nr, /*ldc=*/ldc,
                    alpha,
                    Abuf.data(), Bbuf.data(),
                    C + i0 * ldc + j0);
            }
        }
    }
}

// Type-specific wrappers (BLAS-style)
void block_sgemm(bool transA, bool transB,
                 int M, int N, int K,
                 float alpha,
                 const float* A, int lda,
                 const float* B, int ldb,
                 float beta,
                 float* C, int ldc)
{
    block_gemm<float>(transA, transB, M, N, K,
                      alpha, A, lda, B, ldb, beta, C, ldc);
}

void block_dgemm(bool transA, bool transB,
                 int M, int N, int K,
                 double alpha,
                 const double* A, int lda,
                 const double* B, int ldb,
                 double beta,
                 double* C, int ldc)
{
    block_gemm<double>(transA, transB, M, N, K,
                       alpha, A, lda, B, ldb, beta, C, ldc);
}
