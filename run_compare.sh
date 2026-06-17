#ifndef UTILS_H
#define UTILS_H

#include <cstddef>
#include <utility>
#include <vector>

// ---------------- utilities ----------------
template <class T>
inline void DoNotOptimize(T&& value)
{
#if defined(__clang__)
    asm volatile("" : "+r,m"(value) : : "memory");
#else
    asm volatile("" : "+m,r"(value) : : "memory");
#endif
}

// Forward declare allclose template (float/double)
template <typename T>
bool allclose(const std::vector<T>& a,
              const std::vector<T>& b);

double row_max(const double* row, size_t len);

template <typename T>
void random_matrix(std::vector<T>& A);

std::pair<size_t, size_t> compute_block_sizes_from_M(size_t N, size_t M_doubles, size_t d);

void naive_sgemm(bool transA, bool transB,
                 int M, int N, int K,
                 float alpha,
                 const float* A, int lda,
                 const float* B, int ldb,
                 float beta,
                 float* C, int ldc);

void naive_dgemm(bool transA, bool transB,
                 int M, int N, int K,
                 double alpha,
                 const double* A, int lda,
                 const double* B, int ldb,
                 double beta,
                 double* C, int ldc);

void block_sgemm(bool transA, bool transB,
                 int M, int N, int K,
                 float alpha,
                 const float* A, int lda,
                 const float* B, int ldb,
                 float beta,
                 float* C, int ldc);

void block_dgemm(bool transA, bool transB,
                 int M, int N, int K,
                 double alpha,
                 const double* A, int lda,
                 const double* B, int ldb,
                 double beta,
                 double* C, int ldc);

#endif // UTILS_H
