#ifndef ATTENTION_H
#define ATTENTION_H

#include <cstddef>
#include <vector>

// Attention variants
std::vector<double> naive_attention(const double* __restrict__ Q,
                                   const double* __restrict__ K,
                                   const double* __restrict__ V,
                                   size_t N, size_t d, double scale);

std::vector<double> flash_attention_v2(const double* __restrict__ Q,
                                      const double* __restrict__ K,
                                      const double* __restrict__ V,
                                      size_t N, size_t d, size_t M_bytes, double scale);

#endif // ATTENTION_H
