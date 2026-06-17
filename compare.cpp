#include "attention.h"
#include "utils.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

int main() {
    size_t d = 64;
    // can change this block size 
    size_t M_bytes = 32 * 1024 * 1024;
    double scale = 1.0f / std::sqrt(double(d));
    std::vector<size_t> Ns = {1024, 2048, 4096, 8192, 16384};

    // std::cout << "M=" << M_bytes/1024 << " KiB\n";

    std::cout << "N, Naive time (ms), Ratio, FlashAttention v2 time (ms), Ratio\n";

    double prev_naive = 0.f;
    double prev_flash = 0.f;
    bool has_prev = false;
    for (size_t N : Ns) {
        std::vector<double> Q(N * d), K(N * d), V(N * d);
        random_matrix<double>(Q);
        random_matrix<double>(K);
        random_matrix<double>(V);
`
        auto t1 = std::chrono::steady_clock::now();
        auto O_naive = naive_attention(Q.data(), K.data(), V.data(), N, d, scale);
        auto t2 = std::chrono::steady_clock::now();
        DoNotOptimize(O_naive.data());
        double dt_naive = std::chrono::duration<double, std::milli>(t2-t1).count();

        auto t3 = std::chrono::steady_clock::now();
        auto O_flash = (Q.data(), K.data(), V.data(), N, d, M_bytes, scale);
        auto t4 = std::chrono::steady_clock::now();
        DoNotOptimize(O_flash.data());
        double dt_flash = std::chrono::duration<double, std::milli>(t4-t3).count();

        if (has_prev) {
            std::printf("%zu, %.2f, %.2f, %.2f, %.2f\n",
                        N, dt_naive, dt_naive / prev_naive,
                        dt_flash, dt_flash / prev_flash);
        } else {
            std::printf("%zu, %.2f, -, %.2f, -\n", N, dt_naive, dt_flash);
        }

        prev_naive = dt_naive;
        prev_flash = dt_flash;
        has_prev = true;
    }
}
