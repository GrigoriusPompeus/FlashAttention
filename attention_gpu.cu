// attention_gpu.cu — GPU port of the Naive vs Flash Attention v2 experiment.
// Compile: nvcc -O2 -o attention_gpu attention_gpu.cu
// Run:     ./attention_gpu
//
// We use float (fp32) rather than the CPU's double because GPU throughput for
// float is ~8x higher on the RTX 2080 Ti; the algorithm is identical either way.
//
// Memory hierarchy reminder:
//   Registers  ~0 cycle latency,  ~256 KB / SM
//   Shared mem ~32 cycle latency, ~48 KB  / block
//   HBM (VRAM) ~300 cycle latency, 616 GB/s on 2080 Ti
//
// Naive attention writes the full N×N score matrix to HBM and reads it back
// twice (once for softmax, once for the output matmul).  Flash Attention keeps
// tiles in shared memory so the N×N matrix never lands in HBM.

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

// ── Compile-time constants ───────────────────────────────────────────────────

#define D   64   // head dimension (same as the CPU experiment)
#define BR  32   // Flash Attention: number of Q rows per thread block
#define BC  32   // Flash Attention: number of K/V rows per KV tile

// ── Helpers ──────────────────────────────────────────────────────────────────

// Abort on CUDA errors.
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

// ────────────────────────────────────────────────────────────────────────────
// NAIVE ATTENTION  (three separate kernel passes)
//
// Pass 1: S[i,j] = dot(Q[i,:], K[j,:]) / sqrt(D)        — writes N×N to HBM
// Pass 2: softmax each row of S in-place                 — reads + writes N×N
// Pass 3: O[i,k] = sum_j S[i,j] * V[j,k]               — reads N×N from HBM
// ────────────────────────────────────────────────────────────────────────────

// Each thread computes one element of S.
__global__ void qk_kernel(const float* Q, const float* K, float* S, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // row of Q
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // row of K
    if (i >= N || j >= N) return;
    float dot = 0.f;
    for (int k = 0; k < D; k++)
        dot += Q[i*D + k] * K[j*D + k];
    S[i*N + j] = dot * rsqrtf((float)D);
}

// One thread per row: sequential scan for max, then two passes for softmax.
// (A warp-reduction softmax would be faster but more code; this is clear.)
__global__ void softmax_kernel(float* S, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float* row = S + i * N;
    float mx = -FLT_MAX;
    for (int j = 0; j < N; j++) mx = fmaxf(mx, row[j]);
    float sum = 0.f;
    for (int j = 0; j < N; j++) { row[j] = expf(row[j] - mx); sum += row[j]; }
    for (int j = 0; j < N; j++) row[j] /= sum;
}

// Each thread computes one element of O.
__global__ void sv_kernel(const float* S, const float* V, float* O, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // row of output
    int k = blockIdx.x * blockDim.x + threadIdx.x;  // column of output
    if (i >= N || k >= D) return;
    float sum = 0.f;
    for (int j = 0; j < N; j++)
        sum += S[i*N + j] * V[j*D + k];
    O[i*D + k] = sum;
}

// Launch naive kernels and return elapsed milliseconds (GPU time only).
float run_naive(const float* Q, const float* K, const float* V,
                float* O, float* S, int N) {
    dim3 blk(16, 16);
    dim3 gNN((N+15)/16, (N+15)/16);   // grid for N×N output
    dim3 gND((D+15)/16, (N+15)/16);   // grid for N×D output

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    CUDA_CHECK(cudaEventRecord(t0));
    qk_kernel     <<<gNN, blk>>>(Q, K, S, N);
    softmax_kernel<<<(N+255)/256, 256>>>(S, N);
    sv_kernel     <<<gND, blk>>>(S, V, O, N);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    return ms;
}

// ────────────────────────────────────────────────────────────────────────────
// FLASH ATTENTION v2  (single fused kernel)
//
// One thread block = BR rows of Q.
// One thread       = one Q row (register-resident throughout).
// Shared memory    = one BC×D tile of K and one BC×D tile of V.
//
// The N×N attention score matrix is never materialized in HBM.
// Instead we iterate over KV tiles and use the online softmax recurrence:
//
//   m_new = max(m_old, tile_max)
//   l_new = exp(m_old - m_new) * l_old + sum(exp(s - m_new))
//   o_new = exp(m_old - m_new) * o_old + sum_j exp(s_j - m_new) * V_j
//
// At the end: O = o / l
// ────────────────────────────────────────────────────────────────────────────

__global__ void flash_attn_kernel(const float* Q, const float* K,
                                   const float* V, float* O, int N) {
    int q_row = blockIdx.x * BR + threadIdx.x;  // global Q row for this thread

    // Shared tiles: one BC×D slice of K and V, refreshed each KV iteration.
    __shared__ float Ks[BC][D];
    __shared__ float Vs[BC][D];

    // Load this thread's Q row into registers — it stays here for the entire
    // kernel and is reused against every KV tile without touching HBM again.
    float q[D] = {};
    if (q_row < N)
        for (int k = 0; k < D; k++) q[k] = Q[q_row*D + k];

    // Online softmax accumulators (live in registers).
    float m = -FLT_MAX;  // running max of all scores seen so far
    float l = 0.f;       // running denominator (sum of exp(s - m))
    float o[D] = {};     // running numerator  (sum of exp(s - m) * V)

    int kv_blocks = (N + BC - 1) / BC;

    for (int kv = 0; kv < kv_blocks; kv++) {
        int kv0 = kv * BC;

        // ── Cooperatively load one K tile and one V tile into shared memory ──
        // BR threads share the work: each loads (BC*D / BR) = 64 elements.
        for (int idx = threadIdx.x; idx < BC * D; idx += BR) {
            int r  = idx / D, c = idx % D;
            int gr = kv0 + r;                          // global K/V row
            Ks[r][c] = (gr < N) ? K[gr*D + c] : 0.f;
            Vs[r][c] = (gr < N) ? V[gr*D + c] : 0.f;
        }
        __syncthreads();  // all threads must finish loading before anyone reads

        if (q_row < N) {
            // ── Compute attention scores for this (Q_row, KV_tile) pair ──────
            float s[BC];
            for (int j = 0; j < BC; j++) {
                int gj = kv0 + j;
                if (gj >= N) { s[j] = -FLT_MAX; continue; }
                float dot = 0.f;
                for (int k = 0; k < D; k++) dot += q[k] * Ks[j][k];
                s[j] = dot * rsqrtf((float)D);
            }

            // ── Online softmax update ─────────────────────────────────────────
            float m_new = m;
            for (int j = 0; j < BC; j++) m_new = fmaxf(m_new, s[j]);

            float scale = expf(m - m_new);   // rescales the old accumulator
            l *= scale;
            for (int j = 0; j < BC; j++) l += expf(s[j] - m_new);

            // Update output: rescale old contribution, add new tile's contribution.
            for (int k = 0; k < D; k++) {
                float pv = 0.f;
                for (int j = 0; j < BC; j++)
                    pv += expf(s[j] - m_new) * Vs[j][k];
                o[k] = scale * o[k] + pv;
            }
            m = m_new;
        }
        __syncthreads();  // guard against next tile load overwriting Ks/Vs
    }

    // Write normalized output.
    if (q_row < N)
        for (int k = 0; k < D; k++) O[q_row*D + k] = o[k] / l;
}

float run_flash(const float* Q, const float* K, const float* V, float* O, int N) {
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    CUDA_CHECK(cudaEventRecord(t0));
    flash_attn_kernel<<<(N + BR - 1) / BR, BR>>>(Q, K, V, O, N);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    return ms;
}

// ── Correctness check (only run at N=1024 to avoid long host transfers) ──────

void verify(const float* dO_naive, const float* dO_flash, int N) {
    int n = N * D;
    float* a = (float*)malloc(n * sizeof(float));
    float* b = (float*)malloc(n * sizeof(float));
    CUDA_CHECK(cudaMemcpy(a, dO_naive, n*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b, dO_flash, n*sizeof(float), cudaMemcpyDeviceToHost));
    float max_diff = 0.f;
    for (int i = 0; i < n; i++)
        max_diff = fmaxf(max_diff, fabsf(a[i] - b[i]));
    printf("  [correctness check N=%d] max |naive - flash| = %.2e %s\n",
           N, max_diff, max_diff < 1e-3f ? "(OK)" : "(MISMATCH!)");
    free(a); free(b);
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main() {
    srand(42);
    int sizes[] = {1024, 2048, 4096, 8192, 16384};
    int nsizes  = 5;

    printf("N, Naive time (ms), Ratio, FlashAttention v2 time (ms), Ratio\n");

    float prev_n = 0.f, prev_f = 0.f;

    for (int si = 0; si < nsizes; si++) {
        int N = sizes[si];
        size_t qkv_bytes = (size_t)N * D * sizeof(float);
        size_t s_bytes   = (size_t)N * N * sizeof(float);  // naive only

        // Initialise Q, K, V on host.
        float* h = (float*)malloc(3 * qkv_bytes);
        for (int i = 0; i < 3*N*D; i++)
            h[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;

        // Allocate GPU buffers.
        float *dQ, *dK, *dV, *dO_n, *dO_f, *dS;
        CUDA_CHECK(cudaMalloc(&dQ,   qkv_bytes));
        CUDA_CHECK(cudaMalloc(&dK,   qkv_bytes));
        CUDA_CHECK(cudaMalloc(&dV,   qkv_bytes));
        CUDA_CHECK(cudaMalloc(&dO_n, qkv_bytes));
        CUDA_CHECK(cudaMalloc(&dO_f, qkv_bytes));
        CUDA_CHECK(cudaMalloc(&dS,   s_bytes));

        CUDA_CHECK(cudaMemcpy(dQ, h,         qkv_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dK, h+N*D,     qkv_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dV, h+2*N*D,   qkv_bytes, cudaMemcpyHostToDevice));

        // Warm-up: one untimed run each, so CUDA initialisation costs don't
        // pollute the first measurement.
        run_naive(dQ, dK, dV, dO_n, dS, N);
        run_flash(dQ, dK, dV, dO_f,     N);

        // Timed runs.
        float mn = run_naive(dQ, dK, dV, dO_n, dS, N);
        float mf = run_flash(dQ, dK, dV, dO_f,     N);

        // Sanity-check outputs agree at the first size.
        if (si == 0) verify(dO_n, dO_f, N);

        if (si == 0)
            printf("%d, %.2f, -, %.2f, -\n", N, mn, mf);
        else
            printf("%d, %.2f, %.2f, %.2f, %.2f\n", N, mn, mn/prev_n, mf, mf/prev_f);

        prev_n = mn;
        prev_f = mf;

        CUDA_CHECK(cudaFree(dQ));
        CUDA_CHECK(cudaFree(dK));
        CUDA_CHECK(cudaFree(dV));
        CUDA_CHECK(cudaFree(dO_n));
        CUDA_CHECK(cudaFree(dO_f));
        CUDA_CHECK(cudaFree(dS));
        free(h);
    }
    return 0;
}
