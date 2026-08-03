// Is the algorithm resolved at a call site's envelope actually a worse kernel
// than the one cuBLASLt picks for the exact shape? Times the same GEMM with
// both, nothing else differing. Raw cuBLASLt, no sofieBLAS or alpaka.

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  std::printf("CUDA %s @%d\n",cudaGetErrorString(e),__LINE__); std::exit(1);} }while(0)
#define CB(x) do{ cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS){ \
  std::printf("cuBLAS %d @%d\n",(int)s,__LINE__); std::exit(1);} }while(0)

// ParticleNet's weight shapes, with the largest m each call site reaches.
// n_pf runs 20..120 and n_sv is 10, so tot = n_pf+n_sv and tot8 = tot*8.
struct Shape { int n, k, mEnv; };
static const Shape kShapes[] = {
  { 32,  20,  120},   // m = n_pf
  { 64,  32,  130},   // m = tot
  { 96,  64,  130},
  {128,  96,  130},
  {256, 288,  130},
  { 64,  64, 1040},   // m = tot8
  { 96, 128, 1040},
  { 96,  96, 1040},
  {128, 192, 1040},
  {128, 128, 1040},
};
static const int kNShapes = sizeof(kShapes) / sizeof(kShapes[0]);

static const int MAXM = 1040, MAXK = 288, MAXN = 256;
static const int ITERS = 200, REPS = 5;

static cublasLtHandle_t lt;
static cublasLtMatmulPreference_t pref;
static cublasLtMatmulDesc_t op;
static void *ws = nullptr;
static size_t wsSize = 1u << 25;
static float *dA, *dB, *dC;

static cublasLtMatrixLayout_t mkL(int rows, int cols, int ld) {
  cublasLtMatrixLayout_t L = nullptr;
  CB(cublasLtMatrixLayoutCreate(&L, CUDA_R_32F, rows, cols, ld));
  return L;
}

// Heuristic pick for a given shape, or false if none.
static bool resolveAt(int m, int n, int k, cublasLtMatmulAlgo_t &out) {
  auto Ad = mkL(m, k, m), Bd = mkL(k, n, k), Cd = mkL(m, n, m);
  cublasLtMatmulHeuristicResult_t h{};
  int got = 0;
  CB(cublasLtMatmulAlgoGetHeuristic(lt, op, Ad, Bd, Cd, Cd, pref, 1, &h, &got));
  cublasLtMatrixLayoutDestroy(Ad);
  cublasLtMatrixLayoutDestroy(Bd);
  cublasLtMatrixLayoutDestroy(Cd);
  if (got) out = h.algo;
  return got > 0;
}

static double timeAt(int m, int n, int k, const cublasLtMatmulAlgo_t &algo) {
  auto Ad = mkL(m, k, m), Bd = mkL(k, n, k), Cd = mkL(m, n, m);
  const float alpha = 1.f, beta = 0.f;
  auto run = [&] {
    CB(cublasLtMatmul(lt, op, &alpha, dA, Ad, dB, Bd, &beta, dC, Cd, dC, Cd,
                      &algo, ws, wsSize, 0));
  };
  for (int i = 0; i < 20; ++i) run();
  CK(cudaDeviceSynchronize());

  cudaEvent_t t0, t1;
  CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
  CK(cudaEventRecord(t0));
  for (int i = 0; i < ITERS; ++i) run();
  CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
  float ms = 0.f; CK(cudaEventElapsedTime(&ms, t0, t1));
  CK(cudaEventDestroy(t0)); CK(cudaEventDestroy(t1));

  cublasLtMatrixLayoutDestroy(Ad);
  cublasLtMatrixLayoutDestroy(Bd);
  cublasLtMatrixLayoutDestroy(Cd);
  return ms / ITERS;
}

// Usable at this shape? cuBLASLt refuses some combinations.
static bool usable(int m, int n, int k, const cublasLtMatmulAlgo_t &algo) {
  auto Ad = mkL(m, k, m), Bd = mkL(k, n, k), Cd = mkL(m, n, m);
  cublasLtMatmulHeuristicResult_t chk{};
  const cublasStatus_t st =
      cublasLtMatmulAlgoCheck(lt, op, Ad, Bd, Cd, Cd, &algo, &chk);
  cublasLtMatrixLayoutDestroy(Ad);
  cublasLtMatrixLayoutDestroy(Bd);
  cublasLtMatrixLayoutDestroy(Cd);
  return st == CUBLAS_STATUS_SUCCESS && chk.workspaceSize <= wsSize;
}

int main() {
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, 0);
  std::printf("device: %s  SMs=%d  cc=%d.%d\n\n", prop.name,
              prop.multiProcessorCount, prop.major, prop.minor);

  CB(cublasLtCreate(&lt));
  CB(cublasLtMatmulPreferenceCreate(&pref));
  CK(cudaMalloc(&ws, wsSize));
  CB(cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsSize, sizeof(wsSize)));
  CK(cudaMalloc(&dA, sizeof(float) * (size_t)MAXM * MAXK));
  CK(cudaMalloc(&dB, sizeof(float) * (size_t)MAXK * MAXN));
  CK(cudaMalloc(&dC, sizeof(float) * (size_t)MAXM * MAXN));
  CK(cudaMemset(dA, 0, sizeof(float) * (size_t)MAXM * MAXK));
  CK(cudaMemset(dB, 0, sizeof(float) * (size_t)MAXK * MAXN));

  CB(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t opN = CUBLAS_OP_N;
  CB(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN)));
  CB(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));

  std::printf("%6s %5s %6s %6s %11s %11s %9s\n",
              "n", "k", "mEnv", "m", "exact_ms", "envelope_ms", "delta%");

  std::vector<double> deltas;
  for (int s = 0; s < kNShapes; ++s) {
    const int n = kShapes[s].n, k = kShapes[s].k, mEnv = kShapes[s].mEnv;

    cublasLtMatmulAlgo_t algoEnv{};
    if (!resolveAt(mEnv, n, k, algoEnv)) {
      std::printf("%6d %5d %6d   no algorithm at envelope\n", n, k, mEnv);
      continue;
    }

    for (int pct : {20, 40, 60, 80, 100}) {
      const int m = std::max(1, mEnv * pct / 100);
      cublasLtMatmulAlgo_t algoExact{};
      if (!resolveAt(m, n, k, algoExact)) continue;
      if (!usable(m, n, k, algoEnv)) {
        std::printf("%6d %5d %6d %6d   envelope algorithm rejected here\n",
                    n, k, mEnv, m);
        continue;
      }

      // Alternate the two so any drift hits both equally, then take the best
      // of each, since drift only ever makes a run slower.
      double bestExact = 1e30, bestEnv = 1e30;
      for (int r = 0; r < REPS; ++r) {
        bestExact = std::min(bestExact, timeAt(m, n, k, algoExact));
        bestEnv   = std::min(bestEnv,   timeAt(m, n, k, algoEnv));
      }
      const double d = (bestEnv - bestExact) / bestExact * 100.0;
      deltas.push_back(d);
      std::printf("%6d %5d %6d %6d %11.5f %11.5f %+9.1f\n", n, k, mEnv, m,
                  bestExact, bestEnv, d);
    }
  }

  if (!deltas.empty()) {
    std::sort(deltas.begin(), deltas.end());
    std::printf("\ndelta%% (envelope vs exact):  min %+.1f   median %+.1f   max %+.1f\n",
                deltas.front(), deltas[deltas.size() / 2], deltas.back());
    std::printf("negative means the envelope algorithm is faster\n");
  }
  return 0;
}
