#pragma once

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

#include <cstdlib>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sofieBLAS/core.hpp"
#include <alpaka/alpaka.hpp>
#include <cublasLt.h>
#include <cublas_v2.h>

#define CHECK_CUDA(err)                                                        \
  if ((err) != cudaSuccess) {                                                  \
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at line "      \
              << __LINE__ << "\n";                                             \
    exit(EXIT_FAILURE);                                                        \
  }

#define CHECK_CUBLAS(status)                                                   \
  do {                                                                         \
    cublasStatus_t _s = (status);                                              \
    if (_s != CUBLAS_STATUS_SUCCESS) {                                         \
      std::cerr << "cuBLAS error " << _s << " at line " << __LINE__ << "\n";   \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)


struct DescKey {
  int transA;   // CUBLAS_OP_N / CUBLAS_OP_T encoded as int
  int transB;
  int epilogue; // cublasLtEpilogue_t encoded as int
  bool operator==(const DescKey &o) const noexcept {
    return transA == o.transA && transB == o.transB && epilogue == o.epilogue;
  }
};

struct DescKeyHash {
  std::size_t operator()(const DescKey &k) const noexcept {
    // Small values: simple polynomial hash
    std::size_t h = static_cast<std::size_t>(k.transA) * 97u
                  + static_cast<std::size_t>(k.transB) * 31u
                  + static_cast<std::size_t>(k.epilogue);
    return h ^ (h >> 16);
  }
};


struct AlgoKey {
  DescKey   dk;
  std::size_t rowsA, colsA;  // physical dimensions of A
  std::size_t rowsB, colsB;  // physical dimensions of B
  bool operator==(const AlgoKey &o) const noexcept {
    return dk == o.dk
        && rowsA == o.rowsA && colsA == o.colsA
        && rowsB == o.rowsB && colsB == o.colsB;
  }
};

struct AlgoKeyHash {
  std::size_t operator()(const AlgoKey &k) const noexcept {
    std::size_t h = DescKeyHash{}(k.dk);
    auto mix = [&](std::size_t v) {
      h ^= std::hash<std::size_t>{}(v) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    };
    mix(k.rowsA); mix(k.colsA);
    mix(k.rowsB); mix(k.colsB);
    return h;
  }
};

// 0 = tune per runtime shape, 1 = tune once per bucket and reuse.
// Layouts are stamped with the exact runtime dims either way; only the
// choice of algorithm is shared across a bucket.
#ifndef SOFIEBLAS_LAYOUT_PROFILE
#define SOFIEBLAS_LAYOUT_PROFILE 1
#endif

// Candidates timed when tuning a bucket. 1 takes the heuristic's first pick.
#ifndef SOFIEBLAS_LAYOUT_TUNE_CANDIDATES
#define SOFIEBLAS_LAYOUT_TUNE_CANDIDATES 1
#endif

#ifndef SOFIEBLAS_LAYOUT_BUCKET_MIN
#define SOFIEBLAS_LAYOUT_BUCKET_MIN 8
#endif

// 0 = never validate a bucket's algorithm, 1 = once when the bucket is tuned,
// 2 = on every call. Only the algorithm is taken from the check; the workspace
// handed to cublasLtMatmul is the full allocation either way.
#ifndef SOFIEBLAS_LAYOUT_VALIDATE
#define SOFIEBLAS_LAYOUT_VALIDATE 1
#endif

struct AlgoProfile {
  cublasLtMatmulHeuristicResult_t tuned{};
  cublasLtMatmulHeuristicResult_t lastValid{};
  AlgoKey lastExact{};
  bool    haveLast = false;
  bool    usable   = true;
};

struct LayoutStats {
  std::size_t matmuls          = 0;
  std::size_t heuristicQueries = 0;
  std::size_t algoChecks       = 0;
  std::size_t algoCheckRejects = 0;
  std::size_t memoHits         = 0;
  std::size_t exactFallbacks   = 0;
  std::size_t tuningRuns       = 0;
};

class BlasCuda {
  cublasLtHandle_t           ltHandle   = nullptr;
  cublasHandle_t             handle     = nullptr;  // legacy cuBLAS for batched ops
  cublasLtMatmulPreference_t preference = nullptr;
  void *d_workspace  = nullptr;
  size_t workspaceSize = 1u << 25;  // 32 MB (was 4 MB)
  cudaStream_t stream = nullptr;

  // One persistent layout descriptor per matrix role, re-stamped with the
  // runtime dimensions before each matmul. The descriptor is host-side metadata
  // consumed by cublasLtMatmul at the call, so a single object can be reused
  // across shapes - this is what lets one Session serve dynamic (runtime) sizes.
  enum LayoutRole { ROLE_A = 0, ROLE_B = 1, ROLE_C = 2 };
  cublasLtMatrixLayout_t roleLayout[3] = {};

  std::unordered_map<DescKey, cublasLtMatmulDesc_t, DescKeyHash> descStore;

  std::unordered_map<AlgoKey, cublasLtMatmulHeuristicResult_t, AlgoKeyHash>
      algoCache;

  // keyed by bucketed dims, one tuned algorithm per bucket
  std::unordered_map<AlgoKey, AlgoProfile, AlgoKeyHash> profileStore;

  LayoutStats stats;

public:
  const LayoutStats &layoutStats() const { return stats; }
  std::size_t profileCount()      const { return profileStore.size(); }
  std::size_t algoCacheSize()     const { return algoCache.size(); }

  BlasCuda(const BlasCuda &) = delete;
  BlasCuda &operator=(const BlasCuda &) = delete;
  BlasCuda(BlasCuda &&) = delete;
  BlasCuda &operator=(BlasCuda &&) = delete;

  BlasCuda(alpaka::QueueCudaRtNonBlocking &queue) : m_queue{queue} {
    stream = static_cast<cudaStream_t>(m_queue.getNativeHandle());

    CHECK_CUBLAS(cublasLtCreate(&ltHandle));

    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUBLAS(cublasSetStream(handle, stream));

    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
    CHECK_CUDA(cudaMalloc(&d_workspace, workspaceSize));
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize,
        sizeof(workspaceSize)));
  }

  ~BlasCuda() {
    for (auto L : roleLayout)
      if (L) cublasLtMatrixLayoutDestroy(L);
    for (auto &[key, desc] : descStore)
      if (desc) cublasLtMatmulDescDestroy(desc);
    if (preference)  cublasLtMatmulPreferenceDestroy(preference);
    if (ltHandle)    cublasLtDestroy(ltHandle);
    if (handle)      cublasDestroy(handle);
    if (d_workspace) cudaFree(d_workspace);
  }

  inline cublasOperation_t charToCuBlasTranspose(char trans) {
    switch (trans) {
    case 'N': case 'n': return CUBLAS_OP_N;
    case 'T': case 't': return CUBLAS_OP_T;
    case 'C': case 'c': return CUBLAS_OP_C;
    default:
      throw std::invalid_argument("Invalid transpose character for cuBLAS.");
    }
  }

  // No-op kept for the generated ctor's API; layouts are created lazily now.
  void addLayoutConfig(std::size_t, std::size_t, std::size_t,
                       std::size_t, std::size_t, std::size_t, char, char) {}

  template <typename T, typename TIdx>
  inline void
  gemm(char transa, char transb, unsigned int m, unsigned int n, unsigned int k,
       float alpha, alpaka::BufCudaRt<T, alpaka::DimInt<1u>, TIdx> const &A,
       alpaka::BufCudaRt<T, alpaka::DimInt<1u>, TIdx> const &B, float beta,
       alpaka::BufCudaRt<T, alpaka::DimInt<1u>, TIdx> &bias,
       alpaka::BufCudaRt<T, alpaka::DimInt<1u>, TIdx> &C) {
    executeMatmul(charToCuBlasTranspose(transa), charToCuBlasTranspose(transb),
                  CUBLASLT_EPILOGUE_BIAS, alpha,
                  alpaka::getPtrNative(A), alpaka::getPtrNative(B), beta,
                  alpaka::getPtrNative(bias), alpaka::getPtrNative(C),
                  static_cast<const void *>(alpaka::getPtrNative(bias)),
                  layoutKeyA(transa, m, k), layoutKeyB(transb, k, n), {m, n});
  }

  template <typename T, typename TIdx>
  inline void
  gemm(char transa, char transb, unsigned int m, unsigned int n, unsigned int k,
       float alpha,
       alpaka::ViewPlainPtr<alpaka::DevCudaRt, T, alpaka::DimInt<1u>, TIdx> const &A,
       alpaka::ViewPlainPtr<alpaka::DevCudaRt, T, alpaka::DimInt<1u>, TIdx> const &B,
       float beta,
       alpaka::ViewPlainPtr<alpaka::DevCudaRt, T, alpaka::DimInt<1u>, TIdx> &bias,
       alpaka::ViewPlainPtr<alpaka::DevCudaRt, T, alpaka::DimInt<1u>, TIdx> &C) {
    executeMatmul(charToCuBlasTranspose(transa), charToCuBlasTranspose(transb),
                  CUBLASLT_EPILOGUE_BIAS, alpha,
                  alpaka::getPtrNative(A), alpaka::getPtrNative(B), beta,
                  alpaka::getPtrNative(bias), alpaka::getPtrNative(C),
                  static_cast<const void *>(alpaka::getPtrNative(bias)),
                  layoutKeyA(transa, m, k), layoutKeyB(transb, k, n), {m, n});
  }

  template <typename T>
  inline void gemm(char transa, char transb, unsigned int m, unsigned int n,
                   unsigned int k, float alpha, T const *A, T const *B,
                   float beta, T *bias, T *C) {
    executeMatmul(charToCuBlasTranspose(transa), charToCuBlasTranspose(transb),
                  CUBLASLT_EPILOGUE_BIAS, alpha, A, B, beta, bias, C,
                  static_cast<const void *>(bias),
                  layoutKeyA(transa, m, k), layoutKeyB(transb, k, n), {m, n});
  }

  template <typename T, typename TIdx>
  inline void gemmrelu(char transa, char transb, unsigned int m, unsigned int n,
                       unsigned int k, float alpha,
                       alpaka::BufCudaRt<T, alpaka::DimInt<1u>, TIdx> const &A,
                       alpaka::BufCudaRt<T, alpaka::DimInt<1u>, TIdx> const &B,
                       float beta,
                       alpaka::BufCudaRt<T, alpaka::DimInt<1u>, TIdx> &bias,
                       alpaka::BufCudaRt<T, alpaka::DimInt<1u>, TIdx> &C) {
    executeMatmul(charToCuBlasTranspose(transa), charToCuBlasTranspose(transb),
                  CUBLASLT_EPILOGUE_RELU_BIAS, alpha,
                  alpaka::getPtrNative(A), alpaka::getPtrNative(B), beta,
                  alpaka::getPtrNative(bias), alpaka::getPtrNative(C),
                  static_cast<const void *>(alpaka::getPtrNative(bias)),
                  layoutKeyA(transa, m, k), layoutKeyB(transb, k, n), {m, n});
  }

  template <typename T, typename TIdx>
  inline void gemmrelu(char transa, char transb, unsigned int m, unsigned int n,
                       unsigned int k, float alpha,
                       alpaka::ViewPlainPtr<alpaka::DevCudaRt, T, alpaka::DimInt<1u>, TIdx> const &A,
                       alpaka::ViewPlainPtr<alpaka::DevCudaRt, T, alpaka::DimInt<1u>, TIdx> const &B,
                       float beta,
                       alpaka::ViewPlainPtr<alpaka::DevCudaRt, T, alpaka::DimInt<1u>, TIdx> &bias,
                       alpaka::ViewPlainPtr<alpaka::DevCudaRt, T, alpaka::DimInt<1u>, TIdx> &C) {
    executeMatmul(charToCuBlasTranspose(transa), charToCuBlasTranspose(transb),
                  CUBLASLT_EPILOGUE_RELU_BIAS, alpha,
                  alpaka::getPtrNative(A), alpaka::getPtrNative(B), beta,
                  alpaka::getPtrNative(bias), alpaka::getPtrNative(C),
                  static_cast<const void *>(alpaka::getPtrNative(bias)),
                  layoutKeyA(transa, m, k), layoutKeyB(transb, k, n), {m, n});
  }

  template <typename T>
  inline void gemmrelu(char transa, char transb, unsigned int m, unsigned int n,
                       unsigned int k, float alpha, T const *A, T const *B,
                       float beta, T *bias, T *C) {
    executeMatmul(charToCuBlasTranspose(transa), charToCuBlasTranspose(transb),
                  CUBLASLT_EPILOGUE_RELU_BIAS, alpha, A, B, beta, bias, C,
                  static_cast<const void *>(bias),
                  layoutKeyA(transa, m, k), layoutKeyB(transb, k, n), {m, n});
  }

  template <typename T, typename TIdx>
  inline void gemmgelu(char transa, char transb, unsigned int m, unsigned int n,
                       unsigned int k, float alpha,
                       alpaka::BufCudaRt<T, alpaka::DimInt<1u>, TIdx> const &A,
                       alpaka::BufCudaRt<T, alpaka::DimInt<1u>, TIdx> const &B,
                       float beta,
                       alpaka::BufCudaRt<T, alpaka::DimInt<1u>, TIdx> &bias,
                       alpaka::BufCudaRt<T, alpaka::DimInt<1u>, TIdx> &C) {
    executeMatmul(charToCuBlasTranspose(transa), charToCuBlasTranspose(transb),
                  CUBLASLT_EPILOGUE_GELU_BIAS, alpha,
                  alpaka::getPtrNative(A), alpaka::getPtrNative(B), beta,
                  alpaka::getPtrNative(bias), alpaka::getPtrNative(C),
                  static_cast<const void *>(alpaka::getPtrNative(bias)),
                  layoutKeyA(transa, m, k), layoutKeyB(transb, k, n), {m, n});
  }

  template <typename T, typename TIdx>
  inline void gemmgelu(char transa, char transb, unsigned int m, unsigned int n,
                       unsigned int k, float alpha,
                       alpaka::ViewPlainPtr<alpaka::DevCudaRt, T, alpaka::DimInt<1u>, TIdx> const &A,
                       alpaka::ViewPlainPtr<alpaka::DevCudaRt, T, alpaka::DimInt<1u>, TIdx> const &B,
                       float beta,
                       alpaka::ViewPlainPtr<alpaka::DevCudaRt, T, alpaka::DimInt<1u>, TIdx> &bias,
                       alpaka::ViewPlainPtr<alpaka::DevCudaRt, T, alpaka::DimInt<1u>, TIdx> &C) {
    executeMatmul(charToCuBlasTranspose(transa), charToCuBlasTranspose(transb),
                  CUBLASLT_EPILOGUE_GELU_BIAS, alpha,
                  alpaka::getPtrNative(A), alpaka::getPtrNative(B), beta,
                  alpaka::getPtrNative(bias), alpaka::getPtrNative(C),
                  static_cast<const void *>(alpaka::getPtrNative(bias)),
                  layoutKeyA(transa, m, k), layoutKeyB(transb, k, n), {m, n});
  }

  template <typename T>
  inline void gemmgelu(char transa, char transb, unsigned int m, unsigned int n,
                       unsigned int k, float alpha, T const *A, T const *B,
                       float beta, T *bias, T *C) {
    executeMatmul(charToCuBlasTranspose(transa), charToCuBlasTranspose(transb),
                  CUBLASLT_EPILOGUE_GELU_BIAS, alpha, A, B, beta, bias, C,
                  static_cast<const void *>(bias),
                  layoutKeyA(transa, m, k), layoutKeyB(transb, k, n), {m, n});
  }

  template <typename T, typename TIdx>
  inline void matmul(char transa, char transb, unsigned int m, unsigned int n,
                     unsigned int k, float alpha,
                     alpaka::BufCudaRt<T, alpaka::DimInt<1u>, TIdx> const &A,
                     alpaka::BufCudaRt<T, alpaka::DimInt<1u>, TIdx> const &B,
                     float beta,
                     alpaka::BufCudaRt<T, alpaka::DimInt<1u>, TIdx> &C) {
    float *c = alpaka::getPtrNative(C);
    executeMatmul(charToCuBlasTranspose(transa), charToCuBlasTranspose(transb),
                  CUBLASLT_EPILOGUE_DEFAULT, alpha,
                  alpaka::getPtrNative(A), alpaka::getPtrNative(B), beta,
                  c, c, nullptr,
                  layoutKeyA(transa, m, k), layoutKeyB(transb, k, n), {m, n});
  }

  template <typename T, typename TIdx>
  inline void matmul(char transa, char transb, unsigned int m, unsigned int n,
                     unsigned int k, float alpha,
                     alpaka::ViewPlainPtr<alpaka::DevCudaRt, T, alpaka::DimInt<1u>, TIdx> const &A,
                     alpaka::ViewPlainPtr<alpaka::DevCudaRt, T, alpaka::DimInt<1u>, TIdx> const &B,
                     float beta,
                     alpaka::ViewPlainPtr<alpaka::DevCudaRt, T, alpaka::DimInt<1u>, TIdx> &C) {
    T *c = alpaka::getPtrNative(C);
    executeMatmul(charToCuBlasTranspose(transa), charToCuBlasTranspose(transb),
                  CUBLASLT_EPILOGUE_DEFAULT, alpha,
                  alpaka::getPtrNative(A), alpaka::getPtrNative(B), beta,
                  c, c, nullptr,
                  layoutKeyA(transa, m, k), layoutKeyB(transb, k, n), {m, n});
  }

  // Raw-pointer overload
  template <typename T>
  inline void matmul(char transa, char transb, unsigned int m, unsigned int n,
                     unsigned int k, float alpha, T const *A, T const *B,
                     float beta, T *C) {
    executeMatmul(charToCuBlasTranspose(transa), charToCuBlasTranspose(transb),
                  CUBLASLT_EPILOGUE_DEFAULT, alpha, A, B, beta, C, C, nullptr,
                  layoutKeyA(transa, m, k), layoutKeyB(transb, k, n), {m, n});
  }

  inline void gemmStridedBatched(
      char transa, char transb,
      int m, int n, int k, float alpha,
      const float *A, int lda, long long strideA,
      const float *B, int ldb, long long strideB,
      float beta,
      float *C, int ldc, long long strideC,
      int batchCount)
  {
    CHECK_CUBLAS(cublasSgemmStridedBatched(
        handle,
        charToCuBlasTranspose(transa), charToCuBlasTranspose(transb),
        m, n, k,
        &alpha,
        A, lda, strideA,
        B, ldb, strideB,
        &beta,
        C, ldc, strideC,
        batchCount));
    // No cudaStreamSynchronize — operations remain asynchronous on the stream
  }

private:
  alpaka::QueueCudaRtNonBlocking m_queue;


  static std::pair<std::size_t, std::size_t>
  layoutKeyA(char trans, std::size_t m, std::size_t k) {
    return (trans == 'N' || trans == 'n') ? std::make_pair(m, k)
                                          : std::make_pair(k, m);
  }

  static std::pair<std::size_t, std::size_t>
  layoutKeyB(char trans, std::size_t k, std::size_t n) {
    return (trans == 'N' || trans == 'n') ? std::make_pair(k, n)
                                          : std::make_pair(n, k);
  }

  // Resolve a matrix role's layout at the runtime dims: create the descriptor
  // once, then overwrite its dims in place on later calls. ld = rows (dense,
  // column-major, as the generated calls produce).
  cublasLtMatrixLayout_t stampLayout(LayoutRole role,
                                     const std::pair<std::size_t, std::size_t> &key) {
    const uint64_t rows = key.first, cols = key.second;
    const int64_t  ld   = static_cast<int64_t>(key.first);
    cublasLtMatrixLayout_t &L = roleLayout[role];
    if (!L) {
      CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&L, CUDA_R_32F, rows, cols, ld));
    } else {
      CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(
          L, CUBLASLT_MATRIX_LAYOUT_ROWS, &rows, sizeof(rows)));
      CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(
          L, CUBLASLT_MATRIX_LAYOUT_COLS, &cols, sizeof(cols)));
      CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(
          L, CUBLASLT_MATRIX_LAYOUT_LD, &ld, sizeof(ld)));
    }
    return L;
  }

  static std::size_t bucketUp(std::size_t x) {
    std::size_t b = SOFIEBLAS_LAYOUT_BUCKET_MIN;
    while (b < x) b <<= 1;
    return b;
  }

  // Time the candidates at the exact runtime shape (the buffers are only valid
  // at that size) and return the fastest usable one.
  cublasLtMatmulHeuristicResult_t
  pickFastest(cublasLtMatmulDesc_t desc,
              const std::pair<std::size_t, std::size_t> &kA,
              const std::pair<std::size_t, std::size_t> &kB,
              const std::pair<std::size_t, std::size_t> &kC,
              std::vector<cublasLtMatmulHeuristicResult_t> &cand, int got,
              float alpha, const float *A, const float *B,
              float beta, const float *D_in, float *C_out) {
    auto lA = stampLayout(ROLE_A, kA);
    auto lB = stampLayout(ROLE_B, kB);
    auto lC = stampLayout(ROLE_C, kC);

    cudaEvent_t e0, e1;
    CHECK_CUDA(cudaEventCreate(&e0));
    CHECK_CUDA(cudaEventCreate(&e1));

    const int reps = 3;
    int   best  = -1;
    float bestMs = 0.f;
    for (int i = 0; i < got; ++i) {
      cublasLtMatmulHeuristicResult_t chk{};
      if (cublasLtMatmulAlgoCheck(ltHandle, desc, lA, lB, lC, lC,
                                  &cand[i].algo, &chk) != CUBLAS_STATUS_SUCCESS)
        continue;
      if (chk.workspaceSize > workspaceSize)
        continue;

      auto run = [&] {
        CHECK_CUBLAS(cublasLtMatmul(ltHandle, desc, &alpha, A, lA, B, lB,
                                    &beta, D_in, lC, C_out, lC,
                                    &cand[i].algo, d_workspace, workspaceSize,
                                    stream));
      };
      run();
      CHECK_CUDA(cudaEventRecord(e0, stream));
      for (int r = 0; r < reps; ++r) run();
      CHECK_CUDA(cudaEventRecord(e1, stream));
      CHECK_CUDA(cudaEventSynchronize(e1));
      stats.tuningRuns += reps + 1;

      float ms = 0.f;
      CHECK_CUDA(cudaEventElapsedTime(&ms, e0, e1));
      if (best < 0 || ms < bestMs) { best = i; bestMs = ms; }
    }

    CHECK_CUDA(cudaEventDestroy(e0));
    CHECK_CUDA(cudaEventDestroy(e1));
    return cand[best < 0 ? 0 : best];
  }

  cublasLtMatmulHeuristicResult_t
  resolveProfiled(cublasLtMatmulDesc_t desc, cublasOperation_t transA,
                  cublasOperation_t transB, cublasLtEpilogue_t epilogue,
                  const std::pair<std::size_t, std::size_t> &kA,
                  const std::pair<std::size_t, std::size_t> &kB,
                  const std::pair<std::size_t, std::size_t> &kC,
                  float alpha, const float *A, const float *B,
                  float beta, const float *D_in, float *C_out) {
    const DescKey dk{(int)transA, (int)transB, (int)epilogue};
    const AlgoKey bkey{dk, bucketUp(kA.first), bucketUp(kA.second),
                           bucketUp(kB.first), bucketUp(kB.second)};

    auto it = profileStore.find(bkey);
    if (it == profileStore.end()) {
      // Query at the bucket bound. This only touches host-side descriptors,
      // so it is safe even though the buffers are sized for the runtime shape.
      const std::pair<std::size_t, std::size_t>
          bA{bucketUp(kA.first), bucketUp(kA.second)},
          bB{bucketUp(kB.first), bucketUp(kB.second)},
          bC{bucketUp(kC.first), bucketUp(kC.second)};
      auto lA = stampLayout(ROLE_A, bA);
      auto lB = stampLayout(ROLE_B, bB);
      auto lC = stampLayout(ROLE_C, bC);

      const int want = SOFIEBLAS_LAYOUT_TUNE_CANDIDATES;
      std::vector<cublasLtMatmulHeuristicResult_t> cand(want);
      int got = 0;
      CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(
          ltHandle, desc, lA, lB, lC, lC, preference, want, cand.data(), &got));
      ++stats.heuristicQueries;
      if (got == 0) {
        std::cerr << "[sofieBLAS] No cuBLASLt algorithm for bucket "
                  << bA.first << "x" << bA.second << " * "
                  << bB.first << "x" << bB.second << "\n";
        exit(EXIT_FAILURE);
      }

      AlgoProfile p{};
      // beta != 0 accumulates, so a candidate cannot be run more than once.
      p.tuned = (want > 1 && got > 1 && beta == 0.0f)
                    ? pickFastest(desc, kA, kB, kC, cand, got,
                                  alpha, A, B, beta, D_in, C_out)
                    : cand[0];
#if SOFIEBLAS_LAYOUT_VALIDATE == 1
      p.usable = checkAlgo(desc, p.tuned.algo, kA, kB, kC);
      if (!p.usable) ++stats.algoCheckRejects;
#endif
      it = profileStore.emplace(bkey, p).first;
    }

    AlgoProfile &p = it->second;

#if SOFIEBLAS_LAYOUT_VALIDATE >= 2
    const AlgoKey exact{dk, kA.first, kA.second, kB.first, kB.second};
    if (p.haveLast && p.lastExact == exact) {
      ++stats.memoHits;
      return p.lastValid;
    }
    if (checkAlgo(desc, p.tuned.algo, kA, kB, kC)) {
      p.lastExact = exact;
      p.lastValid = p.tuned;
      p.haveLast  = true;
      return p.tuned;
    }
    ++stats.algoCheckRejects;
#else
    if (p.usable)
      return p.tuned;
#endif

    ++stats.exactFallbacks;
    return getOrComputeAlgo(transA, transB, epilogue, kA, kB, kC);
  }

  bool checkAlgo(cublasLtMatmulDesc_t desc, const cublasLtMatmulAlgo_t &algo,
                 const std::pair<std::size_t, std::size_t> &kA,
                 const std::pair<std::size_t, std::size_t> &kB,
                 const std::pair<std::size_t, std::size_t> &kC) {
    auto lA = stampLayout(ROLE_A, kA);
    auto lB = stampLayout(ROLE_B, kB);
    auto lC = stampLayout(ROLE_C, kC);
    cublasLtMatmulHeuristicResult_t chk{};
    const cublasStatus_t st =
        cublasLtMatmulAlgoCheck(ltHandle, desc, lA, lB, lC, lC, &algo, &chk);
    ++stats.algoChecks;
    return st == CUBLAS_STATUS_SUCCESS && chk.workspaceSize <= workspaceSize;
  }

  cublasLtMatmulDesc_t &getOrCreateDesc(cublasOperation_t transA,
                                         cublasOperation_t transB,
                                         cublasLtEpilogue_t epilogue) {
    DescKey key{(int)transA, (int)transB, (int)epilogue};
    auto it = descStore.find(key);
    if (it != descStore.end())
      return it->second;

    cublasLtMatmulDesc_t desc = nullptr;
    CHECK_CUBLAS(
        cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        desc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        desc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
    // For bias epilogues: set a non-null dummy pointer so the descriptor is
    // valid for cublasLtMatmulAlgoGetHeuristic (real pointer patched per call).
    if (epilogue != CUBLASLT_EPILOGUE_DEFAULT) {
      const void *dummy = d_workspace;
      CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
          desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &dummy, sizeof(dummy)));
    }
    descStore.emplace(key, desc);
    return descStore.at(key);
  }

  cublasLtMatmulHeuristicResult_t &
  getOrComputeAlgo(cublasOperation_t transA, cublasOperation_t transB,
                   cublasLtEpilogue_t epilogue,
                   const std::pair<std::size_t, std::size_t> &kA,
                   const std::pair<std::size_t, std::size_t> &kB,
                   const std::pair<std::size_t, std::size_t> &kC) {
    AlgoKey key{{(int)transA, (int)transB, (int)epilogue},
                kA.first, kA.second, kB.first, kB.second};
    auto it = algoCache.find(key);
    if (it != algoCache.end())
      return it->second;

    auto &desc = getOrCreateDesc(transA, transB, epilogue);
    auto lA = stampLayout(ROLE_A, kA);
    auto lB = stampLayout(ROLE_B, kB);
    auto lC = stampLayout(ROLE_C, kC);   // C and D share the same layout
    cublasLtMatmulHeuristicResult_t h{};
    int returnedResults = 0;
    CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(
        ltHandle, desc, lA, lB, lC, lC,
        preference, 1, &h, &returnedResults));
    ++stats.heuristicQueries;
    if (returnedResults == 0) {
      std::cerr << "[sofieBLAS] No suitable cuBLASLt algorithm found for "
                << "transA=" << transA << " transB=" << transB
                << " epilogue=" << epilogue
                << " A=[" << kA.first << "x" << kA.second << "]"
                << " B=[" << kB.first << "x" << kB.second << "]\n";
      exit(EXIT_FAILURE);
    }
    algoCache.emplace(key, h);
    return algoCache.at(key);
  }

  void executeMatmul(cublasOperation_t transA, cublasOperation_t transB,
                     cublasLtEpilogue_t epilogue,
                     float alpha, const float *A, const float *B,
                     float beta, const float *D_in, float *C_out,
                     const void *bias_ptr,
                     const std::pair<std::size_t, std::size_t> &kA,
                     const std::pair<std::size_t, std::size_t> &kB,
                     const std::pair<std::size_t, std::size_t> &kC) {
    ++stats.matmuls;

    // Descriptor first: tuning may need to run the matmul, bias included.
    auto &desc = getOrCreateDesc(transA, transB, epilogue);
    if (bias_ptr) {
      CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
          desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
          &bias_ptr, sizeof(bias_ptr)));
    }

#if SOFIEBLAS_LAYOUT_PROFILE
    cublasLtMatmulHeuristicResult_t h =
        resolveProfiled(desc, transA, transB, epilogue, kA, kB, kC,
                        alpha, A, B, beta, D_in, C_out);
#else
    cublasLtMatmulHeuristicResult_t h =
        getOrComputeAlgo(transA, transB, epilogue, kA, kB, kC);
#endif

    // Re-stamp the shared role layouts to this call's exact shape. Tuning and
    // validation leave them at other sizes, so this has to happen last.
    auto lA = stampLayout(ROLE_A, kA);
    auto lB = stampLayout(ROLE_B, kB);
    auto lC = stampLayout(ROLE_C, kC);
    CHECK_CUBLAS(cublasLtMatmul(
        ltHandle, desc,
        &alpha, A, lA,
                B, lB,
        &beta, D_in, lC,
               C_out, lC,
        &h.algo, d_workspace, workspaceSize, stream));
  }
};

namespace traits {

template <> class sofieBLAS<alpaka::TagGpuCudaRt> {
public:
  using Impl = BlasCuda;
};

} // namespace traits

#endif // ALPAKA_ACC_GPU_CUDA_ENABLED
