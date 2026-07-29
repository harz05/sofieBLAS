#pragma once

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
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

// 0 = resolve the algorithm per runtime shape, so algoCache holds one entry
//     per distinct size and every new size pays a heuristic query
// 1 = resolve it per call site, at the envelope addLayoutConfig declares, so
//     every size that call site produces shares one entry
// The matmul runs at the exact runtime dims either way; only the choice of
// algorithm is shared.
#ifndef SOFIEBLAS_LAYOUT_PROFILE
#define SOFIEBLAS_LAYOUT_PROFILE 1
#endif

// Resolve each declared envelope in the constructor, so the first infer()
// finds every algorithm already cached.
#ifndef SOFIEBLAS_LAYOUT_WARMUP
#define SOFIEBLAS_LAYOUT_WARMUP 1
#endif

// A call site's maximum shape, as declared by addLayoutConfig from the
// generated Session constructor.
struct ShapeEnvelope {
  std::size_t rowsA, colsA, rowsB, colsB, rowsC, colsC;
};

struct LayoutStats {
  std::size_t matmuls          = 0;
  std::size_t heuristicQueries = 0;
  std::size_t warmupQueries    = 0;  // of which resolved in the constructor
  std::size_t envelopeMisses   = 0;  // calls no declared envelope covered
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

  // call-site envelopes declared by addLayoutConfig
  std::vector<ShapeEnvelope> envelopes;

  LayoutStats stats;

public:
  const LayoutStats &layoutStats() const { return stats; }
  std::size_t algoCacheSize()     const { return algoCache.size(); }
  std::size_t envelopeCount()     const { return envelopes.size(); }

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

  // Records the call site's envelope. The generated constructor evaluates its
  // shape expressions with its own parameters, so for a dynamic model these are
  // the largest dims the call site will ever use. Layouts are created lazily,
  // so nothing is registered here beyond the envelope and, with warmup on, the
  // algorithm resolved for it.
  void addLayoutConfig(std::size_t m, std::size_t n, std::size_t k,
                       std::size_t, std::size_t, std::size_t,
                       char transa, char transb) {
    const auto kA = layoutKeyA(transa, m, k);
    const auto kB = layoutKeyB(transb, k, n);
    const std::pair<std::size_t, std::size_t> kC{m, n};
    envelopes.push_back({kA.first, kA.second, kB.first, kB.second, m, n});

#if SOFIEBLAS_LAYOUT_PROFILE && SOFIEBLAS_LAYOUT_WARMUP
    // The constructor does not know which epilogue this call site uses, so
    // resolve all three. Unused ones cost one heuristic query each, off the
    // inference path.
    const cublasOperation_t tA = charToCuBlasTranspose(transa);
    const cublasOperation_t tB = charToCuBlasTranspose(transb);
    const cublasLtEpilogue_t eps[] = {CUBLASLT_EPILOGUE_DEFAULT,
                                      CUBLASLT_EPILOGUE_BIAS,
                                      CUBLASLT_EPILOGUE_RELU_BIAS};
    for (cublasLtEpilogue_t ep : eps) {
      const std::size_t before = stats.heuristicQueries;
      getOrComputeAlgo(tA, tB, ep, kA, kB, kC, /*required=*/false);
      stats.warmupQueries += stats.heuristicQueries - before;
    }
#endif
  }

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

  // Tightest declared envelope covering this call, or null if none does.
  // Tightest matters: several envelopes may cover a small shape, but only the
  // call site's own matches its weight dims exactly and so has zero excess
  // on those axes.
  const ShapeEnvelope *
  findEnvelope(const std::pair<std::size_t, std::size_t> &kA,
               const std::pair<std::size_t, std::size_t> &kB,
               const std::pair<std::size_t, std::size_t> &kC) const {
    const ShapeEnvelope *best = nullptr;
    std::size_t bestExcess = std::numeric_limits<std::size_t>::max();
    for (const auto &e : envelopes) {
      if (e.rowsA < kA.first || e.colsA < kA.second ||
          e.rowsB < kB.first || e.colsB < kB.second ||
          e.rowsC < kC.first || e.colsC < kC.second)
        continue;
      const std::size_t ex = (e.rowsA - kA.first) + (e.colsA - kA.second) +
                             (e.rowsB - kB.first) + (e.colsB - kB.second);
      if (ex < bestExcess) { bestExcess = ex; best = &e; }
    }
    return best;
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

  // required=false is used by constructor warmup, which speculatively resolves
  // epilogues the call site may never use: those may legitimately have no
  // algorithm and must not abort.
  cublasLtMatmulHeuristicResult_t *
  getOrComputeAlgo(cublasOperation_t transA, cublasOperation_t transB,
                   cublasLtEpilogue_t epilogue,
                   const std::pair<std::size_t, std::size_t> &kA,
                   const std::pair<std::size_t, std::size_t> &kB,
                   const std::pair<std::size_t, std::size_t> &kC,
                   bool required = true) {
    AlgoKey key{{(int)transA, (int)transB, (int)epilogue},
                kA.first, kA.second, kB.first, kB.second};
    auto it = algoCache.find(key);
    if (it != algoCache.end())
      return &it->second;

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
      if (!required)
        return nullptr;
      std::cerr << "[sofieBLAS] No suitable cuBLASLt algorithm found for "
                << "transA=" << transA << " transB=" << transB
                << " epilogue=" << epilogue
                << " A=[" << kA.first << "x" << kA.second << "]"
                << " B=[" << kB.first << "x" << kB.second << "]\n";
      exit(EXIT_FAILURE);
    }
    return &algoCache.emplace(key, h).first->second;
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

    auto &desc = getOrCreateDesc(transA, transB, epilogue);
    if (bias_ptr) {
      CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
          desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
          &bias_ptr, sizeof(bias_ptr)));
    }

#if SOFIEBLAS_LAYOUT_PROFILE
    // Resolve the algorithm at this call site's declared envelope, so every
    // runtime size it produces shares one cache entry. An algorithm chosen for
    // a larger shape stays valid at a smaller one; if cuBLASLt ever disagreed
    // the matmul below would report it rather than compute silently wrong.
    const ShapeEnvelope *env = findEnvelope(kA, kB, kC);
    if (!env)
      ++stats.envelopeMisses;
    const std::pair<std::size_t, std::size_t>
        aA = env ? std::make_pair(env->rowsA, env->colsA) : kA,
        aB = env ? std::make_pair(env->rowsB, env->colsB) : kB,
        aC = env ? std::make_pair(env->rowsC, env->colsC) : kC;
    auto *h = getOrComputeAlgo(transA, transB, epilogue, aA, aB, aC);
#else
    auto *h = getOrComputeAlgo(transA, transB, epilogue, kA, kB, kC);
#endif

    // Re-stamp the shared role layouts to this call's exact shape. Resolving
    // the algorithm leaves them at the envelope size, so this happens last.
    auto lA = stampLayout(ROLE_A, kA);
    auto lB = stampLayout(ROLE_B, kB);
    auto lC = stampLayout(ROLE_C, kC);
    CHECK_CUBLAS(cublasLtMatmul(
        ltHandle, desc,
        &alpha, A, lA,
                B, lB,
        &beta, D_in, lC,
               C_out, lC,
        &h->algo, d_workspace, workspaceSize, stream));
  }
};

namespace traits {

template <> class sofieBLAS<alpaka::TagGpuCudaRt> {
public:
  using Impl = BlasCuda;
};

} // namespace traits

#endif // ALPAKA_ACC_GPU_CUDA_ENABLED
