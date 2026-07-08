#pragma once

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

#include <cstdlib>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

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


struct PairHash {
  std::size_t
  operator()(const std::pair<std::size_t, std::size_t> &p) const noexcept {
    std::size_t h1 = std::hash<std::size_t>{}(p.first);
    std::size_t h2 = std::hash<std::size_t>{}(p.second);
    return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
  }
};

struct PairEq {
  bool operator()(const std::pair<std::size_t, std::size_t> &a,
                  const std::pair<std::size_t, std::size_t> &b) const noexcept {
    return a.first == b.first && a.second == b.second;
  }
};

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
  std::size_t rowsA, colsA;  // physical dimensions of A in layoutStore
  std::size_t rowsB, colsB;  // physical dimensions of B in layoutStore
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

class BlasCuda {
  cublasLtHandle_t           ltHandle   = nullptr;
  cublasHandle_t             handle     = nullptr;  // legacy cuBLAS for batched ops
  cublasLtMatmulPreference_t preference = nullptr;
  void *d_workspace  = nullptr;
  size_t workspaceSize = 1u << 25;  // 32 MB (was 4 MB)
  cudaStream_t stream = nullptr;

  std::unordered_map<std::pair<std::size_t, std::size_t>,
                     cublasLtMatrixLayout_t, PairHash, PairEq>
      layoutStore;

  std::unordered_map<DescKey, cublasLtMatmulDesc_t, DescKeyHash> descStore;

  std::unordered_map<AlgoKey, cublasLtMatmulHeuristicResult_t, AlgoKeyHash>
      algoCache;

public:
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
    for (auto &[key, layout] : layoutStore)
      if (layout) cublasLtMatrixLayoutDestroy(layout);
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

  void addLayoutConfig(std::size_t m, std::size_t n, std::size_t k,
                       std::size_t lda, std::size_t ldb, std::size_t ldc,
                       char transa, char transb) {
    // Physical A: (m×k) if NoTrans, (k×m) if Trans
    if (transa == 'N' || transa == 'n')
      checkAndAddLayout(m, k, lda);
    else
      checkAndAddLayout(k, m, lda);
    // Physical B: (k×n) if NoTrans, (n×k) if Trans
    if (transb == 'N' || transb == 'n')
      checkAndAddLayout(k, n, ldb);
    else
      checkAndAddLayout(n, k, ldb);
    // C is always (m×n)
    checkAndAddLayout(m, n, ldc);
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

  void checkAndAddLayout(std::size_t rows, std::size_t cols, std::size_t ld) {
    auto key = std::make_pair(rows, cols);
    if (layoutStore.find(key) == layoutStore.end()) {
      cublasLtMatrixLayout_t layout = nullptr;
      CHECK_CUBLAS(
          cublasLtMatrixLayoutCreate(&layout, CUDA_R_32F, rows, cols, ld));
      layoutStore.emplace(key, layout);
    }
  }

  // create the layout on lookup if missing (runtime dims with dynamic shapes),
  // ld = rows since the store assumes dense layouts
  cublasLtMatrixLayout_t getOrCreateLayout(const std::pair<std::size_t, std::size_t> &key) {
    checkAndAddLayout(key.first, key.second, key.first);
    return layoutStore.at(key);
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
    cublasLtMatmulHeuristicResult_t h{};
    int returnedResults = 0;
    CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(
        ltHandle, desc,
        getOrCreateLayout(kA), getOrCreateLayout(kB),
        getOrCreateLayout(kC), getOrCreateLayout(kC),
        preference, 1, &h, &returnedResults));
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
    // Retrieve (or lazily compute) the cached algorithm for this shape
    auto &h = getOrComputeAlgo(transA, transB, epilogue, kA, kB, kC);

    // Retrieve the cached descriptor and patch the real bias pointer in-place
    auto &desc = getOrCreateDesc(transA, transB, epilogue);
    if (bias_ptr) {
      CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
          desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
          &bias_ptr, sizeof(bias_ptr)));
    }

    CHECK_CUBLAS(cublasLtMatmul(
        ltHandle, desc,
        &alpha, A, layoutStore.at(kA),
                B, layoutStore.at(kB),
        &beta, D_in, layoutStore.at(kC),
               C_out, layoutStore.at(kC),
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
