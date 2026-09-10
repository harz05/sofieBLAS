#pragma once

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

#include <cstdlib>
#include <functional>
#include <iostream>
#include <list>
#include <stdexcept>
#include <string>
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

// The cuBLASLt forwarding of the shared BlasLt implementation in
// backends/gpu/detail
struct CublasLtApi {
  using Queue = alpaka::QueueCudaRtNonBlocking;
  using Handle = cublasLtHandle_t;
  using BlasHandle = cublasHandle_t;
  using Preference = cublasLtMatmulPreference_t;
  using Stream = cudaStream_t;
  using Layout = cublasLtMatrixLayout_t;
  using MatmulDesc = cublasLtMatmulDesc_t;
  using HeuristicResult = cublasLtMatmulHeuristicResult_t;
  using Operation = cublasOperation_t;
  using Epilogue = cublasLtEpilogue_t;

  static constexpr auto OpN = CUBLAS_OP_N;
  static constexpr auto OpT = CUBLAS_OP_T;
  static constexpr auto OpC = CUBLAS_OP_C;
  static constexpr auto EpilogueDefault = CUBLASLT_EPILOGUE_DEFAULT;
  static constexpr auto EpilogueBias = CUBLASLT_EPILOGUE_BIAS;
  static constexpr auto EpilogueReluBias = CUBLASLT_EPILOGUE_RELU_BIAS;
  static constexpr auto EpilogueGeluBias = CUBLASLT_EPILOGUE_GELU_BIAS;
  static constexpr auto ComputeF32 = CUBLAS_COMPUTE_32F;
  static constexpr auto RealF32 = CUDA_R_32F;
  static constexpr auto DescTransA = CUBLASLT_MATMUL_DESC_TRANSA;
  static constexpr auto DescTransB = CUBLASLT_MATMUL_DESC_TRANSB;
  static constexpr auto DescEpilogue = CUBLASLT_MATMUL_DESC_EPILOGUE;
  static constexpr auto DescBiasPointer = CUBLASLT_MATMUL_DESC_BIAS_POINTER;
  static constexpr auto PrefMaxWorkspace =
      CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES;
  static constexpr const char *name = "cuBLASLt";

  static constexpr auto ltCreate = cublasLtCreate;
  static constexpr auto ltDestroy = cublasLtDestroy;
  static constexpr auto blasCreate = cublasCreate;
  static constexpr auto blasDestroy = cublasDestroy;
  static constexpr auto blasSetStream = cublasSetStream;
  static constexpr auto prefCreate = cublasLtMatmulPreferenceCreate;
  static constexpr auto prefDestroy = cublasLtMatmulPreferenceDestroy;
  static constexpr auto prefSetAttribute = cublasLtMatmulPreferenceSetAttribute;
  static constexpr auto layoutCreate = cublasLtMatrixLayoutCreate;
  static constexpr auto layoutDestroy = cublasLtMatrixLayoutDestroy;
  static constexpr auto descCreate = cublasLtMatmulDescCreate;
  static constexpr auto descDestroy = cublasLtMatmulDescDestroy;
  static constexpr auto descSetAttribute = cublasLtMatmulDescSetAttribute;
  static constexpr auto getHeuristic = cublasLtMatmulAlgoGetHeuristic;
  static constexpr auto matmul = cublasLtMatmul;
  static constexpr auto sgemmStridedBatched = cublasSgemmStridedBatched;

  // cudaMalloc has a templated C++ overload, so a pointer to it is ambiguous
  static cudaError_t rtMalloc(void **ptr, std::size_t size) {
    return cudaMalloc(ptr, size);
  }
  static cudaError_t rtFree(void *ptr) { return cudaFree(ptr); }
};

#define SOFIEBLAS_CHECK_LT(status) CHECK_CUBLAS(status)
#define SOFIEBLAS_CHECK_RT(err) CHECK_CUDA(err)

#include "../gpu/detail/sofieBLAS_blaslt_common.tpp"

using BlasCuda = BlasLt<CublasLtApi>;

namespace traits {

template <> class sofieBLAS<alpaka::TagGpuCudaRt> {
public:
  using Impl = BlasCuda;
};

} // namespace traits

#endif // ALPAKA_ACC_GPU_CUDA_ENABLED
