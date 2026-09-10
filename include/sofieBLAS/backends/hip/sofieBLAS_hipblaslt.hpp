#pragma once

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

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
#include <hipblas/hipblas.h>
#include <hipblaslt/hipblaslt.h>

#define CHECK_HIP(err)                                                         \
  if ((err) != hipSuccess) {                                                   \
    std::cerr << "HIP error: " << hipGetErrorString(err) << " at line "        \
              << __LINE__ << "\n";                                             \
    exit(EXIT_FAILURE);                                                        \
  }

#define CHECK_HIPBLAS(status)                                                  \
  do {                                                                         \
    hipblasStatus_t _s = (status);                                             \
    if (_s != HIPBLAS_STATUS_SUCCESS) {                                        \
      std::cerr << "hipBLAS error " << _s << " at line " << __LINE__ << "\n";  \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// The hipBLASLt forwarding of the shared BlasLt implementation in
// backends/gpu/detail
struct HipblasLtApi {
  using Queue = alpaka::QueueHipRtNonBlocking;
  using Handle = hipblasLtHandle_t;
  using BlasHandle = hipblasHandle_t;
  using Preference = hipblasLtMatmulPreference_t;
  using Stream = hipStream_t;
  using Layout = hipblasLtMatrixLayout_t;
  using MatmulDesc = hipblasLtMatmulDesc_t;
  using HeuristicResult = hipblasLtMatmulHeuristicResult_t;
  using Operation = hipblasOperation_t;
  using Epilogue = hipblasLtEpilogue_t;

  static constexpr auto OpN = HIPBLAS_OP_N;
  static constexpr auto OpT = HIPBLAS_OP_T;
  static constexpr auto OpC = HIPBLAS_OP_C;
  static constexpr auto EpilogueDefault = HIPBLASLT_EPILOGUE_DEFAULT;
  static constexpr auto EpilogueBias = HIPBLASLT_EPILOGUE_BIAS;
  static constexpr auto EpilogueReluBias = HIPBLASLT_EPILOGUE_RELU_BIAS;
  static constexpr auto EpilogueGeluBias = HIPBLASLT_EPILOGUE_GELU_BIAS;
  static constexpr auto ComputeF32 = HIPBLAS_COMPUTE_32F;
  static constexpr auto RealF32 = HIP_R_32F;
  static constexpr auto DescTransA = HIPBLASLT_MATMUL_DESC_TRANSA;
  static constexpr auto DescTransB = HIPBLASLT_MATMUL_DESC_TRANSB;
  static constexpr auto DescEpilogue = HIPBLASLT_MATMUL_DESC_EPILOGUE;
  static constexpr auto DescBiasPointer = HIPBLASLT_MATMUL_DESC_BIAS_POINTER;
  static constexpr auto PrefMaxWorkspace =
      HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES;
  static constexpr const char *name = "hipBLASLt";

  static constexpr auto ltCreate = hipblasLtCreate;
  static constexpr auto ltDestroy = hipblasLtDestroy;
  static constexpr auto blasCreate = hipblasCreate;
  static constexpr auto blasDestroy = hipblasDestroy;
  static constexpr auto blasSetStream = hipblasSetStream;
  static constexpr auto prefCreate = hipblasLtMatmulPreferenceCreate;
  static constexpr auto prefDestroy = hipblasLtMatmulPreferenceDestroy;
  static constexpr auto prefSetAttribute =
      hipblasLtMatmulPreferenceSetAttribute;
  static constexpr auto layoutCreate = hipblasLtMatrixLayoutCreate;
  static constexpr auto layoutDestroy = hipblasLtMatrixLayoutDestroy;
  static constexpr auto descCreate = hipblasLtMatmulDescCreate;
  static constexpr auto descDestroy = hipblasLtMatmulDescDestroy;
  static constexpr auto descSetAttribute = hipblasLtMatmulDescSetAttribute;
  static constexpr auto getHeuristic = hipblasLtMatmulAlgoGetHeuristic;
  static constexpr auto matmul = hipblasLtMatmul;
  static constexpr auto sgemmStridedBatched = hipblasSgemmStridedBatched;

  // hipMalloc has a templated C++ overload, so a pointer to it is ambiguous
  static hipError_t rtMalloc(void **ptr, std::size_t size) {
    return hipMalloc(ptr, size);
  }
  static hipError_t rtFree(void *ptr) { return hipFree(ptr); }
};

#define SOFIEBLAS_CHECK_LT(status) CHECK_HIPBLAS(status)
#define SOFIEBLAS_CHECK_RT(err) CHECK_HIP(err)

#include "../gpu/detail/sofieBLAS_blaslt_common.tpp"

using BlasHip = BlasLt<HipblasLtApi>;

namespace traits {

template <> class sofieBLAS<alpaka::TagGpuHipRt> {
public:
  using Impl = BlasHip;
};

} // namespace traits

#endif // ALPAKA_ACC_GPU_HIP_ENABLED
