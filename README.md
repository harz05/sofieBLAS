# sofieBLAS
sofieBLAS is an abstract C++ (header-only) interface for BLAS operations targeting heterogeneous architectures. It currently supports only ALPAKA buffers and the GEMM operation, acting as a thin, efficient wrapper over existing BLAS libraries such as OpenBLAS, MKL, BLIS, Apple Accelerate, cuBLASLt, and hipBLASLt - allowing the actual backend to be selected through template-based dispatching using traits.

We plan to extend support to more BLAS routines and buffer types in future releases.

## Features

- Unified Interface: Common C++ API over multiple BLAS backends.
- Heterogeneous Support:
  - CPU: OpenBLAS, MKL, BLIS, Apple Accelerate (any CBLAS-compatible library).
  - GPU: NVIDIA (cuBLASLt) and AMD (hipBLASLt).
- Template-Based Dispatching: Backend selection via traits at compile-time, keyed on the Alpaka accelerator tag (e.g. `alpaka::TagCpuSerial`, `alpaka::TagGpuCudaRt`, `alpaka::TagGpuHipRt`).
- Header-Only: Lightweight, easy to integrate- no separate compilation required.
- Minimal Dependency Overhead: Only depends on the backend BLAS libraries of choice.
- One File Per Backend: Each vendor library (CPU or GPU) lives in its own header under `include/sofieBLAS/backends/<cpu|cuda|hip>/`, selected at compile time via a preprocessor macro (`ALPAKA_ACC_GPU_CUDA_ENABLED`, `ALPAKA_ACC_GPU_HIP_ENABLED`, `SOFIEBLAS_USE_OPENBLAS`, `SOFIEBLAS_USE_MKL`, `SOFIEBLAS_USE_BLIS`, `SOFIEBLAS_USE_ACCELERATE`) so only the code for the library you actually build against gets compiled.

## Selecting a backend

Backend selection happens entirely at compile time, in two steps:

1. **Compiler-flag macros decide the backend implementations.** A macro (or macro pair) must be defined to compile in a given backend's header - see the table below. For CPU, if `ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED` is defined but none of the `SOFIEBLAS_USE_*` macros are, sofieBLAS defaults to OpenBLAS; the GPU backends have no such default (each is tied 1:1 to its `ALPAKA_ACC_GPU_*_ENABLED` macro, so there's nothing to default between).
2. **The Alpaka tag you instantiate `sofieBLAS<Tag>` with decides which compiled-in backend a given call site actually uses**, via the `traits::sofieBLAS<Tag>` specialization (e.g. `sofieBLAS<alpaka::TagGpuHipRt>` resolves to the hipBLASLt backend). There is no runtime dispatch - if the macro for that tag's backend wasn't defined, the code simply won't compile.

| Accelerator | Macro(s) | Alpaka tag |
| --- | --- | --- |
| CPU (OpenBLAS) | `ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED` + `SOFIEBLAS_USE_OPENBLAS` | `alpaka::TagCpuSerial` (or other `TagCpu*`) |
| CPU (Intel MKL) | `ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED` + `SOFIEBLAS_USE_MKL` | `alpaka::TagCpuSerial` (or other `TagCpu*`) |
| CPU (BLIS) | `ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED` + `SOFIEBLAS_USE_BLIS` | `alpaka::TagCpuSerial` (or other `TagCpu*`) |
| CPU (Apple Accelerate) | `ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED` + `SOFIEBLAS_USE_ACCELERATE` | `alpaka::TagCpuSerial` (or other `TagCpu*`) |
| NVIDIA GPU (cuBLASLt) | `ALPAKA_ACC_GPU_CUDA_ENABLED` | `alpaka::TagGpuCudaRt` |
| AMD GPU (hipBLASLt) | `ALPAKA_ACC_GPU_HIP_ENABLED` | `alpaka::TagGpuHipRt` |

`sofieBLAS/sofieBLAS.hpp` includes the matching backend header(s) for you based on these macros; nothing else needs to change in your source beyond picking the right tag.

## Building tests and benchmarks

sofieBLAS itself is header-only (`add_subdirectory`/`find_package(sofieBLAS)` gives you an `INTERFACE` target with no build step). Tests and benchmarks are opt-in via CMake options and auto-detect whichever backends are available on the machine for each backend target (`test_cpu`/`test_cuda`/`test_hip`, `bench_cpu`/`bench_cuda`/`bench_hip`) and is skipped if its dependency isn't found:

```bash
cmake -B build -S . -DSOFIEBLAS_BUILD_TESTS=ON -DSOFIEBLAS_BUILD_BENCHMARKS=ON
cmake --build build -j"$(nproc)"

# Run the correctness tests (matmul/gemm/gemmrelu/gemmgelu vs. a reference
# implementation, per available backend)
ctest --test-dir build --output-on-failure

# Run the GEMM throughput benchmark
./build/benchmark/bench_cuda -w 5 -n 20 --sizes 256,512,1024,2048,4096
```

`CPU_BLAS_LIB` (`OpenBLAS`/`MKL`/`BLIS`/`Accelerate`) and `CUDA_BASE`/`ROCM_BASE`/`ONEAPI_BASE`/`BLIS_BASE` are available as `-D` cache variables to point at non-default install locations; alpaka is picked up via `find_package(alpaka)` if already installed, otherwise fetched automatically. See `tests/CMakeLists.txt` and `benchmark/CMakeLists.txt` for the target definitions, and `cmake/SofieBLASBackends.cmake` for the shared backend-detection logic.

## Usage example

```cpp
#include "sofieBLAS/sofieBLAS.hpp"
#include <alpaka/alpaka.hpp>
#include <iostream>

int main() {
  constexpr uint32_t size = 4;

  // Create Alpaka CPU device and blocking queue
  alpaka::PlatformCpu platform;
  auto device = alpaka::getDevByIdx(platform, 0u);
  alpaka::Queue<alpaka::DevCpu, alpaka::Blocking> queue{device};

  // Allocate and initialize matrices A, B, C on CPU
  auto A = alpaka::allocBuf<float, uint32_t>(device, size * size);
  auto B = alpaka::allocBuf<float, uint32_t>(device, size * size);
  auto C = alpaka::allocBuf<float, uint32_t>(device, size * size);
  // (Initialize A and B here...)

  // Create sofieBLAS instance for the CPU backend selected via
  // SOFIEBLAS_USE_OPENBLAS / SOFIEBLAS_USE_MKL / SOFIEBLAS_USE_BLIS / SOFIEBLAS_USE_ACCELERATE
  sofieBLAS<alpaka::TagCpuSerial> blas(queue);

  // C = alpha * op(A) * op(B) + beta * C  (leading dimensions inferred from m, n, k)
  blas.matmul('N', 'N', size, size, size, 1.0f, A, B, 0.0f, C);

  alpaka::wait(queue);
  std::cout << "GEMM completed on CPU backend.\n";

  return 0;
}
```

Switching to a GPU is the same code shape, just a different Alpaka tag, queue/device type, and build-time macro. For example, on AMD (built with `-DALPAKA_ACC_GPU_HIP_ENABLED`):

```cpp
alpaka::PlatformHipRt platform;
auto device = alpaka::getDevByIdx(platform, 0u);
alpaka::Queue<alpaka::DevHipRt, alpaka::NonBlocking> queue{device};
sofieBLAS<alpaka::TagGpuHipRt> blas(queue);

blas.matmul('N', 'N', size, size, size, 1.0f, dA, dB, 0.0f, dC);
```

The GPU backends (`BlasCuda`, `BlasHip`) additionally expose 
- `gemmrelu`/`gemmgelu` (fused bias + activation via cuBLASLt/hipBLASLt epilogues)
- `gemmStridedBatched` for batched gemm operations through strides
- `addOperationConfig` that creates the matrix layouts and resolves the multiply algorithm for a call site's shape ahead of its first call (see below).

## GEMM call instantiation and the algorithm cache

A GEMM call computes `C = alpha * op(A) * op(B) + beta * C`, where A and B are the input matrices, C the output, and `op` an optional transpose. To run one, cuBLASLt and hipBLASLt need three kinds of objects besides the data:

- a **matrix layout** per matrix: a descriptor holding its rows, columns and leading dimension;
- a **matmul descriptor**: the operation settings (the transposes and the epilogue);
- an **algorithm**: the concrete GEMM kernel the library selects for the given settings and dimensions, obtained by querying its heuristic (`cublasLtMatmulAlgoGetHeuristic` / `hipblasLtMatmulAlgoGetHeuristic`). The query runs on the host and is not free.

The CUDA backend (`BlasCuda`, over cuBLASLt) and the HIP backend (`BlasHip`, over hipBLASLt) behave identically: all three objects are created the first time a combination appears and cached, keyed by the exact dimensions plus, for descriptors and algorithms, the transposes and the epilogue. One instance therefore serves GEMM calls at sizes that vary at runtime: a size seen for the first time creates and caches its objects, and a repeated size reuses them without another heuristic query.

### addOperationConfig

`addOperationConfig(m, n, k, lda, ldb, ldc, transa, transb, epilogue)` creates all three objects for one operation (the matrix layouts, the matmul descriptor and the algorithm) for the given dimensions, transposes and epilogue, before the corresponding call is made. It is optional: a combination that was never configured is created and cached on its first call. The `epilogue` argument is the `Epilogue` enum from `sofieBLAS/core.hpp` and names which call the site will make, because the fused epilogue is part of the selected kernel:

| `Epilogue` value | call it configures |
| --- | --- |
| `Epilogue::Default` | `matmul` (no bias) |
| `Epilogue::Bias` | `gemm` (adds the bias vector) |
| `Epilogue::ReluBias` | `gemmrelu` (bias, then ReLU) |
| `Epilogue::GeluBias` | `gemmgelu` (bias, then GELU) |

### Initializing the cache limit

The algorithm cache is unbounded by default. Passing a limit as the second constructor argument caps the number of cached algorithms; when an insertion would exceed the limit, the least recently used entries are evicted. Choose a limit at least as large as the number of distinct shapes the workload uses regularly, or leave it unbounded. `algoCacheSize()` returns the current number of entries.

```cpp
sofieBLAS<alpaka::TagGpuCudaRt> blas(queue);       // unbounded algorithm cache (default)
sofieBLAS<alpaka::TagGpuCudaRt> capped(queue, 32); // at most 32 entries, LRU eviction

blas.addOperationConfig(64, 3, 5, 64, 5, 64, 'N', 'N', Epilogue::Default);
blas.matmul('N', 'N', 64, 3, 5, 1.0f, dA, dB, 0.0f, dC); // created by addOperationConfig: cache hit
blas.matmul('N', 'N', 37, 3, 5, 1.0f, dA, dB, 0.0f, dC); // new size: created on first use
blas.gemmrelu('N', 'N', 64, 3, 5, 1.0f, dA, dB, 0.0f, dBias, dC); // same size, other epilogue: layouts reused, descriptor and algorithm created on first use
```


## Contributing

Contributions, feature suggestions, and issue reports are welcome! Feel free to [open a PR](https://github.com/ML4EP/sofieBLAS/pulls) or an [issue](https://github.com/ML4EP/sofieBLAS/issues).



## License

[GNU General Public License v3.0](https://github.com/ML4EP/sofieBLAS/blob/main/LICENSE)

