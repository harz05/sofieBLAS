#include "sofieBLAS/sofieBLAS.hpp"
#include <alpaka/alpaka.hpp>

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using Idx = uint32_t;

struct BenchOptions {
  int warmup = 5;
  int iterations = 20;
  std::vector<int> sizes = {256, 512, 1024, 2048, 4096};
};

static BenchOptions parseArgs(int argc, char **argv) {
  BenchOptions opt;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto nextVal = [&](void) -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << arg << "\n";
        std::exit(EXIT_FAILURE);
      }
      return argv[++i];
    };
    if (arg == "-w" || arg == "--warmup") {
      opt.warmup = std::stoi(nextVal());
    } else if (arg == "-n" || arg == "--iterations") {
      opt.iterations = std::stoi(nextVal());
    } else if (arg == "--sizes") {
      opt.sizes.clear();
      std::stringstream ss(nextVal());
      std::string tok;
      while (std::getline(ss, tok, ','))
        opt.sizes.push_back(std::stoi(tok));
    } else if (arg == "-h" || arg == "--help") {
      std::cout << "Usage: " << argv[0]
                << " [-w warmup] [-n iterations] [--sizes s1,s2,...]\n";
      std::exit(EXIT_SUCCESS);
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      std::exit(EXIT_FAILURE);
    }
  }
  return opt;
}

static void fillSeq(float *M, int n, float start = 0.01f, float step = 0.001f) {
  for (int i = 0; i < n; ++i)
    M[i] = start + static_cast<float>(i % 997) * step;
}

static void printHeader() {
  std::cout << std::left << std::setw(10) << "Backend" << std::right
            << std::setw(10) << "M=N=K" << std::setw(14) << "avg (ms)"
            << std::setw(14) << "GFLOP/s" << "\n";
  std::cout << std::string(48, '-') << "\n";
}

static void printRow(const std::string &backend, int size, double avgMs,
                     double gflops) {
  std::cout << std::left << std::setw(10) << backend << std::right
            << std::setw(10) << size << std::setw(14) << std::fixed
            << std::setprecision(3) << avgMs << std::setw(14)
            << std::setprecision(2) << gflops << "\n";
}

static double gflopsFor(int size, double avgSeconds) {
  const double flops = 2.0 * static_cast<double>(size) *
                       static_cast<double>(size) * static_cast<double>(size);
  return flops / avgSeconds / 1e9;
}

// ---------------------------------------------------------------------------
// CPU benchmark
// ---------------------------------------------------------------------------
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED

static void runCpuBench(const BenchOptions &opt) {
  alpaka::PlatformCpu platform{};
  auto dev = alpaka::getDevByIdx(platform, 0u);
  alpaka::Queue<alpaka::DevCpu, alpaka::Blocking> queue{dev};
  sofieBLAS<alpaka::TagCpuSerial> blas(queue);

  for (int size : opt.sizes) {
    const int M = size, N = size, K = size;
    auto hA = alpaka::allocBuf<float, Idx>(dev, static_cast<Idx>(M * K));
    auto hB = alpaka::allocBuf<float, Idx>(dev, static_cast<Idx>(K * N));
    auto hC = alpaka::allocBuf<float, Idx>(dev, static_cast<Idx>(M * N));
    fillSeq(alpaka::getPtrNative(hA), M * K);
    fillSeq(alpaka::getPtrNative(hB), K * N, 0.02f, 0.0005f);

    for (int i = 0; i < opt.warmup; ++i)
      blas.matmul('N', 'N', M, N, K, 1.f, hA, hB, 0.f, hC);

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < opt.iterations; ++i)
      blas.matmul('N', 'N', M, N, K, 1.f, hA, hB, 0.f, hC);
    auto t1 = std::chrono::steady_clock::now();

    double avgSeconds =
        std::chrono::duration<double>(t1 - t0).count() / opt.iterations;
    printRow("cpu", size, avgSeconds * 1e3, gflopsFor(size, avgSeconds));
  }
}

#endif // ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED

// ---------------------------------------------------------------------------
// CUDA benchmark
// ---------------------------------------------------------------------------
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

static void runCudaBench(const BenchOptions &opt) {
  alpaka::PlatformCudaRt platform{};
  auto dev = alpaka::getDevByIdx(platform, 0u);
  alpaka::Queue<alpaka::DevCudaRt, alpaka::NonBlocking> queue{dev};
  sofieBLAS<alpaka::TagGpuCudaRt> blas(queue);

  alpaka::PlatformCpu hostPlatform{};
  auto hostDev = alpaka::getDevByIdx(hostPlatform, 0u);

  for (int size : opt.sizes) {
    const int M = size, N = size, K = size;
    auto hA = alpaka::allocBuf<float, Idx>(hostDev, static_cast<Idx>(M * K));
    auto hB = alpaka::allocBuf<float, Idx>(hostDev, static_cast<Idx>(K * N));
    fillSeq(alpaka::getPtrNative(hA), M * K);
    fillSeq(alpaka::getPtrNative(hB), K * N, 0.02f, 0.0005f);

    auto dA = alpaka::allocAsyncBuf<float, Idx>(queue, static_cast<Idx>(M * K));
    auto dB = alpaka::allocAsyncBuf<float, Idx>(queue, static_cast<Idx>(K * N));
    auto dC = alpaka::allocAsyncBuf<float, Idx>(queue, static_cast<Idx>(M * N));
    alpaka::memcpy(queue, dA, hA);
    alpaka::memcpy(queue, dB, hB);
    alpaka::wait(queue);

    blas.addOperationConfig(M, N, K, M, K, M, 'N', 'N', Epilogue::Default);

    for (int i = 0; i < opt.warmup; ++i)
      blas.matmul('N', 'N', M, N, K, 1.f, dA, dB, 0.f, dC);
    alpaka::wait(queue);

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < opt.iterations; ++i)
      blas.matmul('N', 'N', M, N, K, 1.f, dA, dB, 0.f, dC);
    alpaka::wait(queue);
    auto t1 = std::chrono::steady_clock::now();

    double avgSeconds =
        std::chrono::duration<double>(t1 - t0).count() / opt.iterations;
    printRow("cuda", size, avgSeconds * 1e3, gflopsFor(size, avgSeconds));
  }
}

#endif // ALPAKA_ACC_GPU_CUDA_ENABLED

// ---------------------------------------------------------------------------
// HIP benchmark
// ---------------------------------------------------------------------------
#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

static void runHipBench(const BenchOptions &opt) {
  alpaka::PlatformHipRt platform{};
  auto dev = alpaka::getDevByIdx(platform, 0u);
  alpaka::Queue<alpaka::DevHipRt, alpaka::NonBlocking> queue{dev};
  sofieBLAS<alpaka::TagGpuHipRt> blas(queue);

  alpaka::PlatformCpu hostPlatform{};
  auto hostDev = alpaka::getDevByIdx(hostPlatform, 0u);

  for (int size : opt.sizes) {
    const int M = size, N = size, K = size;
    auto hA = alpaka::allocBuf<float, Idx>(hostDev, static_cast<Idx>(M * K));
    auto hB = alpaka::allocBuf<float, Idx>(hostDev, static_cast<Idx>(K * N));
    fillSeq(alpaka::getPtrNative(hA), M * K);
    fillSeq(alpaka::getPtrNative(hB), K * N, 0.02f, 0.0005f);

    auto dA = alpaka::allocAsyncBuf<float, Idx>(queue, static_cast<Idx>(M * K));
    auto dB = alpaka::allocAsyncBuf<float, Idx>(queue, static_cast<Idx>(K * N));
    auto dC = alpaka::allocAsyncBuf<float, Idx>(queue, static_cast<Idx>(M * N));
    alpaka::memcpy(queue, dA, hA);
    alpaka::memcpy(queue, dB, hB);
    alpaka::wait(queue);

    blas.addOperationConfig(M, N, K, M, K, M, 'N', 'N', Epilogue::Default);

    for (int i = 0; i < opt.warmup; ++i)
      blas.matmul('N', 'N', M, N, K, 1.f, dA, dB, 0.f, dC);
    alpaka::wait(queue);

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < opt.iterations; ++i)
      blas.matmul('N', 'N', M, N, K, 1.f, dA, dB, 0.f, dC);
    alpaka::wait(queue);
    auto t1 = std::chrono::steady_clock::now();

    double avgSeconds =
        std::chrono::duration<double>(t1 - t0).count() / opt.iterations;
    printRow("hip", size, avgSeconds * 1e3, gflopsFor(size, avgSeconds));
  }
}

#endif // ALPAKA_ACC_GPU_HIP_ENABLED

int main(int argc, char **argv) {
  BenchOptions opt = parseArgs(argc, argv);

  std::cout << "sofieBLAS benchmark  (warmup=" << opt.warmup
            << ", iterations=" << opt.iterations << ")\n\n";
  printHeader();

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
  runCpuBench(opt);
#endif
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
  runCudaBench(opt);
#endif
#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
  runHipBench(opt);
#endif

  return EXIT_SUCCESS;
}
