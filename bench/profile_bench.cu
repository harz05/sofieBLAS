// Replays ParticleNet's GEMM inventory through BlasCuda over a sweep of
// runtime sizes, checks numerics against a CPU reference, and reports the
// selection counters. Build twice, -DSOFIEBLAS_LAYOUT_PROFILE=0 and =1.

#include <alpaka/alpaka.hpp>
#include <cuda_runtime.h>
#include <unistd.h>

#include "sofieBLAS/backends/cuda/sofieBLAS_cublas.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

// Resident set size in MB, from /proc/self/statm.
static double hostRssMB() {
  FILE *f = std::fopen("/proc/self/statm", "r");
  if (!f) return 0.0;
  long total = 0, rss = 0;
  if (std::fscanf(f, "%ld %ld", &total, &rss) != 2) rss = 0;
  std::fclose(f);
  return (double)rss * (double)sysconf(_SC_PAGESIZE) / (1024.0 * 1024.0);
}

static double gpuUsedMB() {
  size_t freeB = 0, totalB = 0;
  cudaMemGetInfo(&freeB, &totalB);
  return (double)(totalB - freeB) / (1024.0 * 1024.0);
}

// Which runtime quantity drives m for a given call site.
enum MKind { M_NPF, M_NSV, M_TOT, M_TOT8 };

struct Call { MKind kind; int n, k; };

// The 15 cuBLASLt matmuls ParticleNet emits per infer, with the weight dims
// read off the generated header.
static const Call kCalls[] = {
  {M_NPF,   32,  20}, {M_NSV,   32,  11},
  {M_TOT8,  64,  64}, {M_TOT8,  64,  64}, {M_TOT8,  64,  64},
  {M_TOT,   64,  32},
  {M_TOT8,  96, 128}, {M_TOT8,  96,  96}, {M_TOT8,  96,  96},
  {M_TOT,   96,  64},
  {M_TOT8, 128, 192}, {M_TOT8, 128, 128}, {M_TOT8, 128, 128},
  {M_TOT,  128,  96},
  {M_TOT,  256, 288},
};
static const int kNCalls = sizeof(kCalls) / sizeof(kCalls[0]);

static const int NSV = 10, NPF_MIN = 20, NPF_MAX = 120;
static const int MAXM = (NPF_MAX + NSV) * 8;
static const int MAXK = 288, MAXN = 256;

static int mFor(MKind kind, int npf) {
  switch (kind) {
    case M_NPF:  return npf;
    case M_NSV:  return NSV;
    case M_TOT:  return npf + NSV;
    default:     return (npf + NSV) * 8;
  }
}

static void cpuGemm(int m, int n, int k, const float *A, const float *B, float *C) {
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i) {
      float s = 0.f;
      for (int l = 0; l < k; ++l) s += A[i + l * m] * B[l + j * k];
      C[i + j * m] = s;
    }
}

int main(int argc, char **argv) {
  const int nEventsArg = (argc > 1) ? std::atoi(argv[1]) : 1000;
  alpaka::PlatformCudaRt plat{};
  auto device = alpaka::getDevByIdx(plat, 0u);
  alpaka::Queue<alpaka::DevCudaRt, alpaka::NonBlocking> queue(device);

  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, 0);
  std::printf("device: %s  SMs=%d  cc=%d.%d\n", prop.name,
              prop.multiProcessorCount, prop.major, prop.minor);
  std::printf("PROFILE=%d WARMUP=%d\n\n", SOFIEBLAS_LAYOUT_PROFILE,
              SOFIEBLAS_LAYOUT_WARMUP);

  BlasCuda blas(queue);

  // Declare each call site's envelope at the maximum n_pf, as the generated
  // Session constructor does.
  for (int c = 0; c < kNCalls; ++c) {
    const int m = mFor(kCalls[c].kind, NPF_MAX);
    const int n = kCalls[c].n, k = kCalls[c].k;
    blas.addLayoutConfig(m, n, k, m, k, m, 'n', 'n');
  }

  float *dA = nullptr, *dB = nullptr, *dC = nullptr;
  CHECK_CUDA(cudaMalloc(&dA, sizeof(float) * (size_t)MAXM * MAXK));
  CHECK_CUDA(cudaMalloc(&dB, sizeof(float) * (size_t)MAXK * MAXN));
  CHECK_CUDA(cudaMalloc(&dC, sizeof(float) * (size_t)MAXM * MAXN));

  std::vector<float> hA((size_t)MAXM * MAXK), hB((size_t)MAXK * MAXN);
  std::mt19937 rng(1);
  std::uniform_real_distribution<float> u(-1.f, 1.f);
  for (auto &v : hA) v = u(rng);
  for (auto &v : hB) v = u(rng);
  CHECK_CUDA(cudaMemcpy(dA, hA.data(), sizeof(float) * hA.size(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dB, hB.data(), sizeof(float) * hB.size(), cudaMemcpyHostToDevice));

  // Numerics: one shape per distinct (n,k) at a size well below every envelope,
  // so the algorithm in use was resolved for a larger shape.
  {
    std::vector<float> hC, ref;
    float worst = 0.f;
    for (int c = 0; c < kNCalls; ++c) {
      const int m = 37, n = kCalls[c].n, k = kCalls[c].k;
      blas.matmul('n', 'n', (unsigned)m, (unsigned)n, (unsigned)k, 1.0f,
                  (const float *)dA, (const float *)dB, 0.0f, dC);
      CHECK_CUDA(cudaDeviceSynchronize());
      hC.assign((size_t)m * n, 0.f);
      ref.assign((size_t)m * n, 0.f);
      CHECK_CUDA(cudaMemcpy(hC.data(), dC, sizeof(float) * hC.size(), cudaMemcpyDeviceToHost));
      cpuGemm(m, n, k, hA.data(), hB.data(), ref.data());
      for (size_t i = 0; i < ref.size(); ++i)
        worst = std::max(worst, std::fabs(hC[i] - ref[i]));
    }
    std::printf("numerics: worst abs err vs CPU = %.3e  %s\n\n",
                worst, worst < 1e-3f ? "OK" : "FAIL");
  }

  // A run of events with n_pf drawn at random, which is what a real workload
  // looks like: new sizes keep arriving instead of being visited once in order.
  {
    const int nEvents = nEventsArg;
    std::mt19937 r(12345);
    std::uniform_int_distribution<int> pick(NPF_MIN, NPF_MAX);
    std::vector<double> lat;
    lat.reserve(nEvents);

    const auto s0 = blas.layoutStats();
    CHECK_CUDA(cudaDeviceSynchronize());
    const double rss0 = hostRssMB(), gpu0 = gpuUsedMB();
    double first100 = 0.0, total = 0.0;
    for (int e = 0; e < nEvents; ++e) {
      const int npf = pick(r);
      CHECK_CUDA(cudaDeviceSynchronize());
      const auto t0 = std::chrono::high_resolution_clock::now();
      for (int c = 0; c < kNCalls; ++c) {
        const int m = mFor(kCalls[c].kind, npf);
        blas.matmul('n', 'n', (unsigned)m, (unsigned)kCalls[c].n,
                    (unsigned)kCalls[c].k, 1.0f,
                    (const float *)dA, (const float *)dB, 0.0f, dC);
      }
      CHECK_CUDA(cudaDeviceSynchronize());
      const auto t1 = std::chrono::high_resolution_clock::now();
      const double ms =
          std::chrono::duration<double, std::milli>(t1 - t0).count();
      lat.push_back(ms);
      total += ms;
      if (e < 100) first100 += ms;
    }
    const auto s1 = blas.layoutStats();
    CHECK_CUDA(cudaDeviceSynchronize());
    const double rss1 = hostRssMB(), gpu1 = gpuUsedMB();

    std::vector<double> srt = lat;
    std::sort(srt.begin(), srt.end());
    auto pct = [&](double p) { return srt[(size_t)(p * (srt.size() - 1))]; };

    std::printf("events=%d  total=%.1f ms  first100=%.2f ms\n", nEvents, total,
                first100);
    std::printf("per-event ms: mean=%.4f p50=%.4f p95=%.4f p99=%.4f max=%.4f\n",
                total / nEvents, pct(0.50), pct(0.95), pct(0.99), srt.back());
    std::printf("memory MB: hostRss=%.3f growth=%.3f  gpuUsed=%.1f growth=%.1f\n",
                rss1, rss1 - rss0, gpu1, gpu1 - gpu0);
    std::printf("counters: matmuls=%zu heur=%zu envMisses=%zu\n",
                s1.matmuls - s0.matmuls,
                s1.heuristicQueries - s0.heuristicQueries,
                s1.envelopeMisses - s0.envelopeMisses);
  }

  const auto s = blas.layoutStats();
  std::printf("\ntotals: envelopes=%zu algoCache=%zu heuristicQueries=%zu "
              "(warmup=%zu)\n",
              blas.envelopeCount(), blas.algoCacheSize(), s.heuristicQueries,
              s.warmupQueries);
  return 0;
}
