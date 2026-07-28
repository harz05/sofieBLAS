// End-to-end ParticleNet latency over a run of events with random n_pf.
// Needs the SOFIE-generated header, so build it standalone against a SOFIE
// build tree. Build twice, -DSOFIEBLAS_LAYOUT_PROFILE=0 and =1.

#include <alpaka/alpaka.hpp>
#include <cuda_runtime.h>

#include "particle-net_FromONNX_GPU_ALPAKA.hxx"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

using Idx = std::size_t;
using Dim = alpaka::DimInt<1>;
using Ext1D = alpaka::Vec<Dim, Idx>;

int main(int argc, char **argv) {
  const char *weights =
      (argc > 1) ? argv[1] : "particle-net_FromONNX_GPU_ALPAKA.dat";
  const int nEvents = (argc > 2) ? std::atoi(argv[2]) : 1000;

  const size_t N = 1, NSV = 10, NPF_MAX = 120, NPF_MIN = 20;

  alpaka::PlatformCpu hp{};
  auto host = alpaka::getDevByIdx(hp, 0u);
  alpaka::PlatformCudaRt pp{};
  auto device = alpaka::getDevByIdx(pp, 0u);
  alpaka::Queue<alpaka::DevCudaRt, alpaka::NonBlocking> q(device);

  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, 0);
  std::printf("device: %s  SMs=%d  cc=%d.%d\n", prop.name,
              prop.multiProcessorCount, prop.major, prop.minor);
  std::printf("SOFIEBLAS_LAYOUT_PROFILE=%d\n\n", SOFIEBLAS_LAYOUT_PROFILE);

  std::mt19937 rng(1);
  std::uniform_real_distribution<float> u(-1.f, 1.f);
  auto mk = [&](size_t n) {
    auto d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{n}));
    auto h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{n}));
    float *p = alpaka::getPtrNative(h);
    for (size_t i = 0; i < n; ++i) p[i] = u(rng);
    alpaka::memcpy(q, d, h);
    alpaka::wait(q);
    return d;
  };

  auto pfp = mk(N * 2 * NPF_MAX), pff = mk(N * 20 * NPF_MAX), pfm = mk(N * NPF_MAX);
  auto svp = mk(N * 2 * NSV), svf = mk(N * 11 * NSV), svm = mk(N * NSV);

  SOFIE_particle_net::Session<alpaka::TagGpuCudaRt> s(weights, N, NPF_MAX, NSV);

  auto one = [&](size_t npf) {
    auto out = s.infer(N, npf, pfp, pff, pfm, NSV, svp, svf, svm);
    cudaDeviceSynchronize();
    return out;
  };

  one(NPF_MAX);  // absorb CUDA context and cuBLASLt library warmup

  std::mt19937 r(12345);
  std::uniform_int_distribution<int> pick((int)NPF_MIN, (int)NPF_MAX);
  std::vector<double> lat;
  lat.reserve(nEvents);
  double total = 0.0, first100 = 0.0;

  for (int e = 0; e < nEvents; ++e) {
    const size_t npf = (size_t)pick(r);
    const auto t0 = std::chrono::high_resolution_clock::now();
    auto out = one(npf);
    const auto t1 = std::chrono::high_resolution_clock::now();
    (void)out;
    const double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    lat.push_back(ms);
    total += ms;
    if (e < 100) first100 += ms;
  }

  std::vector<double> srt = lat;
  std::sort(srt.begin(), srt.end());
  auto pct = [&](double p) { return srt[(size_t)(p * (srt.size() - 1))]; };

  std::printf("events=%d  total=%.1f ms  first100=%.2f ms\n", nEvents, total,
              first100);
  std::printf("per-event ms: mean=%.4f p50=%.4f p95=%.4f p99=%.4f max=%.4f\n",
              total / nEvents, pct(0.50), pct(0.95), pct(0.99), srt.back());
  return 0;
}
