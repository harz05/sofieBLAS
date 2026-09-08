#include "sofieBLAS/sofieBLAS.hpp"
#include <alpaka/alpaka.hpp>

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using Idx = uint32_t;
using Dim1D = alpaka::DimInt<1u>;

// ---------------------------------------------------------------------------
// Reference implementations (column-major, float)
// ---------------------------------------------------------------------------
static inline float cm(const float *M, int row, int col, int ld) {
  return M[col * ld + row];
}

// C = alpha * op(A) * op(B) + beta * C  (in-place, column-major)
static void refMatmul(float *C, const float *A, const float *B, int m, int n,
                      int k, float alpha, float beta, bool transA,
                      bool transB) {
  int lda = transA ? k : m;
  int ldb = transB ? n : k;
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < m; ++i) {
      float sum = 0.f;
      for (int p = 0; p < k; ++p) {
        float a = transA ? cm(A, p, i, lda) : cm(A, i, p, lda);
        float b = transB ? cm(B, j, p, ldb) : cm(B, p, j, ldb);
        sum += a * b;
      }
      C[j * m + i] = alpha * sum + beta * C[j * m + i];
    }
  }
}

// C = alpha * op(A) * op(B) + beta * bias_matrix + bias_vec (per-row broadcast)
static void refGemm(float *C, const float *A, const float *B, const float *bias,
                    int m, int n, int k, float alpha, float beta, bool transA,
                    bool transB) {
  int lda = transA ? k : m;
  int ldb = transB ? n : k;
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < m; ++i) {
      float sum = 0.f;
      for (int p = 0; p < k; ++p) {
        float a = transA ? cm(A, p, i, lda) : cm(A, i, p, lda);
        float b = transB ? cm(B, j, p, ldb) : cm(B, p, j, ldb);
        sum += a * b;
      }
      C[j * m + i] = alpha * sum + beta * bias[j * m + i] + bias[i];
    }
  }
}

static void refGemmRelu(float *C, const float *A, const float *B,
                        const float *bias, int m, int n, int k, float alpha,
                        float beta, bool transA, bool transB) {
  refGemm(C, A, B, bias, m, n, k, alpha, beta, transA, transB);
  for (int i = 0; i < m * n; ++i)
    C[i] = C[i] > 0.f ? C[i] : 0.f;
}

static void refGemmGelu(float *C, const float *A, const float *B,
                        const float *bias, int m, int n, int k, float alpha,
                        float beta, bool transA, bool transB) {
  refGemm(C, A, B, bias, m, n, k, alpha, beta, transA, transB);
  constexpr float kInvSqrt2 = 0.7071067811865476f;
  for (int i = 0; i < m * n; ++i)
    C[i] *= 0.5f * (1.f + std::erff(C[i] * kInvSqrt2));
}

static int gFailures = 0;

static void checkClose(const float *got, const float *expected, int n,
                       const std::string &name, float rtol = 1e-4f,
                       float atol = 1e-4f) {
  bool pass = true;
  for (int i = 0; i < n; ++i) {
    float diff = std::abs(got[i] - expected[i]);
    float thr = atol + rtol * std::abs(expected[i]);
    if (diff > thr) {
      std::cerr << "  FAIL [" << name << "] idx=" << i << " got=" << got[i]
                << " expected=" << expected[i] << " diff=" << diff << "\n";
      pass = false;
    }
  }
  if (pass)
    std::cout << "  PASS  " << name << "\n";
  else
    ++gFailures;
}

static void fillSeq(float *M, int n, float start = 1.f, float step = 1.f) {
  for (int i = 0; i < n; ++i)
    M[i] = start + static_cast<float>(i) * step;
}

static void fillVal(float *M, int n, float v) {
  for (int i = 0; i < n; ++i)
    M[i] = v;
}

// ---------------------------------------------------------------------------
// CPU tests
// ---------------------------------------------------------------------------

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED

static void runCpuTests() {
  std::cout << "\n=== CPU Tests ===\n";

  alpaka::PlatformCpu platform{};
  auto dev = alpaka::getDevByIdx(platform, 0u);
  alpaka::Queue<alpaka::DevCpu, alpaka::Blocking> queue{dev};
  sofieBLAS<alpaka::TagCpuSerial> blas(queue);

  constexpr int M = 4, N = 3, K = 5;

  // Allocate host buffers
  auto hA = alpaka::allocBuf<float, Idx>(dev, static_cast<Idx>(M * K));
  auto hB = alpaka::allocBuf<float, Idx>(dev, static_cast<Idx>(K * N));
  auto hC = alpaka::allocBuf<float, Idx>(dev, static_cast<Idx>(M * N));
  auto hBias = alpaka::allocBuf<float, Idx>(dev, static_cast<Idx>(M * N));

  float *A = alpaka::getPtrNative(hA);
  float *B = alpaka::getPtrNative(hB);
  float *C = alpaka::getPtrNative(hC);
  float *bias = alpaka::getPtrNative(hBias);

  fillSeq(A, M * K);
  fillSeq(B, K * N, 1.f, 0.5f);
  fillSeq(bias, M * N, 0.1f, 0.1f);

  std::vector<float> ref(M * N);

  // --- matmul NN ---
  fillVal(C, M * N, 0.f);
  blas.matmul('N', 'N', M, N, K, 1.f, hA, hB, 0.f, hC);
  std::copy(C, C + M * N, ref.data());
  std::fill(ref.begin(), ref.end(), 0.f);
  refMatmul(ref.data(), A, B, M, N, K, 1.f, 0.f, false, false);
  checkClose(C, ref.data(), M * N, "cpu::matmul NN");

  // --- matmul TN  (A^T: K×M physical → M×K logical) ---
  {
    auto hAt = alpaka::allocBuf<float, Idx>(dev, static_cast<Idx>(K * M));
    float *At = alpaka::getPtrNative(hAt);
    fillSeq(At, K * M);
    fillVal(C, M * N, 0.f);
    blas.matmul('T', 'N', M, N, K, 1.f, hAt, hB, 0.f, hC);
    std::fill(ref.begin(), ref.end(), 0.f);
    refMatmul(ref.data(), At, B, M, N, K, 1.f, 0.f, true, false);
    checkClose(C, ref.data(), M * N, "cpu::matmul TN");
  }

  // --- matmul NT ---
  {
    auto hBt = alpaka::allocBuf<float, Idx>(dev, static_cast<Idx>(N * K));
    float *Bt = alpaka::getPtrNative(hBt);
    fillSeq(Bt, N * K, 1.f, 0.5f);
    fillVal(C, M * N, 0.f);
    blas.matmul('N', 'T', M, N, K, 1.f, hA, hBt, 0.f, hC);
    std::fill(ref.begin(), ref.end(), 0.f);
    refMatmul(ref.data(), A, Bt, M, N, K, 1.f, 0.f, false, true);
    checkClose(C, ref.data(), M * N, "cpu::matmul NT");
  }

  // --- matmul TT ---
  {
    auto hAt = alpaka::allocBuf<float, Idx>(dev, static_cast<Idx>(K * M));
    auto hBt = alpaka::allocBuf<float, Idx>(dev, static_cast<Idx>(N * K));
    float *At = alpaka::getPtrNative(hAt);
    float *Bt = alpaka::getPtrNative(hBt);
    fillSeq(At, K * M);
    fillSeq(Bt, N * K, 1.f, 0.5f);
    fillVal(C, M * N, 0.f);
    blas.matmul('T', 'T', M, N, K, 1.f, hAt, hBt, 0.f, hC);
    std::fill(ref.begin(), ref.end(), 0.f);
    refMatmul(ref.data(), At, Bt, M, N, K, 1.f, 0.f, true, true);
    checkClose(C, ref.data(), M * N, "cpu::matmul TT");
  }

  // --- matmul: alpha scaling ---
  fillVal(C, M * N, 0.f);
  blas.matmul('N', 'N', M, N, K, 2.5f, hA, hB, 0.f, hC);
  std::fill(ref.begin(), ref.end(), 0.f);
  refMatmul(ref.data(), A, B, M, N, K, 2.5f, 0.f, false, false);
  checkClose(C, ref.data(), M * N, "cpu::matmul alpha=2.5");

  // --- matmul: beta accumulation ---
  fillSeq(C, M * N, 10.f); // pre-fill C
  blas.matmul('N', 'N', M, N, K, 1.f, hA, hB, 0.5f, hC);
  {
    std::vector<float> C0(M * N);
    fillSeq(C0.data(), M * N, 10.f);
    refMatmul(ref.data(), A, B, M, N, K, 1.f, 0.5f, false, false);
    std::copy(C0.begin(), C0.end(), ref.data());
    refMatmul(ref.data(), A, B, M, N, K, 1.f, 0.5f, false, false);
  }
  checkClose(C, ref.data(), M * N, "cpu::matmul beta=0.5");

  // --- gemm NN (beta=0, no prior accumulation) ---
  fillVal(C, M * N, 0.f);
  fillSeq(bias, M * N, 0.1f, 0.1f);
  blas.gemm('N', 'N', M, N, K, 1.f, hA, hB, 0.f, hBias, hC);
  std::fill(ref.begin(), ref.end(), 0.f);
  refGemm(ref.data(), A, B, bias, M, N, K, 1.f, 0.f, false, false);
  checkClose(C, ref.data(), M * N, "cpu::gemm NN beta=0");

  // --- gemm NN (beta=1 accumulation) ---
  fillVal(C, M * N, 0.f);
  blas.gemm('N', 'N', M, N, K, 1.f, hA, hB, 1.f, hBias, hC);
  std::fill(ref.begin(), ref.end(), 0.f);
  refGemm(ref.data(), A, B, bias, M, N, K, 1.f, 1.f, false, false);
  checkClose(C, ref.data(), M * N, "cpu::gemm NN beta=1");

  // --- gemm TN ---
  {
    auto hAt = alpaka::allocBuf<float, Idx>(dev, static_cast<Idx>(K * M));
    float *At = alpaka::getPtrNative(hAt);
    fillSeq(At, K * M);
    fillVal(C, M * N, 0.f);
    blas.gemm('T', 'N', M, N, K, 1.f, hAt, hB, 0.f, hBias, hC);
    std::fill(ref.begin(), ref.end(), 0.f);
    refGemm(ref.data(), At, B, bias, M, N, K, 1.f, 0.f, true, false);
    checkClose(C, ref.data(), M * N, "cpu::gemm TN");
  }

  // --- gemmrelu: all-positive matmul result stays unchanged ---
  {
    // A and B with positive values ensure result is positive before bias
    auto hAp = alpaka::allocBuf<float, Idx>(dev, static_cast<Idx>(M * K));
    auto hBp = alpaka::allocBuf<float, Idx>(dev, static_cast<Idx>(K * N));
    auto hBiasp = alpaka::allocBuf<float, Idx>(dev, static_cast<Idx>(M * N));
    float *Ap = alpaka::getPtrNative(hAp);
    float *Bp = alpaka::getPtrNative(hBp);
    float *biasp = alpaka::getPtrNative(hBiasp);
    fillSeq(Ap, M * K, 0.1f, 0.1f);
    fillSeq(Bp, K * N, 0.1f, 0.1f);
    fillVal(biasp, M * N, 0.f);
    fillVal(C, M * N, 0.f);
    blas.gemmrelu('N', 'N', M, N, K, 1.f, hAp, hBp, 0.f, hBiasp, hC);
    std::fill(ref.begin(), ref.end(), 0.f);
    refGemmRelu(ref.data(), Ap, Bp, biasp, M, N, K, 1.f, 0.f, false, false);
    checkClose(C, ref.data(), M * N, "cpu::gemmrelu all-positive");
  }

  // --- gemmrelu: negative values clamped to zero ---
  {
    // Use alpha=-1 to force negative results
    auto hBiasz = alpaka::allocBuf<float, Idx>(dev, static_cast<Idx>(M * N));
    fillVal(alpaka::getPtrNative(hBiasz), M * N, 0.f);
    fillVal(C, M * N, 0.f);
    blas.gemmrelu('N', 'N', M, N, K, -1.f, hA, hB, 0.f, hBiasz, hC);
    std::fill(ref.begin(), ref.end(), 0.f);
    refGemmRelu(ref.data(), A, B, alpaka::getPtrNative(hBiasz), M, N, K, -1.f,
                0.f, false, false);
    checkClose(C, ref.data(), M * N, "cpu::gemmrelu alpha=-1 (clamped)");
  }

  // --- gemmrelu with bias ---
  fillVal(C, M * N, 0.f);
  fillSeq(bias, M * N, -5.f, 2.f);
  blas.gemmrelu('N', 'N', M, N, K, 1.f, hA, hB, 0.f, hBias, hC);
  std::fill(ref.begin(), ref.end(), 0.f);
  refGemmRelu(ref.data(), A, B, bias, M, N, K, 1.f, 0.f, false, false);
  checkClose(C, ref.data(), M * N, "cpu::gemmrelu with mixed bias");

  // --- gemmgelu NN ---
  fillVal(C, M * N, 0.f);
  fillVal(bias, M * N, 0.f);
  blas.gemmgelu('N', 'N', M, N, K, 1.f, hA, hB, 0.f, hBias, hC);
  std::fill(ref.begin(), ref.end(), 0.f);
  refGemmGelu(ref.data(), A, B, bias, M, N, K, 1.f, 0.f, false, false);
  checkClose(C, ref.data(), M * N, "cpu::gemmgelu NN");

  // --- gemmgelu with bias ---
  fillVal(C, M * N, 0.f);
  fillSeq(bias, M * N, -2.f, 0.5f);
  blas.gemmgelu('N', 'N', M, N, K, 1.f, hA, hB, 0.f, hBias, hC);
  std::fill(ref.begin(), ref.end(), 0.f);
  refGemmGelu(ref.data(), A, B, bias, M, N, K, 1.f, 0.f, false, false);
  checkClose(C, ref.data(), M * N, "cpu::gemmgelu with bias");

  // --- gemmgelu TN ---
  {
    auto hAt = alpaka::allocBuf<float, Idx>(dev, static_cast<Idx>(K * M));
    float *At = alpaka::getPtrNative(hAt);
    fillSeq(At, K * M);
    fillVal(C, M * N, 0.f);
    fillVal(bias, M * N, 0.f);
    blas.gemmgelu('T', 'N', M, N, K, 1.f, hAt, hB, 0.f, hBias, hC);
    std::fill(ref.begin(), ref.end(), 0.f);
    refGemmGelu(ref.data(), At, B, bias, M, N, K, 1.f, 0.f, true, false);
    checkClose(C, ref.data(), M * N, "cpu::gemmgelu TN");
  }

  // --- edge: zero matrix ---
  {
    auto hZ = alpaka::allocBuf<float, Idx>(dev, static_cast<Idx>(M * K));
    fillVal(alpaka::getPtrNative(hZ), M * K, 0.f);
    fillVal(C, M * N, 99.f);
    fillVal(bias, M * N, 0.f);
    blas.matmul('N', 'N', M, N, K, 1.f, hZ, hB, 0.f, hC);
    std::fill(ref.begin(), ref.end(), 0.f);
    checkClose(C, ref.data(), M * N, "cpu::matmul zero-A");
  }

  // --- edge: identity-like (square, known result) ---
  {
    constexpr int S = 3;
    auto hI = alpaka::allocBuf<float, Idx>(dev, static_cast<Idx>(S * S));
    auto hX = alpaka::allocBuf<float, Idx>(dev, static_cast<Idx>(S * S));
    auto hY = alpaka::allocBuf<float, Idx>(dev, static_cast<Idx>(S * S));
    float *I = alpaka::getPtrNative(hI);
    float *X = alpaka::getPtrNative(hX);
    float *Y = alpaka::getPtrNative(hY);
    fillVal(I, S * S, 0.f);
    for (int i = 0; i < S; ++i)
      I[i * S + i] = 1.f;
    fillSeq(X, S * S);
    fillVal(Y, S * S, 0.f);
    blas.matmul('N', 'N', S, S, S, 1.f, hI, hX, 0.f, hY);
    checkClose(Y, X, S * S, "cpu::matmul identity×X=X");
  }
}

#endif // ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
static int ldaFor(char trans, int m, int k) {
  return (trans == 'N' || trans == 'n') ? m : k;
}
static int ldbFor(char trans, int k, int n) {
  return (trans == 'N' || trans == 'n') ? k : n;
}
#endif

// ---------------------------------------------------------------------------
// CUDA tests
// ---------------------------------------------------------------------------

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

static void runCudaTests() {
  std::cout << "\n=== CUDA Tests ===\n";

  alpaka::PlatformCudaRt platform{};
  auto dev = alpaka::getDevByIdx(platform, 0u);
  alpaka::Queue<alpaka::DevCudaRt, alpaka::NonBlocking> queue{dev};
  sofieBLAS<alpaka::TagGpuCudaRt> blas(queue);

  alpaka::PlatformCpu hostPlatform{};
  auto hostDev = alpaka::getDevByIdx(hostPlatform, 0u);

  constexpr int M = 4, N = 3, K = 5;

  auto hA = alpaka::allocBuf<float, Idx>(hostDev, static_cast<Idx>(M * K));
  auto hB = alpaka::allocBuf<float, Idx>(hostDev, static_cast<Idx>(K * N));
  auto hC = alpaka::allocBuf<float, Idx>(hostDev, static_cast<Idx>(M * N));
  auto hBias = alpaka::allocBuf<float, Idx>(hostDev, static_cast<Idx>(M * N));

  float *A = alpaka::getPtrNative(hA);
  float *B = alpaka::getPtrNative(hB);
  float *bias = alpaka::getPtrNative(hBias);

  fillSeq(A, M * K);
  fillSeq(B, K * N, 1.f, 0.5f);
  fillVal(bias, M * N, 0.f);

  auto dA = alpaka::allocAsyncBuf<float, Idx>(queue, static_cast<Idx>(M * K));
  auto dB = alpaka::allocAsyncBuf<float, Idx>(queue, static_cast<Idx>(K * N));
  auto dC = alpaka::allocAsyncBuf<float, Idx>(queue, static_cast<Idx>(M * N));
  auto dBias =
      alpaka::allocAsyncBuf<float, Idx>(queue, static_cast<Idx>(M * N));

  alpaka::memcpy(queue, dA, hA);
  alpaka::memcpy(queue, dB, hB);
  alpaka::memcpy(queue, dBias, hBias);
  alpaka::wait(queue);

  std::vector<float> ref(M * N);
  float *C = alpaka::getPtrNative(hC);

  auto verify = [&](const std::string &name) {
    alpaka::memcpy(queue, hC, dC);
    alpaka::wait(queue);
    checkClose(C, ref.data(), M * N, name);
  };

  // ---- matmul NN ----
  blas.addOperationConfig(M, N, K, ldaFor('N', M, K), ldbFor('N', K, N), M, 'N',
                          'N', Epilogue::Default);
  std::fill(ref.begin(), ref.end(), 0.f);
  refMatmul(ref.data(), A, B, M, N, K, 1.f, 0.f, false, false);
  blas.matmul('N', 'N', M, N, K, 1.f, dA, dB, 0.f, dC);
  verify("cuda::matmul NN");

  // ---- matmul TN ----
  {
    auto hAt = alpaka::allocBuf<float, Idx>(hostDev, static_cast<Idx>(K * M));
    float *At = alpaka::getPtrNative(hAt);
    fillSeq(At, K * M);
    auto dAt =
        alpaka::allocAsyncBuf<float, Idx>(queue, static_cast<Idx>(K * M));
    alpaka::memcpy(queue, dAt, hAt);
    alpaka::wait(queue);
    blas.addOperationConfig(M, N, K, ldaFor('T', M, K), ldbFor('N', K, N), M,
                            'T', 'N', Epilogue::Default);
    std::fill(ref.begin(), ref.end(), 0.f);
    refMatmul(ref.data(), At, B, M, N, K, 1.f, 0.f, true, false);
    blas.matmul('T', 'N', M, N, K, 1.f, dAt, dB, 0.f, dC);
    verify("cuda::matmul TN");
  }

  // ---- matmul NT ----
  {
    auto hBt = alpaka::allocBuf<float, Idx>(hostDev, static_cast<Idx>(N * K));
    float *Bt = alpaka::getPtrNative(hBt);
    fillSeq(Bt, N * K, 1.f, 0.5f);
    auto dBt =
        alpaka::allocAsyncBuf<float, Idx>(queue, static_cast<Idx>(N * K));
    alpaka::memcpy(queue, dBt, hBt);
    alpaka::wait(queue);
    blas.addOperationConfig(M, N, K, ldaFor('N', M, K), ldbFor('T', K, N), M,
                            'N', 'T', Epilogue::Default);
    std::fill(ref.begin(), ref.end(), 0.f);
    refMatmul(ref.data(), A, Bt, M, N, K, 1.f, 0.f, false, true);
    blas.matmul('N', 'T', M, N, K, 1.f, dA, dBt, 0.f, dC);
    verify("cuda::matmul NT");
  }

  // ---- matmul alpha=2.5 ----
  std::fill(ref.begin(), ref.end(), 0.f);
  refMatmul(ref.data(), A, B, M, N, K, 2.5f, 0.f, false, false);
  blas.matmul('N', 'N', M, N, K, 2.5f, dA, dB, 0.f, dC);
  verify("cuda::matmul alpha=2.5");

  // ---- gemm NN beta=0 ----
  fillSeq(bias, M * N, 0.1f, 0.1f);
  alpaka::memcpy(queue, dBias, hBias);
  alpaka::wait(queue);
  std::fill(ref.begin(), ref.end(), 0.f);
  refGemm(ref.data(), A, B, bias, M, N, K, 1.f, 0.f, false, false);
  blas.gemm('N', 'N', M, N, K, 1.f, dA, dB, 0.f, dBias, dC);
  verify("cuda::gemm NN beta=0");

  // ---- gemm NN beta=1 ----
  // D_in = bias, so result = A*B + 1*bias_matrix + bias_vec
  std::fill(ref.begin(), ref.end(), 0.f);
  refGemm(ref.data(), A, B, bias, M, N, K, 1.f, 1.f, false, false);
  blas.gemm('N', 'N', M, N, K, 1.f, dA, dB, 1.f, dBias, dC);
  verify("cuda::gemm NN beta=1");

  // ---- gemm TN ----
  {
    auto hAt = alpaka::allocBuf<float, Idx>(hostDev, static_cast<Idx>(K * M));
    float *At = alpaka::getPtrNative(hAt);
    fillSeq(At, K * M);
    auto dAt =
        alpaka::allocAsyncBuf<float, Idx>(queue, static_cast<Idx>(K * M));
    alpaka::memcpy(queue, dAt, hAt);
    alpaka::wait(queue);
    blas.addOperationConfig(M, N, K, ldaFor('T', M, K), ldbFor('N', K, N), M,
                            'T', 'N', Epilogue::Bias);
    std::fill(ref.begin(), ref.end(), 0.f);
    refGemm(ref.data(), At, B, bias, M, N, K, 1.f, 0.f, true, false);
    blas.gemm('T', 'N', M, N, K, 1.f, dAt, dB, 0.f, dBias, dC);
    verify("cuda::gemm TN");
  }

  // ---- gemmrelu: all-positive (relu is identity) ----
  {
    auto hAp = alpaka::allocBuf<float, Idx>(hostDev, static_cast<Idx>(M * K));
    auto hBp = alpaka::allocBuf<float, Idx>(hostDev, static_cast<Idx>(K * N));
    auto hBiasz =
        alpaka::allocBuf<float, Idx>(hostDev, static_cast<Idx>(M * N));
    float *Ap = alpaka::getPtrNative(hAp);
    float *Bp = alpaka::getPtrNative(hBp);
    fillSeq(Ap, M * K, 0.1f, 0.1f);
    fillSeq(Bp, K * N, 0.1f, 0.1f);
    fillVal(alpaka::getPtrNative(hBiasz), M * N, 0.f);
    auto dAp =
        alpaka::allocAsyncBuf<float, Idx>(queue, static_cast<Idx>(M * K));
    auto dBp =
        alpaka::allocAsyncBuf<float, Idx>(queue, static_cast<Idx>(K * N));
    auto dBiasz =
        alpaka::allocAsyncBuf<float, Idx>(queue, static_cast<Idx>(M * N));
    alpaka::memcpy(queue, dAp, hAp);
    alpaka::memcpy(queue, dBp, hBp);
    alpaka::memcpy(queue, dBiasz, hBiasz);
    alpaka::wait(queue);
    blas.addOperationConfig(M, N, K, M, K, M, 'N', 'N', Epilogue::ReluBias);
    std::fill(ref.begin(), ref.end(), 0.f);
    refGemmRelu(ref.data(), Ap, Bp, alpaka::getPtrNative(hBiasz), M, N, K, 1.f,
                0.f, false, false);
    blas.gemmrelu('N', 'N', M, N, K, 1.f, dAp, dBp, 0.f, dBiasz, dC);
    verify("cuda::gemmrelu all-positive");
  }

  // ---- gemmrelu: alpha=-1 forces negatives -> clamped to zero ----
  {
    auto hBiasz =
        alpaka::allocBuf<float, Idx>(hostDev, static_cast<Idx>(M * N));
    fillVal(alpaka::getPtrNative(hBiasz), M * N, 0.f);
    auto dBiasz =
        alpaka::allocAsyncBuf<float, Idx>(queue, static_cast<Idx>(M * N));
    alpaka::memcpy(queue, dBiasz, hBiasz);
    alpaka::wait(queue);
    std::fill(ref.begin(), ref.end(), 0.f);
    refGemmRelu(ref.data(), A, B, alpaka::getPtrNative(hBiasz), M, N, K, -1.f,
                0.f, false, false);
    blas.gemmrelu('N', 'N', M, N, K, -1.f, dA, dB, 0.f, dBiasz, dC);
    verify("cuda::gemmrelu alpha=-1 (clamped)");
  }

  // ---- gemmrelu with mixed bias ----
  fillSeq(bias, M * N, -5.f, 2.f);
  alpaka::memcpy(queue, dBias, hBias);
  alpaka::wait(queue);
  std::fill(ref.begin(), ref.end(), 0.f);
  refGemmRelu(ref.data(), A, B, bias, M, N, K, 1.f, 0.f, false, false);
  blas.gemmrelu('N', 'N', M, N, K, 1.f, dA, dB, 0.f, dBias, dC);
  verify("cuda::gemmrelu with mixed bias");

  // ---- gemmgelu NN ----
  fillVal(bias, M * N, 0.f);
  alpaka::memcpy(queue, dBias, hBias);
  alpaka::wait(queue);
  std::fill(ref.begin(), ref.end(), 0.f);
  refGemmGelu(ref.data(), A, B, bias, M, N, K, 1.f, 0.f, false, false);
  blas.gemmgelu('N', 'N', M, N, K, 1.f, dA, dB, 0.f, dBias, dC);
  verify("cuda::gemmgelu NN");

  // ---- gemmgelu with bias ----
  fillSeq(bias, M * N, -2.f, 0.5f);
  alpaka::memcpy(queue, dBias, hBias);
  alpaka::wait(queue);
  std::fill(ref.begin(), ref.end(), 0.f);
  refGemmGelu(ref.data(), A, B, bias, M, N, K, 1.f, 0.f, false, false);
  blas.gemmgelu('N', 'N', M, N, K, 1.f, dA, dB, 0.f, dBias, dC);
  verify("cuda::gemmgelu with bias");

  // ---- edge: zero A ----
  {
    auto hZero = alpaka::allocBuf<float, Idx>(hostDev, static_cast<Idx>(M * K));
    fillVal(alpaka::getPtrNative(hZero), M * K, 0.f);
    auto dZero =
        alpaka::allocAsyncBuf<float, Idx>(queue, static_cast<Idx>(M * K));
    alpaka::memcpy(queue, dZero, hZero);
    alpaka::wait(queue);
    std::fill(ref.begin(), ref.end(), 0.f);
    blas.matmul('N', 'N', M, N, K, 1.f, dZero, dB, 0.f, dC);
    verify("cuda::matmul zero-A");
  }
}

static void runDynamicShapeTests() {
  std::cout << "\n=== CUDA Dynamic-Shape Tests ===\n";

  alpaka::PlatformCudaRt platform{};
  auto dev = alpaka::getDevByIdx(platform, 0u);
  alpaka::Queue<alpaka::DevCudaRt, alpaka::NonBlocking> queue{dev};

  alpaka::PlatformCpu hostPlatform{};
  auto hostDev = alpaka::getDevByIdx(hostPlatform, 0u);

  // M0 is the construction-time size given to addOperationConfig; the buffers
  // hold MCAP rows so sizes above M0 are exercised too.
  constexpr int MCAP = 96, M0 = 64, N = 3, K = 5;

  auto hA = alpaka::allocBuf<float, Idx>(hostDev, static_cast<Idx>(MCAP * K));
  auto hB = alpaka::allocBuf<float, Idx>(hostDev, static_cast<Idx>(K * N));
  auto hC = alpaka::allocBuf<float, Idx>(hostDev, static_cast<Idx>(MCAP * N));
  float *A = alpaka::getPtrNative(hA);
  float *B = alpaka::getPtrNative(hB);
  float *C = alpaka::getPtrNative(hC);
  fillSeq(A, MCAP * K, 0.5f, 0.25f);
  fillSeq(B, K * N, 1.f, 0.5f);

  auto dA =
      alpaka::allocAsyncBuf<float, Idx>(queue, static_cast<Idx>(MCAP * K));
  auto dB = alpaka::allocAsyncBuf<float, Idx>(queue, static_cast<Idx>(K * N));
  auto dC =
      alpaka::allocAsyncBuf<float, Idx>(queue, static_cast<Idx>(MCAP * N));
  alpaka::memcpy(queue, dA, hA);
  alpaka::memcpy(queue, dB, hB);
  alpaka::wait(queue);

  // One instance serving sizes never passed to addOperationConfig (issue #10),
  // including m=1 and a size above the construction-time one.
  sofieBLAS<alpaka::TagGpuCudaRt> blas(queue);
  blas.addOperationConfig(M0, N, K, ldaFor('N', M0, K), ldbFor('N', K, N), M0,
                          'N', 'N', Epilogue::Default);

  std::vector<float> ref;
  auto runAt = [&](int m, const std::string &name) {
    ref.assign(static_cast<std::size_t>(m) * N, 0.f);
    refMatmul(ref.data(), A, B, m, N, K, 1.f, 0.f, false, false);
    blas.matmul('N', 'N', static_cast<unsigned>(m), static_cast<unsigned>(N),
                static_cast<unsigned>(K), 1.f, dA, dB, 0.f, dC);
    alpaka::memcpy(queue, hC, dC);
    alpaka::wait(queue);
    checkClose(C, ref.data(), m * N, name);
  };

  for (int m : {M0, 37, 8, 51, 1, M0, MCAP})
    runAt(m, "cuda::dynamic m=" + std::to_string(m));

  // Generated code calls the raw-pointer overloads; one call keeps them
  // compiled and resolving to the right overload.
  ref.assign(static_cast<std::size_t>(45) * N, 0.f);
  refMatmul(ref.data(), A, B, 45, N, K, 1.f, 0.f, false, false);
  blas.matmul('N', 'N', 45u, static_cast<unsigned>(N), static_cast<unsigned>(K),
              1.f, alpaka::getPtrNative(dA), alpaka::getPtrNative(dB), 0.f,
              alpaka::getPtrNative(dC));
  alpaka::memcpy(queue, hC, dC);
  alpaka::wait(queue);
  checkClose(C, ref.data(), 45 * N, "cuda::dynamic raw pointers m=45");

  // 32 distinct sizes through a cache limited to 8 entries.
  {
    sofieBLAS<alpaka::TagGpuCudaRt> capped(queue, 8);
    capped.addOperationConfig(M0, N, K, ldaFor('N', M0, K), ldbFor('N', K, N),
                              M0, 'N', 'N', Epilogue::Default);
    float worst = 0.f;
    for (int m = M0 + 1; m <= MCAP; ++m) {
      ref.assign(static_cast<std::size_t>(m) * N, 0.f);
      refMatmul(ref.data(), A, B, m, N, K, 1.f, 0.f, false, false);
      capped.matmul('N', 'N', static_cast<unsigned>(m),
                    static_cast<unsigned>(N), static_cast<unsigned>(K), 1.f, dA,
                    dB, 0.f, dC);
      alpaka::memcpy(queue, hC, dC);
      alpaka::wait(queue);
      for (std::size_t i = 0; i < ref.size(); ++i)
        worst = std::max(worst, std::abs(C[i] - ref[i]));
    }
    if (capped.algoCacheSize() <= 8 && worst < 1e-3f) {
      std::cout << "  PASS  cuda::cache limit honoured\n";
    } else {
      std::cerr << "  FAIL [cuda::cache limit honoured] "
                << capped.algoCacheSize() << " entries, worst err " << worst
                << "\n";
      ++gFailures;
    }
  }
}

#endif // ALPAKA_ACC_GPU_CUDA_ENABLED

// ---------------------------------------------------------------------------
// HIP tests
// ---------------------------------------------------------------------------

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

static void runHipTests() {
  std::cout << "\n=== HIP Tests ===\n";

  alpaka::PlatformHipRt platform{};
  auto dev = alpaka::getDevByIdx(platform, 0u);
  alpaka::Queue<alpaka::DevHipRt, alpaka::NonBlocking> queue{dev};
  sofieBLAS<alpaka::TagGpuHipRt> blas(queue);

  alpaka::PlatformCpu hostPlatform{};
  auto hostDev = alpaka::getDevByIdx(hostPlatform, 0u);

  constexpr int M = 4, N = 3, K = 5;

  auto hA = alpaka::allocBuf<float, Idx>(hostDev, static_cast<Idx>(M * K));
  auto hB = alpaka::allocBuf<float, Idx>(hostDev, static_cast<Idx>(K * N));
  auto hC = alpaka::allocBuf<float, Idx>(hostDev, static_cast<Idx>(M * N));
  auto hBias = alpaka::allocBuf<float, Idx>(hostDev, static_cast<Idx>(M * N));

  float *A = alpaka::getPtrNative(hA);
  float *B = alpaka::getPtrNative(hB);
  float *bias = alpaka::getPtrNative(hBias);

  fillSeq(A, M * K);
  fillSeq(B, K * N, 1.f, 0.5f);
  fillVal(bias, M * N, 0.f);

  auto dA = alpaka::allocAsyncBuf<float, Idx>(queue, static_cast<Idx>(M * K));
  auto dB = alpaka::allocAsyncBuf<float, Idx>(queue, static_cast<Idx>(K * N));
  auto dC = alpaka::allocAsyncBuf<float, Idx>(queue, static_cast<Idx>(M * N));
  auto dBias =
      alpaka::allocAsyncBuf<float, Idx>(queue, static_cast<Idx>(M * N));

  alpaka::memcpy(queue, dA, hA);
  alpaka::memcpy(queue, dB, hB);
  alpaka::memcpy(queue, dBias, hBias);
  alpaka::wait(queue);

  std::vector<float> ref(M * N);
  float *C = alpaka::getPtrNative(hC);

  auto verify = [&](const std::string &name) {
    alpaka::memcpy(queue, hC, dC);
    alpaka::wait(queue);
    checkClose(C, ref.data(), M * N, name);
  };

  // ---- matmul NN ----
  blas.addOperationConfig(M, N, K, ldaFor('N', M, K), ldbFor('N', K, N), M, 'N',
                          'N', Epilogue::Default);
  std::fill(ref.begin(), ref.end(), 0.f);
  refMatmul(ref.data(), A, B, M, N, K, 1.f, 0.f, false, false);
  blas.matmul('N', 'N', M, N, K, 1.f, dA, dB, 0.f, dC);
  verify("hip::matmul NN");

  // ---- matmul TN ----
  {
    auto hAt = alpaka::allocBuf<float, Idx>(hostDev, static_cast<Idx>(K * M));
    float *At = alpaka::getPtrNative(hAt);
    fillSeq(At, K * M);
    auto dAt =
        alpaka::allocAsyncBuf<float, Idx>(queue, static_cast<Idx>(K * M));
    alpaka::memcpy(queue, dAt, hAt);
    alpaka::wait(queue);
    blas.addOperationConfig(M, N, K, ldaFor('T', M, K), ldbFor('N', K, N), M,
                            'T', 'N', Epilogue::Default);
    std::fill(ref.begin(), ref.end(), 0.f);
    refMatmul(ref.data(), At, B, M, N, K, 1.f, 0.f, true, false);
    blas.matmul('T', 'N', M, N, K, 1.f, dAt, dB, 0.f, dC);
    verify("hip::matmul TN");
  }

  // ---- matmul NT ----
  {
    auto hBt = alpaka::allocBuf<float, Idx>(hostDev, static_cast<Idx>(N * K));
    float *Bt = alpaka::getPtrNative(hBt);
    fillSeq(Bt, N * K, 1.f, 0.5f);
    auto dBt =
        alpaka::allocAsyncBuf<float, Idx>(queue, static_cast<Idx>(N * K));
    alpaka::memcpy(queue, dBt, hBt);
    alpaka::wait(queue);
    blas.addOperationConfig(M, N, K, ldaFor('N', M, K), ldbFor('T', K, N), M,
                            'N', 'T', Epilogue::Default);
    std::fill(ref.begin(), ref.end(), 0.f);
    refMatmul(ref.data(), A, Bt, M, N, K, 1.f, 0.f, false, true);
    blas.matmul('N', 'T', M, N, K, 1.f, dA, dBt, 0.f, dC);
    verify("hip::matmul NT");
  }

  // ---- matmul alpha=2.5 ----
  std::fill(ref.begin(), ref.end(), 0.f);
  refMatmul(ref.data(), A, B, M, N, K, 2.5f, 0.f, false, false);
  blas.matmul('N', 'N', M, N, K, 2.5f, dA, dB, 0.f, dC);
  verify("hip::matmul alpha=2.5");

  // ---- gemm NN beta=0 ----
  fillSeq(bias, M * N, 0.1f, 0.1f);
  alpaka::memcpy(queue, dBias, hBias);
  alpaka::wait(queue);
  std::fill(ref.begin(), ref.end(), 0.f);
  refGemm(ref.data(), A, B, bias, M, N, K, 1.f, 0.f, false, false);
  blas.gemm('N', 'N', M, N, K, 1.f, dA, dB, 0.f, dBias, dC);
  verify("hip::gemm NN beta=0");

  // ---- gemm NN beta=1 ----
  // D_in = bias, so result = A*B + 1*bias_matrix + bias_vec
  std::fill(ref.begin(), ref.end(), 0.f);
  refGemm(ref.data(), A, B, bias, M, N, K, 1.f, 1.f, false, false);
  blas.gemm('N', 'N', M, N, K, 1.f, dA, dB, 1.f, dBias, dC);
  verify("hip::gemm NN beta=1");

  // ---- gemm TN ----
  {
    auto hAt = alpaka::allocBuf<float, Idx>(hostDev, static_cast<Idx>(K * M));
    float *At = alpaka::getPtrNative(hAt);
    fillSeq(At, K * M);
    auto dAt =
        alpaka::allocAsyncBuf<float, Idx>(queue, static_cast<Idx>(K * M));
    alpaka::memcpy(queue, dAt, hAt);
    alpaka::wait(queue);
    blas.addOperationConfig(M, N, K, ldaFor('T', M, K), ldbFor('N', K, N), M,
                            'T', 'N', Epilogue::Bias);
    std::fill(ref.begin(), ref.end(), 0.f);
    refGemm(ref.data(), At, B, bias, M, N, K, 1.f, 0.f, true, false);
    blas.gemm('T', 'N', M, N, K, 1.f, dAt, dB, 0.f, dBias, dC);
    verify("hip::gemm TN");
  }

  // ---- gemmrelu: all-positive ----
  {
    auto hAp = alpaka::allocBuf<float, Idx>(hostDev, static_cast<Idx>(M * K));
    auto hBp = alpaka::allocBuf<float, Idx>(hostDev, static_cast<Idx>(K * N));
    auto hBiasz =
        alpaka::allocBuf<float, Idx>(hostDev, static_cast<Idx>(M * N));
    float *Ap = alpaka::getPtrNative(hAp);
    float *Bp = alpaka::getPtrNative(hBp);
    fillSeq(Ap, M * K, 0.1f, 0.1f);
    fillSeq(Bp, K * N, 0.1f, 0.1f);
    fillVal(alpaka::getPtrNative(hBiasz), M * N, 0.f);
    auto dAp =
        alpaka::allocAsyncBuf<float, Idx>(queue, static_cast<Idx>(M * K));
    auto dBp =
        alpaka::allocAsyncBuf<float, Idx>(queue, static_cast<Idx>(K * N));
    auto dBiasz =
        alpaka::allocAsyncBuf<float, Idx>(queue, static_cast<Idx>(M * N));
    alpaka::memcpy(queue, dAp, hAp);
    alpaka::memcpy(queue, dBp, hBp);
    alpaka::memcpy(queue, dBiasz, hBiasz);
    alpaka::wait(queue);
    blas.addOperationConfig(M, N, K, M, K, M, 'N', 'N', Epilogue::ReluBias);
    std::fill(ref.begin(), ref.end(), 0.f);
    refGemmRelu(ref.data(), Ap, Bp, alpaka::getPtrNative(hBiasz), M, N, K, 1.f,
                0.f, false, false);
    blas.gemmrelu('N', 'N', M, N, K, 1.f, dAp, dBp, 0.f, dBiasz, dC);
    verify("hip::gemmrelu all-positive");
  }

  // ---- gemmrelu: alpha=-1 forces negatives ----
  {
    auto hBiasz =
        alpaka::allocBuf<float, Idx>(hostDev, static_cast<Idx>(M * N));
    fillVal(alpaka::getPtrNative(hBiasz), M * N, 0.f);
    auto dBiasz =
        alpaka::allocAsyncBuf<float, Idx>(queue, static_cast<Idx>(M * N));
    alpaka::memcpy(queue, dBiasz, hBiasz);
    alpaka::wait(queue);
    std::fill(ref.begin(), ref.end(), 0.f);
    refGemmRelu(ref.data(), A, B, alpaka::getPtrNative(hBiasz), M, N, K, -1.f,
                0.f, false, false);
    blas.gemmrelu('N', 'N', M, N, K, -1.f, dA, dB, 0.f, dBiasz, dC);
    verify("hip::gemmrelu alpha=-1 (clamped)");
  }

  // ---- gemmrelu with mixed bias ----
  fillSeq(bias, M * N, -5.f, 2.f);
  alpaka::memcpy(queue, dBias, hBias);
  alpaka::wait(queue);
  std::fill(ref.begin(), ref.end(), 0.f);
  refGemmRelu(ref.data(), A, B, bias, M, N, K, 1.f, 0.f, false, false);
  blas.gemmrelu('N', 'N', M, N, K, 1.f, dA, dB, 0.f, dBias, dC);
  verify("hip::gemmrelu with mixed bias");

  // ---- gemmgelu NN ----
  fillVal(bias, M * N, 0.f);
  alpaka::memcpy(queue, dBias, hBias);
  alpaka::wait(queue);
  std::fill(ref.begin(), ref.end(), 0.f);
  refGemmGelu(ref.data(), A, B, bias, M, N, K, 1.f, 0.f, false, false);
  blas.gemmgelu('N', 'N', M, N, K, 1.f, dA, dB, 0.f, dBias, dC);
  verify("hip::gemmgelu NN");

  // ---- gemmgelu with bias ----
  fillSeq(bias, M * N, -2.f, 0.5f);
  alpaka::memcpy(queue, dBias, hBias);
  alpaka::wait(queue);
  std::fill(ref.begin(), ref.end(), 0.f);
  refGemmGelu(ref.data(), A, B, bias, M, N, K, 1.f, 0.f, false, false);
  blas.gemmgelu('N', 'N', M, N, K, 1.f, dA, dB, 0.f, dBias, dC);
  verify("hip::gemmgelu with bias");

  // ---- edge: zero A ----
  {
    auto hZero = alpaka::allocBuf<float, Idx>(hostDev, static_cast<Idx>(M * K));
    fillVal(alpaka::getPtrNative(hZero), M * K, 0.f);
    auto dZero =
        alpaka::allocAsyncBuf<float, Idx>(queue, static_cast<Idx>(M * K));
    alpaka::memcpy(queue, dZero, hZero);
    alpaka::wait(queue);
    std::fill(ref.begin(), ref.end(), 0.f);
    blas.matmul('N', 'N', M, N, K, 1.f, dZero, dB, 0.f, dC);
    verify("hip::matmul zero-A");
  }
}

static void runHipDynamicShapeTests() {
  std::cout << "\n=== HIP Dynamic-Shape Tests ===\n";

  alpaka::PlatformHipRt platform{};
  auto dev = alpaka::getDevByIdx(platform, 0u);
  alpaka::Queue<alpaka::DevHipRt, alpaka::NonBlocking> queue{dev};

  alpaka::PlatformCpu hostPlatform{};
  auto hostDev = alpaka::getDevByIdx(hostPlatform, 0u);

  // M0 is the construction-time size given to addOperationConfig; the buffers
  // hold MCAP rows so sizes above M0 are exercised too.
  constexpr int MCAP = 96, M0 = 64, N = 3, K = 5;

  auto hA = alpaka::allocBuf<float, Idx>(hostDev, static_cast<Idx>(MCAP * K));
  auto hB = alpaka::allocBuf<float, Idx>(hostDev, static_cast<Idx>(K * N));
  auto hC = alpaka::allocBuf<float, Idx>(hostDev, static_cast<Idx>(MCAP * N));
  float *A = alpaka::getPtrNative(hA);
  float *B = alpaka::getPtrNative(hB);
  float *C = alpaka::getPtrNative(hC);
  fillSeq(A, MCAP * K, 0.5f, 0.25f);
  fillSeq(B, K * N, 1.f, 0.5f);

  auto dA =
      alpaka::allocAsyncBuf<float, Idx>(queue, static_cast<Idx>(MCAP * K));
  auto dB = alpaka::allocAsyncBuf<float, Idx>(queue, static_cast<Idx>(K * N));
  auto dC =
      alpaka::allocAsyncBuf<float, Idx>(queue, static_cast<Idx>(MCAP * N));
  alpaka::memcpy(queue, dA, hA);
  alpaka::memcpy(queue, dB, hB);
  alpaka::wait(queue);

  // One instance serving sizes never passed to addOperationConfig (issue #10),
  // including m=1 and a size above the construction-time one.
  sofieBLAS<alpaka::TagGpuHipRt> blas(queue);
  blas.addOperationConfig(M0, N, K, ldaFor('N', M0, K), ldbFor('N', K, N), M0,
                          'N', 'N', Epilogue::Default);

  std::vector<float> ref;
  auto runAt = [&](int m, const std::string &name) {
    ref.assign(static_cast<std::size_t>(m) * N, 0.f);
    refMatmul(ref.data(), A, B, m, N, K, 1.f, 0.f, false, false);
    blas.matmul('N', 'N', static_cast<unsigned>(m), static_cast<unsigned>(N),
                static_cast<unsigned>(K), 1.f, dA, dB, 0.f, dC);
    alpaka::memcpy(queue, hC, dC);
    alpaka::wait(queue);
    checkClose(C, ref.data(), m * N, name);
  };

  for (int m : {M0, 37, 8, 51, 1, M0, MCAP})
    runAt(m, "hip::dynamic m=" + std::to_string(m));

  // Generated code calls the raw-pointer overloads; one call keeps them
  // compiled and resolving to the right overload.
  ref.assign(static_cast<std::size_t>(45) * N, 0.f);
  refMatmul(ref.data(), A, B, 45, N, K, 1.f, 0.f, false, false);
  blas.matmul('N', 'N', 45u, static_cast<unsigned>(N), static_cast<unsigned>(K),
              1.f, alpaka::getPtrNative(dA), alpaka::getPtrNative(dB), 0.f,
              alpaka::getPtrNative(dC));
  alpaka::memcpy(queue, hC, dC);
  alpaka::wait(queue);
  checkClose(C, ref.data(), 45 * N, "hip::dynamic raw pointers m=45");

  // 32 distinct sizes through a cache limited to 8 entries.
  {
    sofieBLAS<alpaka::TagGpuHipRt> capped(queue, 8);
    capped.addOperationConfig(M0, N, K, ldaFor('N', M0, K), ldbFor('N', K, N),
                              M0, 'N', 'N', Epilogue::Default);
    float worst = 0.f;
    for (int m = M0 + 1; m <= MCAP; ++m) {
      ref.assign(static_cast<std::size_t>(m) * N, 0.f);
      refMatmul(ref.data(), A, B, m, N, K, 1.f, 0.f, false, false);
      capped.matmul('N', 'N', static_cast<unsigned>(m),
                    static_cast<unsigned>(N), static_cast<unsigned>(K), 1.f, dA,
                    dB, 0.f, dC);
      alpaka::memcpy(queue, hC, dC);
      alpaka::wait(queue);
      for (std::size_t i = 0; i < ref.size(); ++i)
        worst = std::max(worst, std::abs(C[i] - ref[i]));
    }
    if (capped.algoCacheSize() <= 8 && worst < 1e-3f) {
      std::cout << "  PASS  hip::cache limit honoured\n";
    } else {
      std::cerr << "  FAIL [hip::cache limit honoured] "
                << capped.algoCacheSize() << " entries, worst err " << worst
                << "\n";
      ++gFailures;
    }
  }
}

#endif // ALPAKA_ACC_GPU_HIP_ENABLED

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
  runCpuTests();
#endif
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
  runCudaTests();
  runDynamicShapeTests();
#endif
#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
  runHipTests();
  runHipDynamicShapeTests();
#endif

  std::cout << "\n";
  if (gFailures == 0)
    std::cout << "All tests passed.\n";
  else
    std::cout << gFailures << " test(s) FAILED.\n";

  return gFailures > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
