set(CXXFLAGS -O3 -DALPAKA_HAS_STD_ATOMIC_REF)
set(CXX_HOST_FLAGS -fPIC -pthread)
set(CXX_CUDA_FLAGS -Wno-deprecated-gpu-targets --extended-lambda --expt-relaxed-constexpr)
set(XCOMPILER_FLAGS -Xcompiler=-fPIC,-pthread)

if(SOFIEBLAS_CPU_BLAS_LIBS)
  add_executable(bench_cpu bench.cc)
  target_compile_features(bench_cpu PUBLIC cxx_std_20)
  target_compile_options(bench_cpu PRIVATE ${CXXFLAGS} ${CXX_HOST_FLAGS})
  target_compile_definitions(bench_cpu PRIVATE
    ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED ${SOFIEBLAS_CPU_BLAS_DEFINE})
  target_include_directories(bench_cpu PRIVATE ${SOFIEBLAS_CPU_BLAS_INCLUDE_DIR})
  target_link_libraries(bench_cpu PRIVATE
    sofieBLAS::sofieBLAS alpaka::alpaka ${SOFIEBLAS_CPU_BLAS_LIBS})

  if(SOFIEBLAS_BUILD_TESTS)
    add_test(NAME sofieBLAS.bench.cpu.smoke
      COMMAND bench_cpu -w 1 -n 1 --sizes 64,128)
  endif()
else()
  message(STATUS "sofieBLAS benchmark: skipping bench_cpu (no CPU BLAS library found)")
endif()

if(SOFIEBLAS_CUDA_ENABLED)
  configure_file(bench.cc bench_cuda_gen.cc COPYONLY)
  add_executable(bench_cuda)
  target_sources(bench_cuda PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/bench_cuda_gen.cc)
  set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/bench_cuda_gen.cc PROPERTIES LANGUAGE CUDA)
  set_target_properties(bench_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_compile_features(bench_cuda PUBLIC cxx_std_20)

  target_compile_options(bench_cuda PRIVATE
    ${CXXFLAGS}
    ${CXX_CUDA_FLAGS}
    ${XCOMPILER_FLAGS}
  )

  target_compile_definitions(bench_cuda PRIVATE ALPAKA_ACC_GPU_CUDA_ENABLED)
  target_link_libraries(bench_cuda PRIVATE
    sofieBLAS::sofieBLAS alpaka::alpaka CUDA::cudart CUDA::cublas CUDA::cublasLt)

  if(SOFIEBLAS_BUILD_TESTS)
    add_test(NAME sofieBLAS.bench.cuda.smoke
      COMMAND bench_cuda -w 1 -n 1 --sizes 64,128)
  endif()
else()
  message(VERBOSE "sofieBLAS benchmark: skipping bench_cuda (CUDA backend not enabled)")
endif()

if(SOFIEBLAS_HIP_ENABLED)
  configure_file(bench.cc bench_hip_gen.cc COPYONLY)
  add_executable(bench_hip)
  target_sources(bench_hip PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/bench_hip_gen.cc)
  set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/bench_hip_gen.cc PROPERTIES LANGUAGE HIP)
  target_compile_features(bench_hip PUBLIC cxx_std_20)

  target_compile_options(bench_hip PRIVATE ${CXXFLAGS} ${CXX_HOST_FLAGS})

  target_compile_definitions(bench_hip PRIVATE ALPAKA_ACC_GPU_HIP_ENABLED)
  target_include_directories(bench_hip PRIVATE ${ROCM_BASE}/include)
  target_link_directories(bench_hip PRIVATE ${ROCM_BASE}/lib)
  target_link_libraries(bench_hip PRIVATE
    sofieBLAS::sofieBLAS alpaka::alpaka hipblaslt hipblas amdhip64)

  if(SOFIEBLAS_BUILD_TESTS)
    add_test(NAME sofieBLAS.bench.hip.smoke
      COMMAND bench_hip -w 1 -n 1 --sizes 64,128)
  endif()
else()
  message(VERBOSE "sofieBLAS benchmark: skipping bench_hip (HIP backend not enabled)")
endif()
