set(CXXFLAGS -O2 -g -DALPAKA_HAS_STD_ATOMIC_REF)
set(CXX_HOST_FLAGS -fPIC -pthread)
set(CXX_CUDA_FLAGS -Wno-deprecated-gpu-targets --extended-lambda --expt-relaxed-constexpr)
set(XCOMPILER_FLAGS -Xcompiler=-fPIC,-pthread)

if(SOFIEBLAS_CPU_BLAS_LIBS)
  add_executable(test_cpu test.cc)
  target_compile_features(test_cpu PUBLIC cxx_std_20)
  target_compile_options(test_cpu PRIVATE ${CXXFLAGS} ${CXX_HOST_FLAGS})
  target_compile_definitions(test_cpu PRIVATE
    ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED ${SOFIEBLAS_CPU_BLAS_DEFINE})
  target_include_directories(test_cpu PRIVATE ${SOFIEBLAS_CPU_BLAS_INCLUDE_DIR})
  target_link_libraries(test_cpu PRIVATE
    sofieBLAS::sofieBLAS alpaka::alpaka ${SOFIEBLAS_CPU_BLAS_LIBS})

  add_test(NAME sofieBLAS.cpu COMMAND test_cpu)
else()
  message(STATUS "sofieBLAS tests: skipping test_cpu (no CPU BLAS library found)")
endif()

if(SOFIEBLAS_CUDA_ENABLED)
  configure_file(test.cc test_cuda_gen.cc COPYONLY)
  add_executable(test_cuda)
  target_sources(test_cuda PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/test_cuda_gen.cc)
  set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/test_cuda_gen.cc PROPERTIES LANGUAGE CUDA)
  set_target_properties(test_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_compile_features(test_cuda PUBLIC cxx_std_20)

  target_compile_options(test_cuda PRIVATE
    ${CXXFLAGS}
    ${CXX_CUDA_FLAGS}
    ${XCOMPILER_FLAGS}
  )

  target_compile_definitions(test_cuda PRIVATE ALPAKA_ACC_GPU_CUDA_ENABLED)
  target_include_directories(test_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(test_cuda PRIVATE
    sofieBLAS::sofieBLAS alpaka::alpaka CUDA::cudart CUDA::cublas CUDA::cublasLt)

  add_test(NAME sofieBLAS.cuda COMMAND test_cuda)
else()
  message(VERBOSE "sofieBLAS tests: skipping test_cuda (CUDA backend not enabled)")
endif()

if(SOFIEBLAS_HIP_ENABLED)
  configure_file(test.cc test_hip_gen.cc COPYONLY)
  add_executable(test_hip)
  target_sources(test_hip PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/test_hip_gen.cc)
  set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/test_hip_gen.cc PROPERTIES LANGUAGE HIP)
  target_compile_features(test_hip PUBLIC cxx_std_20)

  target_compile_options(test_hip PRIVATE ${CXXFLAGS} ${CXX_HOST_FLAGS})

  target_compile_definitions(test_hip PRIVATE ALPAKA_ACC_GPU_HIP_ENABLED)
  target_include_directories(test_hip PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${ROCM_BASE}/include)
  target_link_directories(test_hip PRIVATE ${ROCM_BASE}/lib)
  target_link_libraries(test_hip PRIVATE
    sofieBLAS::sofieBLAS alpaka::alpaka hipblaslt hipblas amdhip64)

  add_test(NAME sofieBLAS.hip COMMAND test_hip)
else()
  message(VERBOSE "sofieBLAS tests: skipping test_hip (HIP backend not enabled)")
endif()
