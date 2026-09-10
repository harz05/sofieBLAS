cmake_minimum_required(VERSION 3.21)
project(sofieBLAS
    VERSION 0.1.0
    DESCRIPTION "Header-only C++ BLAS abstraction over heterogeneous backends"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(SOFIEBLAS_BUILD_TESTS "Build sofieBLAS unit tests" OFF)
option(SOFIEBLAS_BUILD_BENCHMARKS "Build sofieBLAS GEMM benchmarks" OFF)

# --- Header-only interface library ---
add_library(sofieBLAS INTERFACE)
add_library(sofieBLAS::sofieBLAS ALIAS sofieBLAS)

target_include_directories(sofieBLAS INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_compile_features(sofieBLAS INTERFACE cxx_std_20)

if(SOFIEBLAS_BUILD_TESTS OR SOFIEBLAS_BUILD_BENCHMARKS)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
  include(SofieBLASBackends)
endif()

if(SOFIEBLAS_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(SOFIEBLAS_BUILD_BENCHMARKS)
  add_subdirectory(benchmark)
endif()

# --- Install (header-only package) ---
install(TARGETS sofieBLAS EXPORT sofieBLASTargets)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/sofieBLASConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/sofieBLASConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sofieBLAS
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/sofieBLASConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
  ARCH_INDEPENDENT
)

install(
  EXPORT sofieBLASTargets
  FILE sofieBLASTargets.cmake
  NAMESPACE sofieBLAS::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sofieBLAS
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/sofieBLASConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/sofieBLASConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sofieBLAS
)
