cmake_minimum_required(VERSION 3.24)

# CUDA is a first-class language here, not a bolted-on custom command.
project(cuda_portfolio
        VERSION 1.0.0
        DESCRIPTION "Production-grade CUDA systems targeting Turing sm_75"
        LANGUAGES CXX CUDA)

# ---------------------------------------------------------------------------
# Global standards
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)          # Thrust/CCCL needs >=17; 17 is the safe floor
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# sm_75 = Turing = GTX 1650. Compiling for the exact architecture avoids
# start-up JIT and unlocks every sm_75 instruction (notably __dp4a).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Put everything in one place so the profiling scripts have a stable path.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
foreach(cfg IN ITEMS DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${cfg} ${CMAKE_BINARY_DIR}/bin)
endforeach()

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ProjectDefaults.cmake)

option(CUDA_PORTFOLIO_BUILD_TESTS "Build GoogleTest suites" ON)

# ---------------------------------------------------------------------------
# Shared header-only support library
# ---------------------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)

add_library(cu_common INTERFACE)
target_include_directories(cu_common INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/common/include)
target_link_libraries(cu_common INTERFACE CUDA::cudart)

if(WIN32)
  # CUDA 13 puts the math-library DLLs under bin/x64. That directory is on the
  # machine PATH, but any shell opened before the toolkit was installed has a
  # stale environment and the test binaries then die with 0xC0000135
  # (DLL not found) and no message. Staging the DLLs beside the executables
  # makes a fresh checkout run without PATH surgery.
  file(GLOB CUDA_RUNTIME_DLLS
    "${CUDAToolkit_BIN_DIR}/cudart64_*.dll"
    "${CUDAToolkit_BIN_DIR}/x64/cudart64_*.dll"
    "${CUDAToolkit_BIN_DIR}/x64/cufft64_*.dll"
    "${CUDAToolkit_BIN_DIR}/x64/curand64_*.dll"
    "${CUDAToolkit_BIN_DIR}/x64/cublas64_*.dll"
    "${CUDAToolkit_BIN_DIR}/x64/cublasLt64_*.dll")
  if(CUDA_RUNTIME_DLLS)
    file(COPY ${CUDA_RUNTIME_DLLS} DESTINATION ${CMAKE_BINARY_DIR}/bin)
    list(LENGTH CUDA_RUNTIME_DLLS _n_dlls)
    message(STATUS "Staged ${_n_dlls} CUDA runtime DLLs into bin/")
  endif()
endif()

# ---------------------------------------------------------------------------
# GoogleTest
# ---------------------------------------------------------------------------
if(CUDA_PORTFOLIO_BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  # Keep GoogleTest on the shared CRT so it links against our targets on MSVC.
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  enable_testing()
  include(GoogleTest)
endif()

# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------
add_subdirectory(01-gguf-inference)
add_subdirectory(02-image-embed)
add_subdirectory(03-hash-kv)
add_subdirectory(04-spatial)
add_subdirectory(05-video)
add_subdirectory(06-audio)
add_subdirectory(07-montecarlo)
add_subdirectory(08-graph)

message(STATUS "")
message(STATUS "cuda_portfolio configured")
message(STATUS "  CUDA architectures : ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Build type         : ${CMAKE_BUILD_TYPE}")
message(STATUS "  Tests              : ${CUDA_PORTFOLIO_BUILD_TESTS}")
message(STATUS "")
