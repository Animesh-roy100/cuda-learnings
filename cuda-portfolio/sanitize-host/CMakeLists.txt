# Host code under AddressSanitizer and UndefinedBehaviorSanitizer.
#
# The GGUF parser, model configuration, tokenizer, dequantization, sampler,
# page allocator and FP32 reference need no GPU, so they build here with no
# CUDA toolkit and RUN on any Linux machine -- including a GitHub-hosted
# runner, which the device suites cannot. The parser tests feed hostile files,
# which is exactly where a sanitizer earns its keep.
#
#   cmake -S sanitize-host -B build-san -DCMAKE_BUILD_TYPE=Debug
#   cmake --build build-san -j && ctest --test-dir build-san --output-on-failure
#
# Device code is checked by compute-sanitizer instead (scripts/sanitize.*).

cmake_minimum_required(VERSION 3.24)
project(cuda_portfolio_host_sanitized LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
  message(FATAL_ERROR "UBSan is not available with MSVC; build this with gcc or clang on Linux")
endif()

set(ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(SAN -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer)
add_compile_options(${SAN} -g -O1 -Wall -Wextra)
add_link_options(${SAN})

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
find_package(Threads REQUIRED)

add_library(host_code STATIC
  ${ROOT}/01-gguf-inference/src/gguf.cpp
  ${ROOT}/19-llm-engine/src/errors.cpp
  ${ROOT}/19-llm-engine/src/model_config.cpp
  ${ROOT}/19-llm-engine/src/tokenizer.cpp
  ${ROOT}/19-llm-engine/src/quant.cpp
  ${ROOT}/19-llm-engine/src/page_allocator.cpp
  ${ROOT}/19-llm-engine/src/sampler.cpp
  ${ROOT}/19-llm-engine/src/reference.cpp)
target_include_directories(host_code PUBLIC
  ${ROOT}/01-gguf-inference/include
  ${ROOT}/19-llm-engine/include)
target_link_libraries(host_code PUBLIC Threads::Threads)

enable_testing()
foreach(t IN ITEMS
    "test_gguf|${ROOT}/01-gguf-inference/tests/test_gguf.cpp"
    "test_llm_host|${ROOT}/19-llm-engine/tests/test_llm_host.cpp")
  string(REPLACE "|" ";" parts "${t}")
  list(GET parts 0 name)
  list(GET parts 1 source)
  add_executable(${name} ${source})
  target_link_libraries(${name} PRIVATE host_code GTest::gtest_main)
  add_test(NAME ${name} COMMAND ${name})
  set_tests_properties(${name} PROPERTIES ENVIRONMENT
    "ASAN_OPTIONS=detect_leaks=1:abort_on_error=1:strict_string_checks=1;UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1")
endforeach()
