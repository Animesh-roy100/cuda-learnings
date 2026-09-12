cuda_portfolio_add_library(tc_gemm SOURCES src/tc_gemm.cu LINK CUDA::cublas)

cuda_portfolio_add_app(bench_tc_gemm
  SOURCES src/bench_tc_gemm.cpp
  LINK tc_gemm)

cuda_portfolio_add_test(test_tc_gemm
  SOURCES tests/test_tc_gemm.cpp
  LINK tc_gemm)
