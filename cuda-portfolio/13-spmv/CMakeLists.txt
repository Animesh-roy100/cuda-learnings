cuda_portfolio_add_library(spmv SOURCES src/spmv.cu)

cuda_portfolio_add_app(bench_spmv
  SOURCES src/bench_spmv.cpp
  LINK spmv)

cuda_portfolio_add_test(test_spmv
  SOURCES tests/test_spmv.cpp
  LINK spmv)
