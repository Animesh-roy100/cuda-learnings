cuda_portfolio_add_library(optical_flow SOURCES src/optical_flow.cu)

cuda_portfolio_add_app(bench_flow
  SOURCES src/bench_flow.cpp
  LINK optical_flow)

cuda_portfolio_add_test(test_optical_flow
  SOURCES tests/test_optical_flow.cpp
  LINK optical_flow)
