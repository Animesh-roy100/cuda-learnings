cuda_portfolio_add_library(mc_pricing
  SOURCES src/mc_pricing.cu
  LINK CUDA::curand)

cuda_portfolio_add_app(bench_mc
  SOURCES src/bench_mc.cpp
  LINK mc_pricing)

cuda_portfolio_add_test(test_mc_pricing
  SOURCES tests/test_mc_pricing.cpp
  LINK mc_pricing)
