cuda_portfolio_add_library(warp_primitives SOURCES src/warp_primitives.cu)

cuda_portfolio_add_app(bench_warp_primitives
  SOURCES src/bench_warp_primitives.cpp
  LINK warp_primitives)

cuda_portfolio_add_test(test_warp_primitives
  SOURCES tests/test_warp_primitives.cpp
  LINK warp_primitives)
