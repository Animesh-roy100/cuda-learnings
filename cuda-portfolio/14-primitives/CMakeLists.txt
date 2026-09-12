cuda_portfolio_add_library(primitives SOURCES src/primitives.cu)

cuda_portfolio_add_app(bench_primitives
  SOURCES src/bench_primitives.cpp
  LINK primitives)

cuda_portfolio_add_test(test_primitives
  SOURCES tests/test_primitives.cpp
  LINK primitives)
