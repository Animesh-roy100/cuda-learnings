cuda_portfolio_add_library(spatial_index SOURCES src/spatial_index.cu)

cuda_portfolio_add_app(bench_spatial
  SOURCES src/bench_spatial.cpp
  LINK spatial_index)

cuda_portfolio_add_test(test_spatial_index
  SOURCES tests/test_spatial_index.cpp
  LINK spatial_index)
