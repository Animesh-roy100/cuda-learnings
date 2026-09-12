cuda_portfolio_add_library(dpi_engine SOURCES src/dpi_engine.cu)

cuda_portfolio_add_app(bench_dpi
  SOURCES src/bench_dpi.cpp
  LINK dpi_engine)

cuda_portfolio_add_test(test_dpi_engine
  SOURCES tests/test_dpi_engine.cpp
  LINK dpi_engine)
