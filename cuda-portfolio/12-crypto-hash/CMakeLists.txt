cuda_portfolio_add_library(sha256_engine SOURCES src/sha256_engine.cu)

cuda_portfolio_add_app(bench_sha256
  SOURCES src/bench_sha256.cpp
  LINK sha256_engine)

cuda_portfolio_add_test(test_sha256_engine
  SOURCES tests/test_sha256_engine.cpp
  LINK sha256_engine)
