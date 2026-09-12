cuda_portfolio_add_library(hash_kv SOURCES src/hash_kv.cu)

cuda_portfolio_add_app(bench_hash_kv
  SOURCES src/bench_hash_kv.cpp
  LINK hash_kv)

cuda_portfolio_add_test(test_hash_kv
  SOURCES tests/test_hash_kv.cpp
  LINK hash_kv)
