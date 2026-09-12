cuda_portfolio_add_library(flash_attention SOURCES src/flash_attention.cu LINK CUDA::cublas)

cuda_portfolio_add_app(bench_flash_attention
  SOURCES src/bench_flash_attention.cpp
  LINK flash_attention)

cuda_portfolio_add_test(test_flash_attention
  SOURCES tests/test_flash_attention.cpp
  LINK flash_attention)
