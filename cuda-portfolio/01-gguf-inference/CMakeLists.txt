cuda_portfolio_add_library(gguf_inference
  SOURCES src/gguf.cpp src/kernels.cu src/kv_cache.cu)

cuda_portfolio_add_app(bench_inference
  SOURCES src/bench_inference.cpp
  LINK gguf_inference)

cuda_portfolio_add_test(test_gguf
  SOURCES tests/test_gguf.cpp
  LINK gguf_inference)

cuda_portfolio_add_test(test_kernels
  SOURCES tests/test_kernels.cpp
  LINK gguf_inference)
