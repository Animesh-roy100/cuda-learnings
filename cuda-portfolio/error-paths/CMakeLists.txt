# Error-path suite: out-of-memory, bad launch configurations, sticky faults,
# and constructors that throw halfway through acquiring device resources.
cuda_portfolio_add_library(error_kernels SOURCES src/error_kernels.cu)

cuda_portfolio_add_test(test_error_paths
  SOURCES tests/test_error_paths.cpp
  LINK error_kernels hash_kv video_pipeline audio_dsp gguf_inference mc_pricing)
