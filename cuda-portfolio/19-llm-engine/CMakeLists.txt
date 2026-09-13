# The GGUF parser comes from 01-gguf-inference, which this runtime extends from a
# set of kernels into a program that generates text.
#
# llm_host holds everything that needs no GPU -- configuration and tensor
# validation, the tokenizer, dequantization, page allocation, sampling, errors
# -- so its tests run in CI under AddressSanitizer and UBSan without a device.
cuda_portfolio_add_library(llm_host
  SOURCES src/errors.cpp src/model_config.cpp src/tokenizer.cpp src/quant.cpp
          src/page_allocator.cpp src/sampler.cpp src/reference.cpp src/llamacpp_reference.cpp
  LINK gguf_inference)

cuda_portfolio_add_library(llm_engine
  SOURCES src/runtime.cu src/gguf_model.cpp src/generator.cpp src/engine.cpp
  LINK llm_host CUDA::cublas CUDA::nvml)

cuda_portfolio_add_app(chat SOURCES src/chat.cpp LINK llm_engine)
cuda_portfolio_add_app(bench_engine SOURCES src/bench_engine.cpp LINK llm_engine)
cuda_portfolio_add_app(compare_llamacpp SOURCES src/compare_llamacpp.cpp LINK llm_engine)

cuda_portfolio_add_test(test_llm_host
  SOURCES tests/test_llm_host.cpp
  LINK llm_host)

cuda_portfolio_add_test(test_llm_engine
  SOURCES tests/test_llm_engine.cpp
  LINK llm_engine)
