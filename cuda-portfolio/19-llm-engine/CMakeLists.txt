# The GGUF parser and its headers come from 01-gguf-inference, which this
# engine extends from a set of kernels into a program that generates text.
cuda_portfolio_add_library(llm_engine
  SOURCES src/engine.cu src/tokenizer.cpp src/quant.cpp src/reference.cpp
  LINK gguf_inference CUDA::cublas)

cuda_portfolio_add_app(chat SOURCES src/chat.cpp LINK llm_engine)
cuda_portfolio_add_app(bench_engine SOURCES src/bench_engine.cpp LINK llm_engine)

cuda_portfolio_add_test(test_llm_engine
  SOURCES tests/test_llm_engine.cpp
  LINK llm_engine)
