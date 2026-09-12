cuda_portfolio_add_library(image_pipeline
  SOURCES src/preprocess.cu src/batcher.cpp)

cuda_portfolio_add_app(bench_image
  SOURCES src/bench_image.cpp
  LINK image_pipeline)

cuda_portfolio_add_test(test_image_pipeline
  SOURCES tests/test_image_pipeline.cpp
  LINK image_pipeline)
