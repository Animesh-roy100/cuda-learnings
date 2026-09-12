cuda_portfolio_add_library(video_pipeline SOURCES src/video_pipeline.cu)

cuda_portfolio_add_app(bench_video
  SOURCES src/bench_video.cpp
  LINK video_pipeline)

cuda_portfolio_add_test(test_video_pipeline
  SOURCES tests/test_video_pipeline.cpp
  LINK video_pipeline)
