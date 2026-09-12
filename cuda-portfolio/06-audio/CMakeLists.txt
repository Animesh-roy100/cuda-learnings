cuda_portfolio_add_library(audio_dsp
  SOURCES src/audio_dsp.cu
  LINK CUDA::cufft)

cuda_portfolio_add_app(bench_audio
  SOURCES src/bench_audio.cpp
  LINK audio_dsp)

cuda_portfolio_add_test(test_audio_dsp
  SOURCES tests/test_audio_dsp.cpp
  LINK audio_dsp)
