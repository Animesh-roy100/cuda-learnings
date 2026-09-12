cuda_portfolio_add_library(ann_index SOURCES src/ann_index.cu)

cuda_portfolio_add_app(bench_ann
  SOURCES src/bench_ann.cpp
  LINK ann_index)

cuda_portfolio_add_test(test_ann_index
  SOURCES tests/test_ann_index.cpp
  LINK ann_index)
