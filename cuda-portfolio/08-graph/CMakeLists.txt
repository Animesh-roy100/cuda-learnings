cuda_portfolio_add_library(graph_engine SOURCES src/graph_engine.cu)

cuda_portfolio_add_app(bench_graph
  SOURCES src/bench_graph.cpp
  LINK graph_engine)

cuda_portfolio_add_test(test_graph_engine
  SOURCES tests/test_graph_engine.cpp
  LINK graph_engine)
