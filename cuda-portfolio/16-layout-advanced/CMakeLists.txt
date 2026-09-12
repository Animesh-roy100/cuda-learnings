# Dynamic parallelism is the one feature here that changes how the whole
# translation unit is compiled: a kernel launch inside device code needs
# relocatable device code (-rdc=true) and the device runtime library, so the
# defaults from ProjectDefaults.cmake are overridden for this target only.
cuda_portfolio_add_library(layout_advanced SOURCES src/layout_advanced.cu)

set_target_properties(layout_advanced PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON)
if(TARGET CUDA::cudadevrt)
  target_link_libraries(layout_advanced PUBLIC CUDA::cudadevrt)
else()
  # CMake 4.x FindCUDAToolkit does not always define the imported target even
  # when the library is present. The device-link step CMake performs for a
  # CUDA_RESOLVE_DEVICE_SYMBOLS target already pulls cudadevrt in, so this is a
  # belt-and-braces link rather than a requirement.
  find_library(CUDA_DEVRT_LIB cudadevrt HINTS ${CUDAToolkit_LIBRARY_DIR})
  if(CUDA_DEVRT_LIB)
    target_link_libraries(layout_advanced PUBLIC ${CUDA_DEVRT_LIB})
  endif()
endif()

cuda_portfolio_add_app(bench_layout_advanced
  SOURCES src/bench_layout_advanced.cpp
  LINK layout_advanced)

cuda_portfolio_add_test(test_layout_advanced
  SOURCES tests/test_layout_advanced.cpp
  LINK layout_advanced)
