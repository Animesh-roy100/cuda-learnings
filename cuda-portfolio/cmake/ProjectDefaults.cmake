# Shared target configuration, so eight projects stay consistent without
# eight copies of the same flag list.

function(cuda_portfolio_apply_flags target)
  target_link_libraries(${target} PUBLIC cu_common)

  # Portable flags first. -lineinfo keeps source correlation in Nsight Compute
  # without the code-motion penalty of a full -G debug build.
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3>
  )

  # MSVC-only. /Zc:preprocessor is required because Thrust/CCCL refuse to build
  # against MSVC's traditional preprocessor -- but it is not a flag gcc or clang
  # understands, so passing it unconditionally breaks every non-Windows build
  # (Linux, WSL, and Google Colab included).
  if(MSVC)
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
      $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
      $<$<COMPILE_LANGUAGE:CXX>:/permissive->
    )
  endif()

  set_target_properties(${target} PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
endfunction()

# A library holding the kernels, with a CUDA-free public header.
function(cuda_portfolio_add_library name)
  cmake_parse_arguments(A "" "" "SOURCES;LINK" ${ARGN})
  add_library(${name} STATIC ${A_SOURCES})
  target_include_directories(${name} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
  cuda_portfolio_apply_flags(${name})
  if(A_LINK)
    target_link_libraries(${name} PUBLIC ${A_LINK})
  endif()
endfunction()

# A runnable benchmark/demo binary.
function(cuda_portfolio_add_app name)
  cmake_parse_arguments(A "" "" "SOURCES;LINK" ${ARGN})
  add_executable(${name} ${A_SOURCES})
  cuda_portfolio_apply_flags(${name})
  if(A_LINK)
    target_link_libraries(${name} PRIVATE ${A_LINK})
  endif()
endfunction()

# A GoogleTest suite, registered with CTest.
function(cuda_portfolio_add_test name)
  if(NOT CUDA_PORTFOLIO_BUILD_TESTS)
    return()
  endif()
  cmake_parse_arguments(A "" "" "SOURCES;LINK" ${ARGN})
  add_executable(${name} ${A_SOURCES})
  cuda_portfolio_apply_flags(${name})
  target_link_libraries(${name} PRIVATE GTest::gtest_main ${A_LINK})
  # Plain add_test, not gtest_discover_tests: discovery runs the binary at
  # build time, which initialises a CUDA context during the build.
  add_test(NAME ${name} COMMAND ${name})
endfunction()
