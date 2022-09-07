project(tools)

add_executable(benchmark_conv benchmark_conv.cpp)
target_link_libraries(benchmark_conv PUBLIC
  iree_runner
  naive_runner
  xsmm_runner
  benchmark
)
target_include_directories(benchmark_conv PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/../
)

add_executable(verify_conv verify_conv.cpp)
target_link_libraries(verify_conv PUBLIC
  naive_runner
)
target_include_directories(verify_conv PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/../
)
