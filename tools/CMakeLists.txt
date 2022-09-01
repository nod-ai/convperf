project(benchmark_conv)

add_executable(benchmark_conv benchmark_conv.cpp)
target_link_libraries(benchmark_conv PUBLIC
  naive_runner
  xsmm_runner
)
target_include_directories(benchmark_conv PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/../
)
