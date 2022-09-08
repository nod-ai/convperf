cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(convperf C CXX)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake/")
set(TP_ROOT ${CMAKE_CURRENT_LIST_DIR}/external)

set(BENCHMARK_SIZES ${CMAKE_CURRENT_LIST_DIR}/benchmark_sizes/resnet50.txt)
add_compile_definitions(BENCHMARK_SIZES=${BENCHMARK_SIZES})
set(INPUT_FORMAT "nhwc")
add_compile_definitions(INPUT_FORMAT=${INPUT_FORMAT})
set(FILTER_FORMAT "hwcf")
add_compile_definitions(FILTER_FORMAT=${FILTER_FORMAT})

include(libxsmm)

set(BENCHMARK_ENABLE_TESTING OFF)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_BINARY_DIR}/lib/cmake/mlir")
add_subdirectory(external/iree EXCLUDE_FROM_ALL)

add_subdirectory(common)

set(IREE_CPU_MULTITHREADED ON)
add_subdirectory(iree)
add_subdirectory(naive)
add_subdirectory(xsmm)
add_subdirectory(tools)
