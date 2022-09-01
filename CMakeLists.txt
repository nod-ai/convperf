cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(convperf C CXX)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake/")
set(TP_ROOT ${CMAKE_CURRENT_LIST_DIR}/external)

include(libxsmm)
add_subdirectory(common)
add_subdirectory(naive)
add_subdirectory(xsmm)
add_subdirectory(tools)
