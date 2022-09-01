project(naive_runner)
set(SRC naive.cpp)
add_library(naive_runner ${SRC})
target_link_libraries(naive_runner PRIVATE common)
target_include_directories(naive_runner PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/../
  ${XSMM_INCLUDE_DIRS}
)
