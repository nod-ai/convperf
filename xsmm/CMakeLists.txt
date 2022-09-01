project(xsmm_runner)
set(SRC xsmm.cpp)
add_library(xsmm_runner ${SRC})
target_link_libraries(xsmm_runner PRIVATE xsmm common pthread)
target_include_directories(xsmm_runner PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/../
  ${XSMM_INCLUDE_DIRS}
)
