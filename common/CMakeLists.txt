project(common)

set(SRC utils.cpp)
add_library(common ${SRC})
target_include_directories(common PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/../
)

