project(iree_runner)
set(SRC iree.cpp)
add_library(iree_runner ${SRC})
target_link_libraries(iree_runner PRIVATE
    common
)
target_include_directories(iree_runner PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/../
)
