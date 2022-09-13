project(common)

set(SRC utils.cpp)
add_library(common ${SRC})
target_include_directories(common PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/../
)
target_link_libraries(common PRIVATE nlohmann_json::nlohmann_json)

