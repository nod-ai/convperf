project(xsmm_runner)
set(SRC xsmm.cpp)
add_library(xsmm_runner ${SRC})
set(LIBS xsmm common pthread)
target_include_directories(xsmm_runner PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/../
  ${XSMM_INCLUDE_DIRS}
)

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
  list(APPEND LIBS OpenMP::OpenMP_CXX)
endif()

target_link_libraries(xsmm_runner PUBLIC ${LIBS})
