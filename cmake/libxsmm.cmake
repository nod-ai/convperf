# From: https://github.com/libxsmm/libxsmm
set(LIBXSMMROOT ${TP_ROOT}/libxsmm)
file(GLOB _GLOB_XSMM_SRCS LIST_DIRECTORIES false CONFIGURE_DEPENDS ${LIBXSMMROOT}/src/*.c)
list(REMOVE_ITEM _GLOB_XSMM_SRCS ${LIBXSMMROOT}/src/libxsmm_generator_gemm_driver.c)
file(GLOB _GLOB_XSMM_DNN_SRCS LIST_DIRECTORIES false CONFIGURE_DEPENDS ${LIBXSMMROOT}/samples/deeplearning/libxsmm_dnn/src/*.c)
set(XSMM_INCLUDE_DIRS
  ${LIBXSMMROOT}/include
  ${LIBXSMMROOT}/samples/deeplearning/libxsmm_dnn/include
  ${TP_ROOT}/libxsmm/src/template/
)

add_library(xsmm STATIC ${_GLOB_XSMM_SRCS} ${_GLOB_XSMM_DNN_SRCS})
target_include_directories(xsmm PUBLIC ${XSMM_INCLUDE_DIRS})
target_compile_definitions(xsmm PUBLIC
  LIBXSMM_DEFAULT_CONFIG
)
target_compile_definitions(xsmm PRIVATE
  __BLAS=0
)
