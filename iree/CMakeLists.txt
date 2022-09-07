project(iree_runner)

set(_TRANSLATE_TOOL_EXECUTABLE $<TARGET_FILE:iree-compile>)
set(PYTHON_TOOL ${CMAKE_CURRENT_LIST_DIR}/compile_conv.py)
set(_ARGS)
list(APPEND _ARGS "--sizes_file=${BENCHMARK_SIZES}")
list(APPEND _ARGS "--input_format=${INPUT_FORMAT}")
list(APPEND _ARGS "--filter_format=${FILTER_FORMAT}")
list(APPEND _ARGS "--compile_tool=${_TRANSLATE_TOOL_EXECUTABLE}")
add_custom_command(
  OUTPUT convs.vmfb
  COMMAND ${PYTHON_TOOL} ${_ARGS}
  DEPENDS ${_TRANSLATE_TOOL_EXECUTABLE}
)

set(_ARGS)
list(APPEND _ARGS "--output_header=convs.h")
list(APPEND _ARGS "--output_impl=convs.c")
list(APPEND _ARGS "--identifier=convs")
list(APPEND _ARGS "--flatten")
list(APPEND _ARGS "convs.vmfb")
add_custom_command(
    OUTPUT "convs.h" "convs.c"
    COMMAND generate_embed_data ${_ARGS}
    DEPENDS generate_embed_data convs.vmfb
)

if(IREE_CPU_MULTITHREADED)
  message(STATUS "Enabling local task")
  add_compile_definitions(USE_LOCAL_TASK)
else()
  message(STATUS "Enabling local sync")
endif()

set(SRC iree.cpp convs.c convs.h device_cpu.c)
add_library(iree_runner ${SRC})
set(IREE_CPU_DRIVER_LIB iree_hal_drivers_local_sync_sync_driver)
if(IREE_CPU_MULTITHREADED)
  set(IREE_CPU_DRIVER_LIB iree_hal_drivers_local_task_task_driver)
endif()
target_link_libraries(iree_runner PUBLIC
  common
  iree_base_base
  iree_hal_hal
  iree_hal_local_local
  ${IREE_CPU_DRIVER_LIB}
  iree_hal_local_loaders_embedded_elf_loader
  iree_hal_local_loaders_vmvx_module_loader
  iree_modules_hal_hal
  iree_task_api
  iree_vm_vm
  iree_vm_bytecode_module
  benchmark
)
target_include_directories(iree_runner PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/../
  ${CMAKE_CURRENT_BINARY_DIR}
)
