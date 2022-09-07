// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Forked from IREE (with modified includes).

#include "iree/iree.h"
#include <iree/base/allocator.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <math.h>
#include <string>
#include <iostream>
#include <benchmark/benchmark.h>

#include "convs.h"

// A function to create the HAL device from the different backend targets.
// The HAL device is returned based on the implementation, and it must be
// released by the caller.
extern "C"{
  iree_status_t create_sample_device(iree_allocator_t host_allocator,
                                     iree_hal_device_t** out_device);
}

namespace convperf {

void IREERunner::setShape(int idx, const Shape4D &shape) {
  for (int i = 0; i < 4; i++) {
    switch(shape.format[i]) {
      case 'n':
      case 'f':
        arg_shape[4*idx + i] = shape.N;
        break;
      case 'h':
        arg_shape[4*idx + i] = shape.H;
        break;
      case 'w':
        arg_shape[4*idx + i] = shape.W;
        break;
      case 'c':
        arg_shape[4*idx + i] = shape.C;
        break;
    }
  }
}

IREERunner::IREERunner(const ConvParams &params) : params(params) {
  auto status = initialize();
  if (status != iree_status_from_code(IREE_STATUS_OK)) {
    printf("Initialization failed!\n");
  }
}

iree_status_t IREERunner::initialize() {
  instance = NULL;
  IREE_RETURN_IF_ERROR(
      iree_vm_instance_create(iree_allocator_system(), &instance));
  IREE_RETURN_IF_ERROR(iree_hal_module_register_all_types(instance));

  device = NULL;
  IREE_RETURN_IF_ERROR(create_sample_device(iree_allocator_system(), &device));
  iree_vm_module_t *hal_module = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_module_create(instance, device, IREE_HAL_MODULE_FLAG_SYNCHRONOUS,
                             iree_allocator_system(), &hal_module));

  const struct iree_file_toc_t* module_file_toc = convs_create();

  iree_vm_module_t *bytecode_module = NULL;
  iree_const_byte_span_t module_data =
      iree_make_const_byte_span(module_file_toc->data, module_file_toc->size);
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_module_create(
      instance, module_data, iree_allocator_null(), iree_allocator_system(),
      &bytecode_module));

  // Allocate a context that will hold the module state across invocations.
  context = NULL;
  iree_vm_module_t* modules[] = {hal_module, bytecode_module};
  IREE_RETURN_IF_ERROR(iree_vm_context_create_with_modules(
      instance, IREE_VM_CONTEXT_FLAG_TRACE_EXECUTION, IREE_ARRAYSIZE(modules),
      &modules[0], iree_allocator_system(), &context));
  iree_vm_module_release(hal_module);
  iree_vm_module_release(bytecode_module);

  // Lookup the entry point function.
  // Note that we use the synchronous variant which operates on pure type/shape
  // erased buffers.
  std::string kMainFunctionName = "module.conv2d_" + params.inputShape.str()
                                + "_" + params.filterShape.str();
  IREE_RETURN_IF_ERROR(iree_vm_context_resolve_function(
      context, iree_make_cstring_view(kMainFunctionName.c_str()), &main_function));

  arg_shape = (iree_hal_dim_t *) malloc(2 * 4 * sizeof(iree_hal_dim_t));

  element_type = IREE_HAL_ELEMENT_TYPE_FLOAT_32;
  IREE_RETURN_IF_ERROR(iree_vm_list_create(
                           /*element_type=*/NULL,
                           /*capacity=*/2, iree_allocator_system(), &inputs),
                       "can't allocate input vm list");
  setShape(0, params.inputShape);
  setShape(1, params.filterShape);
  return iree_status_from_code(IREE_STATUS_OK);
}

void IREERunner::setup(const float *input, const float *filter, float *output) {
  size_t ndim{4};
  for (int i = 0; i < 2; i++) {
    iree_const_byte_span_t initial_data;
    if (i == 0) {
      initial_data = iree_make_const_byte_span((void *) input, params.inputShape.getLinearizedShape() * sizeof(float));
    } else {
      initial_data = iree_make_const_byte_span((void *) filter, params.filterShape.getLinearizedShape() * sizeof(float));
    }
    iree_hal_buffer_view_t *arg_buffer_view = NULL;
    iree_hal_buffer_view_allocate_buffer(
        iree_hal_device_allocator(device), ndim, &arg_shape[i * ndim],
        element_type, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
        (iree_hal_buffer_params_t){
            .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
            .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
        },
        initial_data, &arg_buffer_view);
    iree_vm_ref_t arg_input_buffer_view_ref = iree_hal_buffer_view_move_ref(arg_buffer_view);
    iree_vm_list_push_ref_move(inputs, &arg_input_buffer_view_ref);
  }
}

void IREERunner::run(const float *input, const float *filter, float *output) {
  iree_vm_list_create(/*element_type=*/NULL,
                      /*capacity=*/1, iree_allocator_system(), &outputs);
  // Synchronously invoke the function.
  IREE_CHECK_OK(iree_vm_invoke(context, main_function,
                               IREE_VM_INVOCATION_FLAG_NONE,
                               /*policy=*/NULL, inputs, outputs,
                               iree_allocator_system()));
}

void IREERunner::getResults(float *output) {
  // Get the result buffers from the invocation.
  iree_hal_buffer_view_t *ret_buffer_view =
      (iree_hal_buffer_view_t*)iree_vm_list_get_ref_deref(
          outputs, 0, iree_hal_buffer_view_get_descriptor());

  // Read back the results and ensure we got the right values.
  IREE_CHECK_OK(iree_hal_device_transfer_d2h(
      device, iree_hal_buffer_view_buffer(ret_buffer_view), 0, output,
      params.outputShape.getLinearizedShape() * sizeof(float),
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
      iree_infinite_timeout()));

}

IREERunner::~IREERunner() {
  free(arg_shape);
  iree_vm_list_release(inputs);
  iree_vm_list_release(outputs);
  iree_hal_device_release(device);
  iree_vm_context_release(context);
  iree_vm_instance_release(instance);
}

}
