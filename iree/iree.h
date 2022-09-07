#pragma once

#include "common/runner.h"
#include "common/utils.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/module.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode_module.h"
#include "iree/base/internal/math.h"

namespace convperf {

class IREERunner : public Runner {
public:
  IREERunner(const ConvParams &params);
  void run(const float *a, const float *b, float *c);
  void setup(const float *a, const float *b, float *c);
  void getResults(float *c);
  ~IREERunner();

private:
  const ConvParams &params;
  iree_vm_instance_t *instance;
  iree_hal_device_t *device;
  iree_vm_context_t *context;
  iree_vm_list_t *inputs, *outputs;
  iree_hal_element_type_t element_type;
  iree_hal_dim_t *arg_shape;
  iree_vm_function_t main_function;
  void setShape(int idx, const Shape4D &shape);
  iree_status_t initialize();
};

}
