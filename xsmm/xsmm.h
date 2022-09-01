#pragma once

#include "common/runner.h"
#include "common/utils.h"
#include "libxsmm_dnn.h"
#include "dnn_common.h"

namespace convperf {

class XSMMRunner : public Runner {
public:
  XSMMRunner(const ConvParams &params);
  void run(const float *a, const float *b, float *c);
  ~XSMMRunner();

private:
  libxsmm_dnn_conv_config cfg;
  float *input_save, *output_save, *filter_save;
  float *input_libxsmm, *output_libxsmm, *filter_libxsmm;
  const ConvParams &params;
};

}
