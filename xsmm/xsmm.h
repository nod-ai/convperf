#pragma once

#include "common/runner.h"
#include "common/utils.h"
#include "libxsmm_dnn.h"

namespace convperf {

class XSMMRunner : public Runner {
public:
  XSMMRunner(const ConvParams &params);
  void setup(const float *a, const float *b, float *c);
  void run(const float *a, const float *b, float *c);
  void getResults(float *c);
  ~XSMMRunner();

private:
  libxsmm_dnn_conv_config cfg;
  float *input_save, *output_save, *filter_save;
  float *input_nchw, *output_nchw, *filter_kcrs;
  float *input_libxsmm, *output_libxsmm, *filter_libxsmm;
  const ConvParams &params;
  Shape4D paddedInputShape, paddedOutputShape;
};

}
