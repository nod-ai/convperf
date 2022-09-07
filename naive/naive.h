#pragma once

#include "common/runner.h"
#include "common/utils.h"

namespace convperf {

class NaiveRunner : public Runner {
public:
  NaiveRunner(const ConvParams &params);
  void setup(const float *a, const float *b, float *c);
  void run(const float *a, const float *b, float *c);
  void getResults(float *c);
  ~NaiveRunner();

private:
  const ConvParams &params;
  Shape4D paddedInputShape, paddedOutputShape;
  float *input_nchw, *output_nchw, *filter_fchw;
};

}
