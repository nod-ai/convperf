#pragma once

#include "common/runner.h"
#include "common/utils.h"

namespace convperf {

class NaiveRunner : public Runner {
public:
  NaiveRunner(const ConvParams &params);
  void run(const float *a, const float *b, float *c);
  ~NaiveRunner();

private:
  const ConvParams &params;
  Shape4D paddedInputShape, paddedOutputShape;
};

}
