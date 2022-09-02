#pragma once

#include "common/runner.h"
#include "common/utils.h"

namespace convperf {

class IREERunner : public Runner {
public:
  IREERunner(const ConvParams &params);
  void run(const float *a, const float *b, float *c);
  ~IREERunner();

private:
  const ConvParams &params;
};

}
