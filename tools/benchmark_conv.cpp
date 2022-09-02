#include "naive/naive.h"
#include "xsmm/xsmm.h"
#include "common/utils.h"
#include <iostream>

int main(int argc, char *argv[]) {
  convperf::ParamFileReader reader;
  auto params = reader.readParams("benchmark_sizes/resnet50.txt");
  size_t alignment{16};
  bool verify{true};
  float *input, *filter, *output;
  float tol{1e-4};
  for (const auto &param : params) {
    std::string msg = "Benchmarking => Input : [" + param.inputShape.str() + "], Filter : [" + param.filterShape.str() + "], Output : ["
                    + param.outputShape.str() + "]\n";
    std::cout << msg;
    auto runner = convperf::XSMMRunner(param);
    input = static_cast<float *>(std::aligned_alloc(alignment, param.inputShape.getLinearizedShape() * sizeof(float)));
    filter = static_cast<float *>(std::aligned_alloc(alignment, param.filterShape.getLinearizedShape() * sizeof(float)));
    output = static_cast<float *>(std::aligned_alloc(alignment, param.outputShape.getLinearizedShape() * sizeof(float)));
    init_random_tensor4d(input, param.inputShape);
    init_random_tensor4d(filter, param.filterShape);
    runner.run(input, filter, output);
    if (verify) {
      auto verifier = convperf::NaiveRunner(param);
      float *golden = static_cast<float *>(std::aligned_alloc(alignment, param.outputShape.getLinearizedShape() * sizeof(float)));
      verifier.run(input, filter, golden);
      float error = convperf::checkTensorsForEquality(golden, output, param.outputShape);
      if (error > tol) {
        printf("Accuracy verification failed [%f] > [%f]\n", error, tol);
      }
    }
    free(input);
    free(filter);
    free(output);
  }
  return 0;
}
