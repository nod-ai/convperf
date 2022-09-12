#include "naive/naive.h"
#include "xsmm/xsmm.h"
#include "common/utils.h"
#include <iostream>

int main(int argc, char *argv[]) {
  convperf::ParamFileReader reader;
  auto params = reader.readParams("benchmark_sizes/resnet50.txt");
  size_t alignment{16};
  for (const auto &param : params) {
    std::string msg = "Verifying => Input : [" + param.inputShape.str() + "], Filter : [" + param.filterShape.str() + "], Output : ["
                    + param.outputShape.str() + "]\n";
    std::cout << msg;
    auto runner = convperf::NaiveRunner(param);
    size_t linearizedInputShape = param.inputShape.getLinearizedShape();
    float *input = static_cast<float *>(std::aligned_alloc(alignment, linearizedInputShape * sizeof(float)));
    size_t linearizedFilterShape = param.filterShape.getLinearizedShape();
    float *filter = static_cast<float *>(std::aligned_alloc(alignment, linearizedFilterShape * sizeof(float)));
    size_t linearizedOutputShape = param.outputShape.getLinearizedShape();
    float *output = static_cast<float *>(std::aligned_alloc(alignment, linearizedOutputShape * sizeof(float)));
    convperf::init_random_tensor(input, linearizedInputShape);
    convperf::init_random_tensor(filter, linearizedFilterShape);
    write_tensor4d_to_file(input, param.inputShape, "input.csv");
    write_tensor4d_to_file(filter, param.filterShape, "filter.csv");
    runner.setup(input, filter, output);
    runner.run(input, filter, output);
    runner.getResults(output);
    write_tensor4d_to_file(output, param.outputShape, "output.csv");
    free(input);
    free(filter);
    free(output);
    break;
  }
  return 0;
}
