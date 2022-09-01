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
    float *input = static_cast<float *>(std::aligned_alloc(alignment, param.inputShape.getLinearizedShape() * sizeof(float)));
    float *filter = static_cast<float *>(std::aligned_alloc(alignment, param.filterShape.getLinearizedShape() * sizeof(float)));
    float *output = static_cast<float *>(std::aligned_alloc(alignment, param.outputShape.getLinearizedShape() * sizeof(float)));
    init_random_tensor4d(input, param.inputShape);
    init_random_tensor4d(filter, param.filterShape);
    write_tensor4d_to_file(input, param.inputShape, "input.csv");
    write_tensor4d_to_file(filter, param.filterShape, "filter.csv");
    runner.run(input, filter, output);
    write_tensor4d_to_file(output, param.outputShape, "output.csv");
    free(input);
    free(filter);
    free(output);
    break;
  }
  return 0;
}
