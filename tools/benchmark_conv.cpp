#include "naive/naive.h"
#include "xsmm/xsmm.h"
#include "common/utils.h"

int main(int argc, char *argv[]) {
  convperf::ParamFileReader reader;
  auto params = reader.readParams("benchmark_sizes/resnet50.txt");
  size_t alignment{16};
  for (const auto &param : params) {
    auto verifier = convperf::NaiveRunner(param);
    auto runner = convperf::XSMMRunner(param);
    float *input = static_cast<float *>(std::aligned_alloc(alignment, param.inputShape.getLinearizedShape() * sizeof(float)));
    float *filter = static_cast<float *>(std::aligned_alloc(alignment, param.filterShape.getLinearizedShape() * sizeof(float)));
    float *output = static_cast<float *>(std::aligned_alloc(alignment, param.outputShape.getLinearizedShape() * sizeof(float)));
    init_random_tensor4d(input, param.inputShape);
    init_random_tensor4d(filter, param.filterShape);
    runner.run(input, filter, output);
    free(input);
    free(filter);
    free(output);
  }
  return 0;
}
