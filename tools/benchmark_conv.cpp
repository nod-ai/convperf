#include "iree/iree.h"
#include "iree/base/internal/flags.h"
#include "naive/naive.h"
#include "xsmm/xsmm.h"
#include "common/utils.h"
#include "benchmark/benchmark.h"
#include <iostream>

static void BenchmarkFunction(benchmark::State &state, const convperf::ConvParams &param) {
  size_t alignment{16};
  bool verify{true};
  float *input, *filter, *output;
  float tol{5e-4};
  input = static_cast<float *>(std::aligned_alloc(alignment, param.inputShape.getLinearizedShape() * sizeof(float)));
  filter = static_cast<float *>(std::aligned_alloc(alignment, param.filterShape.getLinearizedShape() * sizeof(float)));
  output = static_cast<float *>(std::aligned_alloc(alignment, param.outputShape.getLinearizedShape() * sizeof(float)));
  init_random_tensor4d(input, param.inputShape);
  init_random_tensor4d(filter, param.filterShape);
  auto runner = convperf::IREERunner(param);
  runner.setup(input, filter, output);
  for (auto _ : state) {
    runner.run(input, filter, output);
  }
  if (verify) {
    runner.getResults(output);
    auto verifier = convperf::NaiveRunner(param);
    float *golden = static_cast<float *>(std::aligned_alloc(alignment, param.outputShape.getLinearizedShape() * sizeof(float)));
    verifier.setup(input, filter, golden);
    verifier.run(input, filter, golden);
    verifier.getResults(golden);
    float error = convperf::checkTensorsForEquality(golden, output, param.outputShape);
    if (error > tol) {
      printf("Accuracy verification failed [%f] > [%f]\n", error, tol);
    }
  }
  free(input);
  free(filter);
  free(output);
}

int main(int argc, char *argv[]) {
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_UNDEFINED_OK |
                           IREE_FLAGS_PARSE_MODE_CONTINUE_AFTER_HELP,
                           &argc, &argv);
  convperf::ParamFileReader reader;
  auto params = reader.readParams("benchmark_sizes/resnet50.txt");
  for (const auto &param : params) {
    std::string msg = "Benchmarking => Input : [" + param.inputShape.str() + "], Filter : [" + param.filterShape.str() + "], Output : ["
                    + param.outputShape.str() + "]\n";
    std::cout << msg;
    ::benchmark::RegisterBenchmark("Conv", BenchmarkFunction, param)
                                 ->MeasureProcessCPUTime()->UseRealTime();
                                 //->Iterations(NUM_REPS); //A fixed number of iterations can be set by uncomment this
  }
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
  return 0;
}
