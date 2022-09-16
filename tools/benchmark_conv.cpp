#include "iree/iree.h"
#include "iree/base/internal/flags.h"
#include "llvm/Support/CommandLine.h"
#include "naive/naive.h"
#include "xsmm/xsmm.h"
#include "common/utils.h"
#include "benchmark/benchmark.h"
#include <iostream>

using namespace llvm;
cl::opt<std::string> runnerType("r", cl::desc("Specify method to benchmark"),
                                cl::value_desc("runner"), cl::init("iree"));
cl::opt<int> batchSize("batch_size", cl::desc("The number of batch size, which is expected to match"
                       "iree-hal-benchmark-dispatch-repeat-count when translating the module"),
                       cl::value_desc("batch_size"), cl::init(100));

union Runner {
  std::unique_ptr<convperf::IREERunner> iree;
  std::unique_ptr<convperf::XSMMRunner> xsmm;
};

static void BenchmarkFunction(benchmark::State &state, const convperf::ConvParams &param) {
  size_t alignment{16};
  bool verify{true};
  float *input, *filter, *output;
  float tol{5e-4};
  size_t linearizedInputShape = param.inputShape.getLinearizedShape();
  input = static_cast<float *>(std::aligned_alloc(alignment, linearizedInputShape * sizeof(float)));
  size_t linearizedFilterShape = param.filterShape.getLinearizedShape();
  filter = static_cast<float *>(std::aligned_alloc(alignment, linearizedFilterShape * sizeof(float)));
  size_t linearizedOutputShape = param.outputShape.getLinearizedShape();
  output = static_cast<float *>(std::aligned_alloc(alignment, linearizedOutputShape * sizeof(float)));
  convperf::init_random_tensor(input, linearizedInputShape);
  convperf::init_random_tensor(filter, linearizedFilterShape);
  std::unique_ptr<convperf::Runner> runner;
  if (runnerType == "iree") {
    runner = std::make_unique<convperf::IREERunner>(param);
  } else if (runnerType == "xsmm") {
    runner = std::make_unique<convperf::XSMMRunner>(param);
  }
  runner->setup(input, filter, output);
  while (state.KeepRunningBatch(batchSize)) {
    runner->run(input, filter, output);
  }
  state.SetItemsProcessed(state.iterations());
  if (verify) {
    runner->getResults(output);
    auto verifier = convperf::NaiveRunner(param);
    float *golden = static_cast<float *>(std::aligned_alloc(alignment, linearizedOutputShape * sizeof(float)));
    verifier.setup(input, filter, golden);
    verifier.run(input, filter, golden);
    verifier.getResults(golden);
    float error = convperf::checkTensorsForEquality(golden, output, linearizedOutputShape);
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
  cl::ParseCommandLineOptions(argc, argv);
  std::cout << "Benchmarking ..." << runnerType << "\n";
  convperf::ParamFileReader reader;
  auto params = reader.readParams(STR(BENCHMARK_SIZES));
  std::cout << "Using batch size = " << batchSize << "\n";
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
