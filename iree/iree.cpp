#include "iree/iree.h"

namespace convperf {

IREERunner::IREERunner(const ConvParams &params) : params(params) {
}

void IREERunner::run(const float *input, const float *filter, float *output) {
}

IREERunner::~IREERunner() {
}

}
