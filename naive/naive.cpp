#include "naive/naive.h"

namespace convperf {

NaiveRunner::NaiveRunner(const ConvParams &params) : params(params) {
  paddedInputShape = params.computePaddedShape(params.inputShape);
  paddedOutputShape = params.computePaddedShape(params.outputShape);
}

void NaiveRunner::run(const float *input, const float *filter, float *output) {
  int ij, ii;
  for (int b = 0; b < params.inputShape.N; b++) {
    for (int ofm = 0; ofm < params.filterShape.N; ofm++) {
      for (int ifm = 0; ifm < params.filterShape.C; ifm++) {
        for (int ofh = 0; ofh < params.outputShape.H; ofh++) {
          ij = ofh * params.strides.H - params.padding.H;
          for (int ofw = 0; ofw < params.outputShape.W; ofw++) {
            ii = ofw * params.strides.W - params.padding.W;
            for (int kh = 0; kh < params.filterShape.H; kh++) {
              if (ij + kh < 0 || ij + kh >= params.inputShape.H) continue;
              for (int kw = 0; kw < params.filterShape.W; kw++) {
                if (ii + kw < 0 || ii + kw >= params.inputShape.W) continue;
                GET_ELEMENT(output, b, ofm, ofh, ofw,
                            paddedOutputShape.C, paddedOutputShape.H, paddedOutputShape.W)
                += GET_ELEMENT(input, b, ifm, ij + kh, ii + kw,
                               paddedInputShape.C, paddedInputShape.H, paddedInputShape.W)
                * GET_ELEMENT(filter, ofm, ifm, kh, kw,
                              params.filterShape.C, params.filterShape.H, params.filterShape.W);
              }
            }
          }
        }
      }
    }
  }
}

NaiveRunner::~NaiveRunner() {
}

}
