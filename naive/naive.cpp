#include "naive/naive.h"

namespace convperf {

NaiveRunner::NaiveRunner(const ConvParams &params) : params(params) {
  paddedInputShape = params.computePaddedShape(params.inputShape);
  paddedOutputShape = params.computePaddedShape(params.outputShape);
}

void NaiveRunner::setup(const float *input, const float *filter, float *output) {
  if (params.inputShape.format == "nhwc") {
    input_nchw = (float *) malloc(params.inputShape.getLinearizedShape() * sizeof(float));
    output_nchw = (float *) malloc(params.outputShape.getLinearizedShape() * sizeof(float));
    convert_nhwc_to_nchw(input, input_nchw, params.inputShape);
  } else {
    input_nchw = (float *)input;
    output_nchw = output;
  }

  if (params.filterShape.format == "hwcf") {
    filter_fchw = (float *) malloc(params.filterShape.getLinearizedShape() * sizeof(float));
    convert_hwcf_to_fchw(filter, filter_fchw, params.filterShape);
  } else {
    filter_fchw = (float *)filter;
  }
}

void NaiveRunner::run(const float *input, const float *filter, float *output) {
  int ij, ii;

  for (int b = 0; b < params.inputShape.N; b++) {
    for (int ofm = 0; ofm < params.outputShape.C; ofm++) {
      for (int ofh = 0; ofh < params.outputShape.H; ofh++) {
        for (int ofw = 0; ofw < params.outputShape.W; ofw++) {
          GET_ELEMENT(output_nchw, b, ofm, ofh, ofw,
                      paddedOutputShape.C, paddedOutputShape.H, paddedOutputShape.W) = 0;
        }
      }
    }
  }

  for (int b = 0; b < params.inputShape.N; b++) {
    for (int ofm = 0; ofm < params.outputShape.C; ofm++) {
      for (int ifm = 0; ifm < params.inputShape.C; ifm++) {
        for (int ofh = 0; ofh < params.outputShape.H; ofh++) {
          ij = ofh * params.strides.H - params.padding.H;
          for (int ofw = 0; ofw < params.outputShape.W; ofw++) {
            ii = ofw * params.strides.W - params.padding.W;
            for (int kh = 0; kh < params.filterShape.H; kh++) {
              if (ij + kh < 0 || ij + kh >= params.inputShape.H) continue;
              for (int kw = 0; kw < params.filterShape.W; kw++) {
                if (ii + kw < 0 || ii + kw >= params.inputShape.W) continue;
                GET_ELEMENT(output_nchw, b, ofm, ofh, ofw,
                            params.outputShape.C, paddedOutputShape.H, paddedOutputShape.W)
                += GET_ELEMENT(input_nchw, b, ifm, ij + kh, ii + kw,
                               params.inputShape.C, paddedInputShape.H, paddedInputShape.W)
                * GET_ELEMENT(filter_fchw, ofm, ifm, kh, kw,
                              params.filterShape.C, params.filterShape.H, params.filterShape.W);
              }
            }
          }
        }
      }
    }
  }
}

void NaiveRunner::getResults(float *output) {
  if (params.outputShape.format == "nhwc") {
    convert_nchw_to_nhwc(output_nchw, output, params.outputShape);
  }
}

NaiveRunner::~NaiveRunner() {
  if (params.inputShape.format == "nhwc") {
    free(input_nchw);
    free(output_nchw);
  }
  if (params.outputShape.format == "hwcf") {
    free(filter_fchw);
  }
}

}
