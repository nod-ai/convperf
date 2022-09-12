#include "utils.h"
#include <sstream>
#include <fstream>
#include <stdlib.h>
#include <iostream>

namespace convperf {

int Shape4D::getLinearizedShape() const {
  return N * H * W * C;
}

std::string Shape4D::str() const {
  std::stringstream ss;
  if ((format == "nchw") || (format == "fchw")) {
    ss << N << 'x' << C << 'x' << H << 'x' << W;
  } else if (format == "hwcf") {
    ss << H << 'x' << W << 'x' << C << 'x' << N;
  } else {
    ss << N << 'x' << H << 'x' << W << 'x' << C;
  }
  return ss.str();
}

void ConvParams::computeOutputShape() {
  outputShape.N = inputShape.N;
  outputShape.C = filterShape.N;
  outputShape.H = (inputShape.H + 2 * padding.H - filterShape.H) / strides.H + 1;
  outputShape.W = (inputShape.W + 2 * padding.W - filterShape.W) / strides.W + 1;
}

Shape4D ConvParams::computePaddedShape(const Shape4D &shape) const {
  Shape4D paddedShape = shape;
  paddedShape.H = shape.H + 2 * padding.H;
  paddedShape.W = shape.W + 2 * padding.W;
  return paddedShape;
}

std::vector<ConvParams> ParamFileReader::readParams(const std::string &filename) {
  std::ifstream sizesFile(filename, std::ios::in);
  std::vector<ConvParams> params;
  while (!sizesFile.eof()) {
    std::string line;
    std::getline(sizesFile, line);
    if (line.empty()) break;
    if (line[0] == '#') continue;
    std::replace(line.begin(), line.end(), ',', ' ');
    std::stringstream linestream(line);
    std::string inputShape, filterShape, outputShape;
    ConvParams param;
    linestream >> inputShape >> filterShape >> outputShape
               >> param.strides.H >> param.strides.W
               >> param.padding.H >> param.padding.W
               >> param.dilations.H >> param.dilations.W;
    std::replace(inputShape.begin(), inputShape.end(), 'x', ' ');
    std::stringstream is(inputShape);
    is >> param.inputShape.N >> param.inputShape.C >>
          param.inputShape.H >> param.inputShape.W;
    param.inputShape.format = STR(INPUT_FORMAT);
    std::replace(outputShape.begin(), outputShape.end(), 'x', ' ');
    std::stringstream os(outputShape);
    os >> param.outputShape.N >> param.outputShape.C >>
          param.outputShape.H >> param.outputShape.W;
    param.outputShape.format = STR(INPUT_FORMAT);
    std::replace(filterShape.begin(), filterShape.end(), 'x', ' ');
    std::stringstream fs(filterShape);
    fs >> param.filterShape.N >> param.filterShape.C >>
          param.filterShape.H >> param.filterShape.W;
    param.filterShape.format = STR(FILTER_FORMAT);
    params.push_back(param);
  }
  return params;
}

void init_random_tensor(float *tensor, size_t shape) {
  for (size_t i = 0; i < shape; i++) {
    tensor[i] = ((float) rand() / (float) RAND_MAX);
  }
}

void write_tensor4d_to_file(const float *tensor, Shape4D shape, std::string filename) {
  std::ofstream outputFile(filename, std::ios::out);
  std::stringstream ss;
  outputFile << shape.str() <<"," << shape.format << "\n";
  for (int i = 0; i < shape[0]; i++) {
    for (int j = 0; j < shape[1]; j++) {
      for (int k = 0; k < shape[2]; k++) {
        for (int l = 0; l < shape[3]; l++) {
          ss << GET_ELEMENT(tensor, i, j, k, l, shape[1], shape[2], shape[3]) << ",";
        }
      }
    }
  }
  std::string out = ss.str();
  out.pop_back();
  outputFile << out;
}

float checkTensorsForEquality(float *a, float *b, size_t shape) {
  float maxError{0};
  for (int i = 0; i < shape; i++) {
    float error = std::abs(a[i] - b[i]);
    if (error > maxError) {
      maxError = error;
    }
  }
  return maxError;
}

void convert_nhwc_to_nchw(const float *src, float *dst, Shape4D shape) {
  for(int i = 0; i < shape.N; i++) {
    for(int j = 0; j < shape.C; j++) {
      for(int k = 0; k < shape.H; k++) {
        for(int l = 0; l < shape.W; l++) {
          GET_ELEMENT(dst, i, j, k, l, shape.C, shape.H, shape.W) =
          GET_ELEMENT(src, i, k, l, j, shape.H, shape.W, shape.C);
        }
      }
    }
  }
}

void convert_nchw_to_nhwc(const float *src, float *dst, Shape4D shape) {
  for(int i = 0; i < shape.N; i++) {
    for(int j = 0; j < shape.H; j++) {
      for(int k = 0; k < shape.W; k++) {
        for(int l = 0; l < shape.C; l++) {
          GET_ELEMENT(dst, i, j, k, l, shape.H, shape.W, shape.C) =
          GET_ELEMENT(src, i, l, j, k, shape.C, shape.H, shape.W);
        }
      }
    }
  }
}

void convert_hwcf_to_fchw(const float *src, float *dst, Shape4D shape) {
  int F = shape.N;
  for(int i = 0; i < F; i++) {
    for(int j = 0; j < shape.C; j++) {
      for(int k = 0; k < shape.H; k++) {
        for(int l = 0; l < shape.W; l++) {
          GET_ELEMENT(dst, i, j, k, l, shape.C, shape.H, shape.W) =
          GET_ELEMENT(src, k, l, j, i, shape.W, shape.C, F);
        }
      }
    }
  }

}

void copy_nchw_with_pad(const float *src, float *dst, Shape4D shape, Shape2D padding) {
  for(int i = 0; i < shape.N; i++) {
    for(int j = 0; j < shape.C; j++) {
      for(int k = 0; k < shape.H; k++) {
        for(int l = 0; l < shape.W; l++) {
          GET_ELEMENT(dst, i, j, (k + padding.H), (l + padding.W), shape.C, (shape.H + 2 * padding.H), (shape.W + 2 * padding.W)) =
          GET_ELEMENT(src, i, j, k, l, shape.C, shape.H, shape.W);
        }
      }
    }
  }
}

void copy_nchw_without_pad(const float *src, float *dst, Shape4D shape, Shape2D padding) {
  for(int i = 0; i < shape.N; i++) {
    for(int j = 0; j < shape.C; j++) {
      for(int k = 0; k < shape.H; k++) {
        for(int l = 0; l < shape.W; l++) {
          GET_ELEMENT(dst, i, j, k, l, shape.C, shape.H, shape.W) =
          GET_ELEMENT(src, i, j, (k + padding.H), (l + padding.W),
                      shape.C, (shape.H + 2 * padding.H), (shape.W + 2 * padding.W));
        }
      }
    }
  }
}

void copy_nhwc_without_pad(const float *src, float *dst, Shape4D shape, Shape2D padding) {
  for(int i = 0; i < shape.N; i++) {
    for(int j = 0; j < shape.H; j++) {
      for(int k = 0; k < shape.W; k++) {
        for(int l = 0; l < shape.C; l++) {
          GET_ELEMENT(dst, i, j, k, l, shape.H, shape.W, shape.C) =
          GET_ELEMENT(src, i, j, (k + padding.H), (l + padding.W),
                      (shape.H + 2 * padding.H), (shape.W + 2 * padding.W), shape.C);
        }
      }
    }
  }
}

}
