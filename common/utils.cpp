#include "utils.h"
#include <sstream>
#include <fstream>
#include <stdlib.h>
#include <iostream>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

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
  json data = json::parse(sizesFile);
  std::vector<ConvParams> params;
  for (auto &config : data["configs"]) {
    ConvParams param;
    param.inputShape.N = config["input"]["N"].get<int>();
    param.inputShape.C = config["input"]["C"].get<int>();
    param.inputShape.H = config["input"]["H"].get<int>();
    param.inputShape.W = config["input"]["W"].get<int>();
    param.inputShape.format = config["input"]["format"].get<std::string>();
    param.outputShape.N = config["output"]["N"].get<int>();
    param.outputShape.C = config["output"]["C"].get<int>();
    param.outputShape.H = config["output"]["H"].get<int>();
    param.outputShape.W = config["output"]["W"].get<int>();
    param.outputShape.format = config["output"]["format"].get<std::string>();
    param.filterShape.N = config["filter"]["F"].get<int>();
    param.filterShape.C = config["filter"]["C"].get<int>();
    param.filterShape.H = config["filter"]["H"].get<int>();
    param.filterShape.W = config["filter"]["W"].get<int>();
    param.filterShape.format = config["filter"]["format"].get<std::string>();
    param.strides.H = config["strides"][0].get<int>();
    param.strides.W = config["strides"][1].get<int>();
    param.padding.H = config["padding"][0].get<int>();
    param.padding.W = config["padding"][1].get<int>();
    param.dilations.H = config["dilations"][0].get<int>();
    param.dilations.W = config["dilations"][1].get<int>();
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
