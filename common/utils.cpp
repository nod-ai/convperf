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
  ss << N << 'x' << C << 'x' << H << 'x' << W;
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
    std::replace(outputShape.begin(), outputShape.end(), 'x', ' ');
    std::stringstream os(outputShape);
    os >> param.outputShape.N >> param.outputShape.C >>
          param.outputShape.H >> param.outputShape.W;
    std::replace(filterShape.begin(), filterShape.end(), 'x', ' ');
    std::stringstream fs(filterShape);
    fs >> param.filterShape.N >> param.filterShape.C >>
          param.filterShape.H >> param.filterShape.W;
    params.push_back(param);
  }
  return params;
}

void init_random_tensor4d(float *tensor, Shape4D shape) {
  for (int i = 0; i < shape.N; i++) {
    for (int j = 0; j < shape.C; j++) {
      for (int k = 0; k < shape.H; k++) {
        for (int l = 0; l < shape.W; l++) {
          GET_ELEMENT(tensor, i, j, k, l, shape.C, shape.H, shape.W)
            = ((float) rand() / (float) RAND_MAX);
        }
      }
    }
  }
}

void write_tensor4d_to_file(const float *tensor, Shape4D shape, std::string filename) {
  std::ofstream outputFile(filename, std::ios::out);
  std::stringstream ss;
  outputFile << shape.N << "," << shape.C << "," << shape.H << ","
             << shape.W << "\n";
  for (int i = 0; i < shape.N; i++) {
    for (int j = 0; j < shape.C; j++) {
      for (int k = 0; k < shape.H; k++) {
        for (int l = 0; l < shape.W; l++) {
          ss << GET_ELEMENT(tensor, i, j, k, l, shape.C, shape.H, shape.W) << ",";
        }
      }
    }
  }
  std::string out = ss.str();
  out.pop_back();
  outputFile << out;
}

}
