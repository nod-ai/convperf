#pragma once

#include <vector>
#include <string>

#define GET_ELEMENT(x, a, b, c, d, sb, sc, sd)      \
  *(x + d + sd * (c + sc * (b + sb * a)))

namespace convperf {

struct Shape4D {
  int N, H, W, C;
  int getLinearizedShape() const;
  std::string str() const;
};

struct Shape2D {
  int H, W;
};

struct ConvParams {
  Shape4D inputShape;
  Shape4D outputShape;
  Shape4D filterShape;
  Shape2D padding;
  Shape2D strides;
  Shape2D dilations;
  void computeOutputShape();
  Shape4D computePaddedShape(const Shape4D &shape) const;
};

struct ParamFileReader {
  std::vector<ConvParams> readParams(const std::string &filename);
};

void init_random_tensor4d(float *tensor, Shape4D shape);
void write_tensor4d_to_file(const float *tensor, Shape4D shape, std::string filename);
float checkTensorsForEquality(float *a, float *b, Shape4D shape);

}
