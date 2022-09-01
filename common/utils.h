#pragma once

#include <vector>
#include <string>

#define GET_ELEMENT(x, a, b, c, d, sb, sc, sd)      \
  *(x + d + c * sd + b * sc * sd + a * sb * sc * sd)

namespace convperf {

struct Shape4D {
  int N, H, W, C;
  int getLinearizedShape() const;
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

void init_random_tensor4d(float *tensor, const Shape4D shape);

}
