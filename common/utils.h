#pragma once

#include <vector>
#include <string>

#define STRINGIFY(x) #x
#define STR(x) STRINGIFY(x)

#define GET_ELEMENT(x, a, b, c, d, sb, sc, sd)      \
  *(x + d + sd * (c + sc * (b + sb * a)))

namespace convperf {

/*
 * We use this struct to store the shapes of
 * the input, output and filter tensors.
 * For the filter,
 * N -> output channels (F)
 * C -> input channels
 * H -> kernel height
 * W -> kernel width
 *
 * The default format is NCHW.
 */
struct Shape4D {
  int N, H, W, C;
  int getLinearizedShape() const;
  std::string str() const;
  std::string format{"nchw"};
  int operator [](int i) {
    if (format == "nchw") {
      switch (i) {
        case 0: return N;
        case 1: return C;
        case 2: return H;
        case 3: return W;
        default: return -1;
      }
    } else {
      switch (i) {
        case 0: return N;
        case 1: return H;
        case 2: return W;
        case 3: return C;
        default: return -1;
      }
    }
  }
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

void convert_nhwc_to_nchw(const float *src, float *dst, Shape4D shape);
void convert_nchw_to_nhwc(const float *src, float *dst, Shape4D shape);
void convert_hwcf_to_fchw(const float *src, float *dst, Shape4D shape);

void copy_nchw_with_pad(const float *src, float *dst, Shape4D shape, Shape2D padding);
void copy_nchw_without_pad(const float *src, float *dst, Shape4D shape, Shape2D padding);
void copy_nhwc_without_pad(const float *src, float *dst, Shape4D shape, Shape2D padding);

}
