#include "xsmm/xsmm.h"
#include "dnn_common.h"
#if defined(_OPENMP)
# include <omp.h>
#endif

/*
 * References:
 * https://github.com/libxsmm/libxsmm/blob/main/
 * samples/deeplearning/libxsmm_dnn/tests/conv/layer_example.c
 *
 */

namespace convperf {

XSMMRunner::XSMMRunner(const ConvParams &params) : params(params) {
  libxsmm_datatype cnn_dtype = LIBXSMM_DATATYPE_F32;
  libxsmm_dnn_conv_eltwise_fuse my_fuse = LIBXSMM_DNN_CONV_ELTWISE_FUSE_NONE;
  int overwrite_output = 1;
  int avoid_bwd_wt_trans = 0;
  int zero_output_rims_fwd = 0;
  int bc = 64, bk = 64;
#if defined(_OPENMP)
  int nThreads = omp_get_max_threads(); /* number of threads */
#else
  int nThreads = 1; /* number of threads */
#endif

  cfg = setup_libxsmm_dnn_conv(
    cnn_dtype, cnn_dtype, params.inputShape.N, params.inputShape.H, params.inputShape.W,
    params.inputShape.C, params.filterShape.N, params.filterShape.H, params.filterShape.W,
    params.strides.H, params.strides.W, params.padding.H, params.padding.W, params.padding.H,
    params.padding.W, params.padding.H, params.padding.W, bc, bk, nThreads,
    my_fuse, overwrite_output, avoid_bwd_wt_trans, zero_output_rims_fwd);

  input_save = (float*)libxsmm_aligned_malloc((size_t) params.inputShape.getLinearizedShape() *sizeof(float), 2097152);
  output_save = (float*)libxsmm_aligned_malloc((size_t) params.outputShape.getLinearizedShape() *sizeof(float), 2097152);
  filter_save = (float*)libxsmm_aligned_malloc((size_t) params.filterShape.getLinearizedShape() *sizeof(float), 2097152);
  input_libxsmm = (float*)libxsmm_aligned_malloc((size_t) params.inputShape.getLinearizedShape() *sizeof(float), 2097152);
  output_libxsmm = (float*)libxsmm_aligned_malloc((size_t) params.outputShape.getLinearizedShape() *sizeof(float), 2097152);
  filter_libxsmm = (float*)libxsmm_aligned_malloc((size_t) params.filterShape.getLinearizedShape() *sizeof(float), 2097152);
}


void XSMMRunner::setup(const float *input, const float *filter, float *output) {

  if (params.inputShape.format == "nhwc") {
    input_nchw = (float*)libxsmm_aligned_malloc((size_t) params.inputShape.getLinearizedShape() *sizeof(float), 2097152);
    output_nchw = (float*)libxsmm_aligned_malloc((size_t) params.outputShape.getLinearizedShape() *sizeof(float), 2097152);
    convert_nhwc_to_nchw(input, input_nchw, params.inputShape);
  } else {
    input_nchw = (float *) input;
    output_nchw = output;
  }

  if (params.filterShape.format == "hwcf") {
    filter_kcrs = (float*)libxsmm_aligned_malloc((size_t) params.filterShape.getLinearizedShape() *sizeof(float), 2097152);
    convert_hwcf_to_fchw(filter, filter_kcrs, params.filterShape);
  } else {
    filter_kcrs = (float *) filter;
  }

  copy_buf((float *)input_nchw, input_save, params.inputShape.getLinearizedShape());
  zero_buf(output_save, params.outputShape.getLinearizedShape());
  copy_buf(output, output_save, params.outputShape.getLinearizedShape());

  /* first touch LIBXSMM */
  zero_buf(input_libxsmm, params.inputShape.getLinearizedShape());
  zero_buf(filter_libxsmm, params.filterShape.getLinearizedShape());
  zero_buf(output_libxsmm, params.outputShape.getLinearizedShape());

  /* Copy input/output/weight tensors to correct format */
  tensor_copy_NCHW_to_NCHWc(input_save, input_libxsmm,
                            params.inputShape.N,
                            params.inputShape.C,
                            params.inputShape.H,
                            params.inputShape.W,
                            cfg.ifmblock);
  tensor_copy_NCHW_to_NCHWc(output_save, output_libxsmm,
                            params.outputShape.N,
                            params.outputShape.C,
                            params.outputShape.H,
                            params.outputShape.W,
                            cfg.ofmblock);
  tensor_copy_KCRS_to_KCRSck((float *)filter_kcrs, filter_libxsmm,
                             params.filterShape.N,
                             params.filterShape.C,
                             params.filterShape.H,
                             params.filterShape.W,
                             cfg.ifmblock,
                             cfg.ofmblock);
}

void XSMMRunner::run(const float *input, const float *filter, float *output) {
  /* run LIBXSMM convolutions */
#if defined(_OPENMP)
#pragma omp parallel
#endif
  {
#if defined(_OPENMP)
    const int tid = omp_get_thread_num();
#else
    const int tid = 0;
#endif
    libxsmm_dnn_conv_fwd_exec(cfg, filter_libxsmm, input_libxsmm, output_libxsmm,
                              nullptr, nullptr, 0, tid, nullptr);
  }
}

void XSMMRunner::getResults(float *output) {

  /* copy out data */
  tensor_copy_NCHWc_to_NCHW(output_libxsmm, output_nchw,
                            params.outputShape.N,
                            params.outputShape.C,
                            params.outputShape.H,
                            params.outputShape.W,
                            cfg.ofmblock);

  if (params.outputShape.format == "nhwc") {
    convert_nchw_to_nhwc(output_nchw, output, params.outputShape);
  }

}

XSMMRunner::~XSMMRunner() {
  if (params.inputShape.format == "nhwc") {
    libxsmm_free(input_nchw);
    libxsmm_free(output_nchw);
  }
  if (params.filterShape.format == "hwcf") {
    libxsmm_free(filter_kcrs);
  }
  libxsmm_free(input_save);
  libxsmm_free(output_save);
  libxsmm_free(filter_save);
  libxsmm_free(input_libxsmm);
  libxsmm_free(output_libxsmm);
  libxsmm_free(filter_libxsmm);
  destroy_libxsmm_dnn_conv(&cfg);
}

}
