/**
 * ops.h — Element-wise operations, reductions, im2col/col2im, and fused kernels.
 */
#ifndef WOWML_OPS_H
#define WOWML_OPS_H

#include "tensor.h"

/* ---- Element-wise Arithmetic with N-D Broadcasting ---- */
wowml_status_t wowml_add(wowml_tensor_t* out, const wowml_tensor_t* a, const wowml_tensor_t* b);
wowml_status_t wowml_sub(wowml_tensor_t* out, const wowml_tensor_t* a, const wowml_tensor_t* b);
wowml_status_t wowml_mul(wowml_tensor_t* out, const wowml_tensor_t* a, const wowml_tensor_t* b);
wowml_status_t wowml_div(wowml_tensor_t* out, const wowml_tensor_t* a, const wowml_tensor_t* b);

/* ---- Scalar Operations ---- */
wowml_status_t wowml_add_scalar(wowml_tensor_t* out, const wowml_tensor_t* a, double scalar);
wowml_status_t wowml_mul_scalar(wowml_tensor_t* out, const wowml_tensor_t* a, double scalar);

/* ---- Unary Operations ---- */
wowml_status_t wowml_neg(wowml_tensor_t* out, const wowml_tensor_t* a);
wowml_status_t wowml_exp(wowml_tensor_t* out, const wowml_tensor_t* a);
wowml_status_t wowml_log(wowml_tensor_t* out, const wowml_tensor_t* a);
wowml_status_t wowml_relu(wowml_tensor_t* out, const wowml_tensor_t* a);
wowml_status_t wowml_sigmoid(wowml_tensor_t* out, const wowml_tensor_t* a);
wowml_status_t wowml_tanh_op(wowml_tensor_t* out, const wowml_tensor_t* a);
wowml_status_t wowml_sqrt(wowml_tensor_t* out, const wowml_tensor_t* a);
wowml_status_t wowml_pow_op(wowml_tensor_t* out, const wowml_tensor_t* a, double exponent);

/* ---- Reductions ---- */
wowml_status_t wowml_sum(wowml_tensor_t* out, const wowml_tensor_t* a, int axis, int keepdims);
wowml_status_t wowml_mean(wowml_tensor_t* out, const wowml_tensor_t* a, int axis, int keepdims);
wowml_status_t wowml_max(wowml_tensor_t* out, const wowml_tensor_t* a, int axis, int keepdims);
wowml_status_t wowml_min(wowml_tensor_t* out, const wowml_tensor_t* a, int axis, int keepdims);

/* ---- Comparison ---- */
wowml_status_t wowml_equal(wowml_tensor_t* out, const wowml_tensor_t* a, const wowml_tensor_t* b);

/* ---- Convolution Acceleration Helpers ---- */
wowml_status_t wowml_im2col(wowml_tensor_t* col, const wowml_tensor_t* im,
                            int kernel_h, int kernel_w, int pad_h, int pad_w,
                            int stride_h, int stride_w, int dilation_h, int dilation_w);
wowml_status_t wowml_col2im(wowml_tensor_t* im, const wowml_tensor_t* col,
                            int kernel_h, int kernel_w, int pad_h, int pad_w,
                            int stride_h, int stride_w, int dilation_h, int dilation_w);

/* ---- Fused Neural Network Kernels ---- */
wowml_status_t wowml_fused_linear_bias_relu(wowml_tensor_t* out, const wowml_tensor_t* x,
                                           const wowml_tensor_t* weight, const wowml_tensor_t* bias);
wowml_status_t wowml_fused_cross_entropy(double* loss, wowml_tensor_t* dlogits,
                                        const wowml_tensor_t* logits, const wowml_tensor_t* targets);

#endif /* WOWML_OPS_H */
