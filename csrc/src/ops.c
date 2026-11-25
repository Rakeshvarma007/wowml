/**
 * ops.c — Element-wise operations, reductions, im2col/col2im, and fused kernels stubs.
 */
#include "wowml/ops.h"
#include <math.h>

wowml_status_t wowml_add(wowml_tensor_t* out, const wowml_tensor_t* a, const wowml_tensor_t* b) {
    (void)out; (void)a; (void)b;
    /* TODO: Broadcasted add */
    return WOWML_ERR_NOT_IMPLEMENTED;
}

wowml_status_t wowml_sub(wowml_tensor_t* out, const wowml_tensor_t* a, const wowml_tensor_t* b) {
    (void)out; (void)a; (void)b;
    /* TODO: Broadcasted sub */
    return WOWML_ERR_NOT_IMPLEMENTED;
}

wowml_status_t wowml_mul(wowml_tensor_t* out, const wowml_tensor_t* a, const wowml_tensor_t* b) {
    (void)out; (void)a; (void)b;
    /* TODO: Broadcasted mul */
    return WOWML_ERR_NOT_IMPLEMENTED;
}

wowml_status_t wowml_div(wowml_tensor_t* out, const wowml_tensor_t* a, const wowml_tensor_t* b) {
    (void)out; (void)a; (void)b;
    /* TODO: Broadcasted div */
    return WOWML_ERR_NOT_IMPLEMENTED;
}

wowml_status_t wowml_add_scalar(wowml_tensor_t* out, const wowml_tensor_t* a, double scalar) {
    (void)out; (void)a; (void)scalar;
    /* TODO: Scalar add */
    return WOWML_ERR_NOT_IMPLEMENTED;
}

wowml_status_t wowml_mul_scalar(wowml_tensor_t* out, const wowml_tensor_t* a, double scalar) {
    (void)out; (void)a; (void)scalar;
    /* TODO: Scalar mul */
    return WOWML_ERR_NOT_IMPLEMENTED;
}

wowml_status_t wowml_neg(wowml_tensor_t* out, const wowml_tensor_t* a) {
    (void)out; (void)a;
    /* TODO: Element-wise negation */
    return WOWML_ERR_NOT_IMPLEMENTED;
}

wowml_status_t wowml_exp(wowml_tensor_t* out, const wowml_tensor_t* a) {
    (void)out; (void)a;
    /* TODO: Element-wise exp */
    return WOWML_ERR_NOT_IMPLEMENTED;
}

wowml_status_t wowml_log(wowml_tensor_t* out, const wowml_tensor_t* a) {
    (void)out; (void)a;
    /* TODO: Element-wise log */
    return WOWML_ERR_NOT_IMPLEMENTED;
}

wowml_status_t wowml_relu(wowml_tensor_t* out, const wowml_tensor_t* a) {
    (void)out; (void)a;
    /* TODO: Element-wise ReLU */
    return WOWML_ERR_NOT_IMPLEMENTED;
}

wowml_status_t wowml_sigmoid(wowml_tensor_t* out, const wowml_tensor_t* a) {
    (void)out; (void)a;
    /* TODO: Element-wise sigmoid */
    return WOWML_ERR_NOT_IMPLEMENTED;
}

wowml_status_t wowml_tanh_op(wowml_tensor_t* out, const wowml_tensor_t* a) {
    (void)out; (void)a;
    /* TODO: Element-wise tanh */
    return WOWML_ERR_NOT_IMPLEMENTED;
}

wowml_status_t wowml_sqrt(wowml_tensor_t* out, const wowml_tensor_t* a) {
    (void)out; (void)a;
    /* TODO: Element-wise sqrt */
    return WOWML_ERR_NOT_IMPLEMENTED;
}

wowml_status_t wowml_pow_op(wowml_tensor_t* out, const wowml_tensor_t* a, double exponent) {
    (void)out; (void)a; (void)exponent;
    /* TODO: Element-wise pow */
    return WOWML_ERR_NOT_IMPLEMENTED;
}

wowml_status_t wowml_sum(wowml_tensor_t* out, const wowml_tensor_t* a, int axis, int keepdims) {
    (void)out; (void)a; (void)axis; (void)keepdims;
    /* TODO: Sum reduction */
    return WOWML_ERR_NOT_IMPLEMENTED;
}

wowml_status_t wowml_mean(wowml_tensor_t* out, const wowml_tensor_t* a, int axis, int keepdims) {
    (void)out; (void)a; (void)axis; (void)keepdims;
    /* TODO: Mean reduction */
    return WOWML_ERR_NOT_IMPLEMENTED;
}

wowml_status_t wowml_max(wowml_tensor_t* out, const wowml_tensor_t* a, int axis, int keepdims) {
    (void)out; (void)a; (void)axis; (void)keepdims;
    /* TODO: Max reduction */
    return WOWML_ERR_NOT_IMPLEMENTED;
}

wowml_status_t wowml_min(wowml_tensor_t* out, const wowml_tensor_t* a, int axis, int keepdims) {
    (void)out; (void)a; (void)axis; (void)keepdims;
    /* TODO: Min reduction */
    return WOWML_ERR_NOT_IMPLEMENTED;
}

wowml_status_t wowml_equal(wowml_tensor_t* out, const wowml_tensor_t* a, const wowml_tensor_t* b) {
    (void)out; (void)a; (void)b;
    /* TODO: Equality comparison */
    return WOWML_ERR_NOT_IMPLEMENTED;
}

wowml_status_t wowml_im2col(wowml_tensor_t* col, const wowml_tensor_t* im,
                            int kernel_h, int kernel_w, int pad_h, int pad_w,
                            int stride_h, int stride_w, int dilation_h, int dilation_w) {
    (void)col; (void)im; (void)kernel_h; (void)kernel_w; (void)pad_h; (void)pad_w;
    (void)stride_h; (void)stride_w; (void)dilation_h; (void)dilation_w;
    /* TODO: im2col transformation */
    return WOWML_ERR_NOT_IMPLEMENTED;
}

wowml_status_t wowml_col2im(wowml_tensor_t* im, const wowml_tensor_t* col,
                            int kernel_h, int kernel_w, int pad_h, int pad_w,
                            int stride_h, int stride_w, int dilation_h, int dilation_w) {
    (void)im; (void)col; (void)kernel_h; (void)kernel_w; (void)pad_h; (void)pad_w;
    (void)stride_h; (void)stride_w; (void)dilation_h; (void)dilation_w;
    /* TODO: col2im gradient transformation */
    return WOWML_ERR_NOT_IMPLEMENTED;
}

wowml_status_t wowml_fused_linear_bias_relu(wowml_tensor_t* out, const wowml_tensor_t* x,
                                           const wowml_tensor_t* weight, const wowml_tensor_t* bias) {
    (void)out; (void)x; (void)weight; (void)bias;
    /* TODO: Single-pass fused (x @ W^T + b) + ReLU */
    return WOWML_ERR_NOT_IMPLEMENTED;
}

wowml_status_t wowml_fused_cross_entropy(double* loss, wowml_tensor_t* dlogits,
                                        const wowml_tensor_t* logits, const wowml_tensor_t* targets) {
    (void)loss; (void)dlogits; (void)logits; (void)targets;
    /* TODO: Numerically stable fused log-softmax + NLL loss & gradient */
    return WOWML_ERR_NOT_IMPLEMENTED;
}
