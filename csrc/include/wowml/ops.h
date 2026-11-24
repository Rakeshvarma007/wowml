/**
 * ops.h — Element-wise and reduction operations.
 */
#ifndef WOWML_OPS_H
#define WOWML_OPS_H

#include "tensor.h"

/* Element-wise arithmetic */
wowml_status_t wowml_add(wowml_tensor_t* out, const wowml_tensor_t* a, const wowml_tensor_t* b);
wowml_status_t wowml_sub(wowml_tensor_t* out, const wowml_tensor_t* a, const wowml_tensor_t* b);
wowml_status_t wowml_mul(wowml_tensor_t* out, const wowml_tensor_t* a, const wowml_tensor_t* b);
wowml_status_t wowml_div(wowml_tensor_t* out, const wowml_tensor_t* a, const wowml_tensor_t* b);

/* Scalar operations */
wowml_status_t wowml_add_scalar(wowml_tensor_t* out, const wowml_tensor_t* a, double scalar);
wowml_status_t wowml_mul_scalar(wowml_tensor_t* out, const wowml_tensor_t* a, double scalar);

/* Unary operations */
wowml_status_t wowml_neg(wowml_tensor_t* out, const wowml_tensor_t* a);
wowml_status_t wowml_exp(wowml_tensor_t* out, const wowml_tensor_t* a);
wowml_status_t wowml_log(wowml_tensor_t* out, const wowml_tensor_t* a);
wowml_status_t wowml_relu(wowml_tensor_t* out, const wowml_tensor_t* a);
wowml_status_t wowml_sigmoid(wowml_tensor_t* out, const wowml_tensor_t* a);
wowml_status_t wowml_tanh_op(wowml_tensor_t* out, const wowml_tensor_t* a);
wowml_status_t wowml_sqrt(wowml_tensor_t* out, const wowml_tensor_t* a);
wowml_status_t wowml_pow_op(wowml_tensor_t* out, const wowml_tensor_t* a, double exponent);

/* Reductions */
wowml_status_t wowml_sum(wowml_tensor_t* out, const wowml_tensor_t* a, int axis);
wowml_status_t wowml_mean(wowml_tensor_t* out, const wowml_tensor_t* a, int axis);
wowml_status_t wowml_max(wowml_tensor_t* out, const wowml_tensor_t* a, int axis);
wowml_status_t wowml_min(wowml_tensor_t* out, const wowml_tensor_t* a, int axis);

/* Comparison */
wowml_status_t wowml_equal(wowml_tensor_t* out, const wowml_tensor_t* a, const wowml_tensor_t* b);

#endif /* WOWML_OPS_H */
