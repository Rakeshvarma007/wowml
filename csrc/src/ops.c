/**
 * ops.c — Element-wise ops (with N-D broadcasting), reductions, im2col/col2im, and fused kernels.
 */
#include "wowml/ops.h"
#include "wowml/matmul.h"
#include "wowml/memory.h"
#include <math.h>
#include <string.h>
#include <float.h>

static int64_t strided_offset(int64_t flat_idx, int ndim, const int64_t* out_shape, const int64_t* strides) {
    int64_t rem = flat_idx;
    int64_t off = 0;
    for (int d = ndim - 1; d >= 0; d--) {
        int64_t s = out_shape[d];
        int64_t coord = (s > 0) ? (rem % s) : 0;
        rem = (s > 0) ? (rem / s) : 0;
        off += coord * strides[d];
    }
    return off;
}

typedef enum { OP_ADD = 0, OP_SUB, OP_MUL, OP_DIV } bin_op_t;

static wowml_status_t binary_op_impl(wowml_tensor_t* out, const wowml_tensor_t* a,
                                     const wowml_tensor_t* b, bin_op_t op) {
    WOWML_CHECK(out != NULL && a != NULL && b != NULL, WOWML_ERR_NULL_PTR);
    WOWML_CHECK(a->dtype == b->dtype && a->dtype == out->dtype, WOWML_ERR_INVALID_DTYPE);

    /* Fast path: same shape and all contiguous */
    int same_shape = (a->ndim == b->ndim && a->ndim == out->ndim);
    if (same_shape) {
        for (int i = 0; i < a->ndim; i++) {
            if (a->shape[i] != b->shape[i] || a->shape[i] != out->shape[i]) {
                same_shape = 0;
                break;
            }
        }
    }

    if (same_shape && wowml_tensor_is_contiguous(a) && wowml_tensor_is_contiguous(b) &&
        wowml_tensor_is_contiguous(out)) {
        int64_t n = out->numel;
        if (out->dtype == WOWML_FLOAT32) {
            const float* pa = (const float*)a->data;
            const float* pb = (const float*)b->data;
            float* po = (float*)out->data;
            switch (op) {
                case OP_ADD:
                    #pragma omp parallel for if (n > 32768)
                    for (int i = 0; i < (int)n; i++) po[i] = pa[i] + pb[i];
                    break;
                case OP_SUB:
                    #pragma omp parallel for if (n > 32768)
                    for (int i = 0; i < (int)n; i++) po[i] = pa[i] - pb[i];
                    break;
                case OP_MUL:
                    #pragma omp parallel for if (n > 32768)
                    for (int i = 0; i < (int)n; i++) po[i] = pa[i] * pb[i];
                    break;
                case OP_DIV:
                    #pragma omp parallel for if (n > 32768)
                    for (int i = 0; i < (int)n; i++) po[i] = pa[i] / pb[i];
                    break;
            }
            return WOWML_OK;
        } else if (out->dtype == WOWML_FLOAT64) {
            const double* pa = (const double*)a->data;
            const double* pb = (const double*)b->data;
            double* po = (double*)out->data;
            switch (op) {
                case OP_ADD:
                    #pragma omp parallel for if (n > 32768)
                    for (int i = 0; i < (int)n; i++) po[i] = pa[i] + pb[i];
                    break;
                case OP_SUB:
                    #pragma omp parallel for if (n > 32768)
                    for (int i = 0; i < (int)n; i++) po[i] = pa[i] - pb[i];
                    break;
                case OP_MUL:
                    #pragma omp parallel for if (n > 32768)
                    for (int i = 0; i < (int)n; i++) po[i] = pa[i] * pb[i];
                    break;
                case OP_DIV:
                    #pragma omp parallel for if (n > 32768)
                    for (int i = 0; i < (int)n; i++) po[i] = pa[i] / pb[i];
                    break;
            }
            return WOWML_OK;
        }
    }

    /* General broadcasted path */
    wowml_tensor_t va, vb;
    WOWML_CHECK(wowml_tensor_broadcast_to(&va, a, out->ndim, out->shape) == WOWML_OK,
                WOWML_ERR_SHAPE_MISMATCH);
    WOWML_CHECK(wowml_tensor_broadcast_to(&vb, b, out->ndim, out->shape) == WOWML_OK,
                WOWML_ERR_SHAPE_MISMATCH);

    int64_t n = out->numel;
    if (out->dtype == WOWML_FLOAT32) {
        const float* pa = (const float*)va.data;
        const float* pb = (const float*)vb.data;
        float* po = (float*)out->data;
        #pragma omp parallel for if (n > 32768)
        for (int i = 0; i < (int)n; i++) {
            int64_t oa = strided_offset(i, out->ndim, out->shape, va.strides);
            int64_t ob = strided_offset(i, out->ndim, out->shape, vb.strides);
            int64_t oo = strided_offset(i, out->ndim, out->shape, out->strides);
            float x = pa[oa], y = pb[ob];
            po[oo] = (op == OP_ADD) ? (x + y) : (op == OP_SUB) ? (x - y) : (op == OP_MUL) ? (x * y) : (x / y);
        }
        return WOWML_OK;
    } else if (out->dtype == WOWML_FLOAT64) {
        const double* pa = (const double*)va.data;
        const double* pb = (const double*)vb.data;
        double* po = (double*)out->data;
        #pragma omp parallel for if (n > 32768)
        for (int i = 0; i < (int)n; i++) {
            int64_t oa = strided_offset(i, out->ndim, out->shape, va.strides);
            int64_t ob = strided_offset(i, out->ndim, out->shape, vb.strides);
            int64_t oo = strided_offset(i, out->ndim, out->shape, out->strides);
            double x = pa[oa], y = pb[ob];
            po[oo] = (op == OP_ADD) ? (x + y) : (op == OP_SUB) ? (x - y) : (op == OP_MUL) ? (x * y) : (x / y);
        }
        return WOWML_OK;
    }
    return WOWML_ERR_INVALID_DTYPE;
}

wowml_status_t wowml_add(wowml_tensor_t* out, const wowml_tensor_t* a, const wowml_tensor_t* b) {
    return binary_op_impl(out, a, b, OP_ADD);
}
wowml_status_t wowml_sub(wowml_tensor_t* out, const wowml_tensor_t* a, const wowml_tensor_t* b) {
    return binary_op_impl(out, a, b, OP_SUB);
}
wowml_status_t wowml_mul(wowml_tensor_t* out, const wowml_tensor_t* a, const wowml_tensor_t* b) {
    return binary_op_impl(out, a, b, OP_MUL);
}
wowml_status_t wowml_div(wowml_tensor_t* out, const wowml_tensor_t* a, const wowml_tensor_t* b) {
    return binary_op_impl(out, a, b, OP_DIV);
}

wowml_status_t wowml_add_scalar(wowml_tensor_t* out, const wowml_tensor_t* a, double scalar) {
    WOWML_CHECK(out != NULL && a != NULL, WOWML_ERR_NULL_PTR);
    WOWML_CHECK(out->numel == a->numel && out->dtype == a->dtype, WOWML_ERR_SHAPE_MISMATCH);
    int64_t n = out->numel;
    if (out->dtype == WOWML_FLOAT32) {
        const float* pa = (const float*)a->data;
        float* po = (float*)out->data;
        float s = (float)scalar;
        for (int64_t i = 0; i < n; i++) po[i] = pa[i] + s;
        return WOWML_OK;
    } else if (out->dtype == WOWML_FLOAT64) {
        const double* pa = (const double*)a->data;
        double* po = (double*)out->data;
        for (int64_t i = 0; i < n; i++) po[i] = pa[i] + scalar;
        return WOWML_OK;
    }
    return WOWML_ERR_INVALID_DTYPE;
}

wowml_status_t wowml_mul_scalar(wowml_tensor_t* out, const wowml_tensor_t* a, double scalar) {
    WOWML_CHECK(out != NULL && a != NULL, WOWML_ERR_NULL_PTR);
    WOWML_CHECK(out->numel == a->numel && out->dtype == a->dtype, WOWML_ERR_SHAPE_MISMATCH);
    int64_t n = out->numel;
    if (out->dtype == WOWML_FLOAT32) {
        const float* pa = (const float*)a->data;
        float* po = (float*)out->data;
        float s = (float)scalar;
        for (int64_t i = 0; i < n; i++) po[i] = pa[i] * s;
        return WOWML_OK;
    } else if (out->dtype == WOWML_FLOAT64) {
        const double* pa = (const double*)a->data;
        double* po = (double*)out->data;
        for (int64_t i = 0; i < n; i++) po[i] = pa[i] * scalar;
        return WOWML_OK;
    }
    return WOWML_ERR_INVALID_DTYPE;
}

#define DEFINE_UNARY_OP(NAME, EXPR_F32, EXPR_F64)                                      \
wowml_status_t NAME(wowml_tensor_t* out, const wowml_tensor_t* a) {                    \
    WOWML_CHECK(out != NULL && a != NULL, WOWML_ERR_NULL_PTR);                         \
    WOWML_CHECK(out->numel == a->numel && out->dtype == a->dtype, WOWML_ERR_SHAPE_MISMATCH); \
    int64_t n = out->numel;                                                            \
    if (out->dtype == WOWML_FLOAT32) {                                                 \
        const float* pa = (const float*)a->data;                                       \
        float* po = (float*)out->data;                                                 \
        _Pragma("omp parallel for if (n > 32768)")                                     \
        for (int i = 0; i < (int)n; i++) { float x = pa[i]; po[i] = (EXPR_F32); }       \
        return WOWML_OK;                                                               \
    } else if (out->dtype == WOWML_FLOAT64) {                                          \
        const double* pa = (const double*)a->data;                                     \
        double* po = (double*)out->data;                                               \
        _Pragma("omp parallel for if (n > 32768)")                                     \
        for (int i = 0; i < (int)n; i++) { double x = pa[i]; po[i] = (EXPR_F64); }      \
        return WOWML_OK;                                                               \
    }                                                                                  \
    return WOWML_ERR_INVALID_DTYPE;                                                    \
}

DEFINE_UNARY_OP(wowml_neg,     -x,                            -x)
DEFINE_UNARY_OP(wowml_exp,     expf(x),                       exp(x))
DEFINE_UNARY_OP(wowml_log,     logf(x),                       log(x))
DEFINE_UNARY_OP(wowml_relu,    (x > 0.0f ? x : 0.0f),         (x > 0.0 ? x : 0.0))
DEFINE_UNARY_OP(wowml_sigmoid, (1.0f / (1.0f + expf(-x))),    (1.0 / (1.0 + exp(-x))))
DEFINE_UNARY_OP(wowml_tanh_op, tanhf(x),                      tanh(x))
DEFINE_UNARY_OP(wowml_sqrt,    sqrtf(x),                      sqrt(x))

wowml_status_t wowml_pow_op(wowml_tensor_t* out, const wowml_tensor_t* a, double exponent) {
    WOWML_CHECK(out != NULL && a != NULL, WOWML_ERR_NULL_PTR);
    WOWML_CHECK(out->numel == a->numel && out->dtype == a->dtype, WOWML_ERR_SHAPE_MISMATCH);
    int64_t n = out->numel;
    if (out->dtype == WOWML_FLOAT32) {
        const float* pa = (const float*)a->data;
        float* po = (float*)out->data;
        float e = (float)exponent;
        for (int64_t i = 0; i < n; i++) po[i] = powf(pa[i], e);
        return WOWML_OK;
    } else if (out->dtype == WOWML_FLOAT64) {
        const double* pa = (const double*)a->data;
        double* po = (double*)out->data;
        for (int64_t i = 0; i < n; i++) po[i] = pow(pa[i], exponent);
        return WOWML_OK;
    }
    return WOWML_ERR_INVALID_DTYPE;
}

typedef enum { RED_SUM = 0, RED_MEAN, RED_MAX, RED_MIN } red_op_t;

static wowml_status_t reduce_impl(wowml_tensor_t* out, const wowml_tensor_t* a,
                                  int axis, int keepdims, red_op_t op) {
    (void)keepdims;
    WOWML_CHECK(out != NULL && a != NULL, WOWML_ERR_NULL_PTR);
    WOWML_CHECK(a->dtype == out->dtype, WOWML_ERR_INVALID_DTYPE);

    /* Global reduction when axis < 0 */
    if (axis < 0) {
        WOWML_CHECK(out->numel == 1, WOWML_ERR_SHAPE_MISMATCH);
        int64_t n = a->numel;
        double acc = (op == RED_MAX) ? -DBL_MAX : (op == RED_MIN) ? DBL_MAX : 0.0;
        for (int64_t i = 0; i < n; i++) {
            double v = 0.0;
            wowml_tensor_get_flat(a, i, &v);
            if (op == RED_SUM || op == RED_MEAN) acc += v;
            else if (op == RED_MAX && v > acc) acc = v;
            else if (op == RED_MIN && v < acc) acc = v;
        }
        if (op == RED_MEAN) acc /= (double)n;
        return wowml_tensor_set_flat(out, 0, acc);
    }

    WOWML_CHECK(axis >= 0 && axis < a->ndim, WOWML_ERR_INVALID_ARGUMENT);
    int64_t axis_len = a->shape[axis];
    int64_t outer = 1, inner = 1;
    for (int d = 0; d < axis; d++) outer *= a->shape[d];
    for (int d = axis + 1; d < a->ndim; d++) inner *= a->shape[d];
    WOWML_CHECK(out->numel == outer * inner, WOWML_ERR_SHAPE_MISMATCH);

    for (int64_t o = 0; o < outer; o++) {
        for (int64_t in = 0; in < inner; in++) {
            double acc = (op == RED_MAX) ? -DBL_MAX : (op == RED_MIN) ? DBL_MAX : 0.0;
            for (int64_t k = 0; k < axis_len; k++) {
                int64_t flat_idx = (o * axis_len + k) * inner + in;
                double v = 0.0;
                wowml_tensor_get_flat(a, flat_idx, &v);
                if (op == RED_SUM || op == RED_MEAN) acc += v;
                else if (op == RED_MAX && v > acc) acc = v;
                else if (op == RED_MIN && v < acc) acc = v;
            }
            if (op == RED_MEAN) acc /= (double)axis_len;
            wowml_tensor_set_flat(out, o * inner + in, acc);
        }
    }
    return WOWML_OK;
}

wowml_status_t wowml_sum(wowml_tensor_t* out, const wowml_tensor_t* a, int axis, int keepdims) {
    return reduce_impl(out, a, axis, keepdims, RED_SUM);
}
wowml_status_t wowml_mean(wowml_tensor_t* out, const wowml_tensor_t* a, int axis, int keepdims) {
    return reduce_impl(out, a, axis, keepdims, RED_MEAN);
}
wowml_status_t wowml_max(wowml_tensor_t* out, const wowml_tensor_t* a, int axis, int keepdims) {
    return reduce_impl(out, a, axis, keepdims, RED_MAX);
}
wowml_status_t wowml_min(wowml_tensor_t* out, const wowml_tensor_t* a, int axis, int keepdims) {
    return reduce_impl(out, a, axis, keepdims, RED_MIN);
}

wowml_status_t wowml_equal(wowml_tensor_t* out, const wowml_tensor_t* a, const wowml_tensor_t* b) {
    WOWML_CHECK(out != NULL && a != NULL && b != NULL, WOWML_ERR_NULL_PTR);
    WOWML_CHECK(a->numel == b->numel && out->numel == a->numel, WOWML_ERR_SHAPE_MISMATCH);
    for (int64_t i = 0; i < a->numel; i++) {
        double va = 0.0, vb = 0.0;
        wowml_tensor_get_flat(a, i, &va);
        wowml_tensor_get_flat(b, i, &vb);
        wowml_tensor_set_flat(out, i, (va == vb) ? 1.0 : 0.0);
    }
    return WOWML_OK;
}

wowml_status_t wowml_im2col(wowml_tensor_t* col, const wowml_tensor_t* im,
                            int kernel_h, int kernel_w, int pad_h, int pad_w,
                            int stride_h, int stride_w, int dilation_h, int dilation_w) {
    WOWML_CHECK(col != NULL && im != NULL, WOWML_ERR_NULL_PTR);
    WOWML_CHECK(im->ndim == 4, WOWML_ERR_INVALID_ARGUMENT);
    int64_t N = im->shape[0], C = im->shape[1], H = im->shape[2], W = im->shape[3];
    int64_t out_h = (H + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int64_t out_w = (W + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    int64_t col_h = C * kernel_h * kernel_w;
    int64_t col_w = N * out_h * out_w;
    WOWML_CHECK(col->ndim == 2 && col->shape[0] == col_h && col->shape[1] == col_w, WOWML_ERR_SHAPE_MISMATCH);

    if (im->dtype == WOWML_FLOAT32) {
        const float* im_d = (const float*)im->data;
        float* col_d = (float*)col->data;
        for (int64_t n = 0; n < N; n++) {
            for (int64_t c = 0; c < C; c++) {
                for (int kh = 0; kh < kernel_h; kh++) {
                    for (int kw = 0; kw < kernel_w; kw++) {
                        int64_t col_row = (c * kernel_h + kh) * kernel_w + kw;
                        for (int64_t oh = 0; oh < out_h; oh++) {
                            int64_t ih = oh * stride_h - pad_h + kh * dilation_h;
                            for (int64_t ow = 0; ow < out_w; ow++) {
                                int64_t iw = ow * stride_w - pad_w + kw * dilation_w;
                                int64_t col_col = (n * out_h + oh) * out_w + ow;
                                float val = 0.0f;
                                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    val = im_d[((n * C + c) * H + ih) * W + iw];
                                }
                                col_d[col_row * col_w + col_col] = val;
                            }
                        }
                    }
                }
            }
        }
        return WOWML_OK;
    }
    return WOWML_ERR_INVALID_DTYPE;
}

wowml_status_t wowml_col2im(wowml_tensor_t* im, const wowml_tensor_t* col,
                            int kernel_h, int kernel_w, int pad_h, int pad_w,
                            int stride_h, int stride_w, int dilation_h, int dilation_w) {
    WOWML_CHECK(im != NULL && col != NULL, WOWML_ERR_NULL_PTR);
    WOWML_CHECK(im->ndim == 4 && col->ndim == 2, WOWML_ERR_INVALID_ARGUMENT);
    int64_t N = im->shape[0], C = im->shape[1], H = im->shape[2], W = im->shape[3];
    int64_t out_h = (H + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int64_t out_w = (W + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    int64_t col_w = N * out_h * out_w;

    if (im->dtype == WOWML_FLOAT32) {
        float* im_d = (float*)im->data;
        const float* col_d = (const float*)col->data;
        memset(im_d, 0, (size_t)im->numel * sizeof(float));
        for (int64_t n = 0; n < N; n++) {
            for (int64_t c = 0; c < C; c++) {
                for (int kh = 0; kh < kernel_h; kh++) {
                    for (int kw = 0; kw < kernel_w; kw++) {
                        int64_t col_row = (c * kernel_h + kh) * kernel_w + kw;
                        for (int64_t oh = 0; oh < out_h; oh++) {
                            int64_t ih = oh * stride_h - pad_h + kh * dilation_h;
                            for (int64_t ow = 0; ow < out_w; ow++) {
                                int64_t iw = ow * stride_w - pad_w + kw * dilation_w;
                                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    int64_t col_col = (n * out_h + oh) * out_w + ow;
                                    im_d[((n * C + c) * H + ih) * W + iw] += col_d[col_row * col_w + col_col];
                                }
                            }
                        }
                    }
                }
            }
        }
        return WOWML_OK;
    }
    return WOWML_ERR_INVALID_DTYPE;
}

wowml_status_t wowml_fused_linear_bias_relu(wowml_tensor_t* out, const wowml_tensor_t* x,
                                           const wowml_tensor_t* weight, const wowml_tensor_t* bias) {
    WOWML_CHECK(out != NULL && x != NULL && weight != NULL, WOWML_ERR_NULL_PTR);
    WOWML_CHECK(x->ndim == 2 && weight->ndim == 2 && out->ndim == 2, WOWML_ERR_INVALID_ARGUMENT);
    int64_t B = x->shape[0], In = x->shape[1], Out = weight->shape[0];
    WOWML_CHECK(weight->shape[1] == In && out->shape[0] == B && out->shape[1] == Out,
                WOWML_ERR_SHAPE_MISMATCH);

    if (x->dtype == WOWML_FLOAT32) {
        const float* X = (const float*)x->data;
        const float* W = (const float*)weight->data;
        const float* b = bias ? (const float*)bias->data : NULL;
        float* Y = (float*)out->data;

        #pragma omp parallel for if (B * Out > 4096)
        for (int i = 0; i < (int)B; i++) {
            for (int64_t j = 0; j < Out; j++) {
                float acc = b ? b[j] : 0.0f;
                const float* x_row = X + (int64_t)i * In;
                const float* w_row = W + j * In;
                for (int64_t k = 0; k < In; k++) {
                    acc += x_row[k] * w_row[k];
                }
                Y[(int64_t)i * Out + j] = (acc > 0.0f) ? acc : 0.0f;
            }
        }
        return WOWML_OK;
    }
    return WOWML_ERR_INVALID_DTYPE;
}

wowml_status_t wowml_fused_cross_entropy(double* loss, wowml_tensor_t* dlogits,
                                        const wowml_tensor_t* logits, const wowml_tensor_t* targets) {
    WOWML_CHECK(loss != NULL && logits != NULL && targets != NULL, WOWML_ERR_NULL_PTR);
    WOWML_CHECK(logits->ndim == 2 && targets->ndim == 1, WOWML_ERR_INVALID_ARGUMENT);
    int64_t B = logits->shape[0], C = logits->shape[1];
    WOWML_CHECK(targets->shape[0] == B, WOWML_ERR_SHAPE_MISMATCH);
    if (dlogits) {
        WOWML_CHECK(dlogits->ndim == 2 && dlogits->shape[0] == B && dlogits->shape[1] == C,
                    WOWML_ERR_SHAPE_MISMATCH);
    }

    if (logits->dtype == WOWML_FLOAT32) {
        const float* L = (const float*)logits->data;
        float* dL = dlogits ? (float*)dlogits->data : NULL;
        double total_loss = 0.0;
        float inv_B = 1.0f / (float)B;

        for (int64_t i = 0; i < B; i++) {
            const float* row = L + i * C;
            float max_val = row[0];
            for (int64_t c = 1; c < C; c++) {
                if (row[c] > max_val) max_val = row[c];
            }
            double sum_exp = 0.0;
            for (int64_t c = 0; c < C; c++) {
                sum_exp += exp((double)(row[c] - max_val));
            }
            double log_sum = (double)max_val + log(sum_exp);
            double target_d = 0.0;
            wowml_tensor_get_flat(targets, i, &target_d);
            int64_t tgt = (int64_t)target_d;
            WOWML_CHECK(tgt >= 0 && tgt < C, WOWML_ERR_INVALID_ARGUMENT);
            total_loss += (log_sum - (double)row[tgt]);

            if (dL) {
                float* drow = dL + i * C;
                for (int64_t c = 0; c < C; c++) {
                    float prob = (float)(exp((double)(row[c] - max_val)) / sum_exp);
                    drow[c] = (prob - (c == tgt ? 1.0f : 0.0f)) * inv_B;
                }
            }
        }
        *loss = total_loss / (double)B;
        return WOWML_OK;
    }
    return WOWML_ERR_INVALID_DTYPE;
}
