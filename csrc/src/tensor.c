/**
 * tensor.c — Complete Tensor lifecycle, N-D strides, zero-copy views, slicing, and broadcasting.
 */
#include "wowml/tensor.h"
#include "wowml/memory.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

static void compute_row_major_strides(int ndim, const int64_t* shape, int64_t* strides) {
    if (ndim <= 0) return;
    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    for (int i = ndim; i < WOWML_MAX_DIMS; i++) {
        strides[i] = 0;
    }
}

wowml_status_t wowml_tensor_alloc(wowml_tensor_t* t, wowml_dtype_t dtype,
                                  int ndim, const int64_t* shape) {
    WOWML_CHECK(t != NULL && shape != NULL, WOWML_ERR_NULL_PTR);
    WOWML_CHECK(ndim > 0 && ndim <= WOWML_MAX_DIMS, WOWML_ERR_INVALID_ARGUMENT);
    size_t elem_size = wowml_dtype_size(dtype);
    WOWML_CHECK(elem_size > 0, WOWML_ERR_INVALID_DTYPE);

    t->dtype = dtype;
    t->ndim = ndim;
    t->numel = 1;
    for (int i = 0; i < ndim; i++) {
        WOWML_CHECK(shape[i] >= 0, WOWML_ERR_INVALID_ARGUMENT);
        t->shape[i] = shape[i];
        t->numel *= shape[i];
    }
    for (int i = ndim; i < WOWML_MAX_DIMS; i++) {
        t->shape[i] = 0;
    }
    compute_row_major_strides(ndim, shape, t->strides);

    size_t nbytes = (size_t)t->numel * elem_size;
    t->data = wowml_aligned_alloc(WOWML_DEFAULT_ALIGNMENT, nbytes);
    if (!t->data) {
        t->numel = 0;
        return WOWML_ERR_OUT_OF_MEMORY;
    }
    t->owns_data = 1;
    return WOWML_OK;
}

wowml_status_t wowml_tensor_free(wowml_tensor_t* t) {
    WOWML_CHECK(t != NULL, WOWML_ERR_NULL_PTR);
    if (t->owns_data && t->data) {
        wowml_aligned_free(t->data);
    }
    t->data = NULL;
    t->numel = 0;
    t->ndim = 0;
    t->owns_data = 0;
    return WOWML_OK;
}

wowml_status_t wowml_tensor_from_data(wowml_tensor_t* t, void* data, wowml_dtype_t dtype,
                                      int ndim, const int64_t* shape, const int64_t* strides,
                                      int owns_data) {
    WOWML_CHECK(t != NULL && data != NULL && shape != NULL, WOWML_ERR_NULL_PTR);
    WOWML_CHECK(ndim > 0 && ndim <= WOWML_MAX_DIMS, WOWML_ERR_INVALID_ARGUMENT);
    WOWML_CHECK(wowml_dtype_size(dtype) > 0, WOWML_ERR_INVALID_DTYPE);

    t->data = data;
    t->dtype = dtype;
    t->ndim = ndim;
    t->owns_data = owns_data ? 1 : 0;
    t->numel = 1;

    for (int i = 0; i < ndim; i++) {
        WOWML_CHECK(shape[i] >= 0, WOWML_ERR_INVALID_ARGUMENT);
        t->shape[i] = shape[i];
        t->numel *= shape[i];
    }
    for (int i = ndim; i < WOWML_MAX_DIMS; i++) {
        t->shape[i] = 0;
    }
    if (strides) {
        for (int i = 0; i < ndim; i++) {
            t->strides[i] = strides[i];
        }
        for (int i = ndim; i < WOWML_MAX_DIMS; i++) {
            t->strides[i] = 0;
        }
    } else {
        compute_row_major_strides(ndim, shape, t->strides);
    }
    return WOWML_OK;
}

wowml_status_t wowml_tensor_zeros(wowml_tensor_t* t, wowml_dtype_t dtype,
                                  int ndim, const int64_t* shape) {
    wowml_status_t s = wowml_tensor_alloc(t, dtype, ndim, shape);
    if (s != WOWML_OK) return s;
    memset(t->data, 0, (size_t)t->numel * wowml_dtype_size(dtype));
    return WOWML_OK;
}

wowml_status_t wowml_tensor_fill(wowml_tensor_t* t, double value) {
    WOWML_CHECK(t != NULL && t->data != NULL, WOWML_ERR_NULL_PTR);
    if (!wowml_tensor_is_contiguous(t)) {
        for (int64_t i = 0; i < t->numel; i++) {
            int64_t rem = i;
            int64_t offset = 0;
            for (int d = t->ndim - 1; d >= 0; d--) {
                int64_t idx = (t->shape[d] > 0) ? (rem % t->shape[d]) : 0;
                rem = (t->shape[d] > 0) ? (rem / t->shape[d]) : 0;
                offset += idx * t->strides[d];
            }
            switch (t->dtype) {
                case WOWML_FLOAT32: ((float*)t->data)[offset] = (float)value; break;
                case WOWML_FLOAT64: ((double*)t->data)[offset] = value; break;
                case WOWML_INT32:   ((int32_t*)t->data)[offset] = (int32_t)value; break;
                case WOWML_INT64:   ((int64_t*)t->data)[offset] = (int64_t)value; break;
                default: return WOWML_ERR_INVALID_DTYPE;
            }
        }
        return WOWML_OK;
    }

    int64_t n = t->numel;
    switch (t->dtype) {
        case WOWML_FLOAT32: {
            float* p = (float*)t->data;
            float v = (float)value;
            for (int64_t i = 0; i < n; i++) p[i] = v;
            break;
        }
        case WOWML_FLOAT64: {
            double* p = (double*)t->data;
            for (int64_t i = 0; i < n; i++) p[i] = value;
            break;
        }
        case WOWML_INT32: {
            int32_t* p = (int32_t*)t->data;
            int32_t v = (int32_t)value;
            for (int64_t i = 0; i < n; i++) p[i] = v;
            break;
        }
        case WOWML_INT64: {
            int64_t* p = (int64_t*)t->data;
            int64_t v = (int64_t)value;
            for (int64_t i = 0; i < n; i++) p[i] = v;
            break;
        }
        default:
            return WOWML_ERR_INVALID_DTYPE;
    }
    return WOWML_OK;
}

wowml_status_t wowml_tensor_ones(wowml_tensor_t* t, wowml_dtype_t dtype,
                                 int ndim, const int64_t* shape) {
    wowml_status_t s = wowml_tensor_alloc(t, dtype, ndim, shape);
    if (s != WOWML_OK) return s;
    return wowml_tensor_fill(t, 1.0);
}

wowml_status_t wowml_tensor_clone(wowml_tensor_t* dst, const wowml_tensor_t* src) {
    WOWML_CHECK(dst != NULL && src != NULL, WOWML_ERR_NULL_PTR);
    if (wowml_tensor_is_contiguous(src)) {
        wowml_status_t s = wowml_tensor_alloc(dst, src->dtype, src->ndim, src->shape);
        if (s != WOWML_OK) return s;
        memcpy(dst->data, src->data, (size_t)src->numel * wowml_dtype_size(src->dtype));
        return WOWML_OK;
    }
    return wowml_tensor_contiguous(dst, src);
}

wowml_status_t wowml_tensor_set_flat(wowml_tensor_t* t, int64_t idx, double val) {
    WOWML_CHECK(t != NULL && t->data != NULL, WOWML_ERR_NULL_PTR);
    WOWML_CHECK(idx >= 0 && idx < t->numel, WOWML_ERR_INVALID_ARGUMENT);
    int64_t offset = idx;
    if (!wowml_tensor_is_contiguous(t)) {
        int64_t rem = idx;
        offset = 0;
        for (int d = t->ndim - 1; d >= 0; d--) {
            int64_t coord = (t->shape[d] > 0) ? (rem % t->shape[d]) : 0;
            rem = (t->shape[d] > 0) ? (rem / t->shape[d]) : 0;
            offset += coord * t->strides[d];
        }
    }
    switch (t->dtype) {
        case WOWML_FLOAT32: ((float*)t->data)[offset] = (float)val; break;
        case WOWML_FLOAT64: ((double*)t->data)[offset] = val; break;
        case WOWML_INT32:   ((int32_t*)t->data)[offset] = (int32_t)val; break;
        case WOWML_INT64:   ((int64_t*)t->data)[offset] = (int64_t)val; break;
        default: return WOWML_ERR_INVALID_DTYPE;
    }
    return WOWML_OK;
}

wowml_status_t wowml_tensor_get_flat(const wowml_tensor_t* t, int64_t idx, double* val) {
    WOWML_CHECK(t != NULL && t->data != NULL && val != NULL, WOWML_ERR_NULL_PTR);
    WOWML_CHECK(idx >= 0 && idx < t->numel, WOWML_ERR_INVALID_ARGUMENT);
    int64_t offset = idx;
    if (!wowml_tensor_is_contiguous(t)) {
        int64_t rem = idx;
        offset = 0;
        for (int d = t->ndim - 1; d >= 0; d--) {
            int64_t coord = (t->shape[d] > 0) ? (rem % t->shape[d]) : 0;
            rem = (t->shape[d] > 0) ? (rem / t->shape[d]) : 0;
            offset += coord * t->strides[d];
        }
    }
    switch (t->dtype) {
        case WOWML_FLOAT32: *val = (double)((const float*)t->data)[offset]; break;
        case WOWML_FLOAT64: *val = ((const double*)t->data)[offset]; break;
        case WOWML_INT32:   *val = (double)((const int32_t*)t->data)[offset]; break;
        case WOWML_INT64:   *val = (double)((const int64_t*)t->data)[offset]; break;
        default: return WOWML_ERR_INVALID_DTYPE;
    }
    return WOWML_OK;
}

int wowml_tensor_is_contiguous(const wowml_tensor_t* t) {
    if (!t || t->ndim <= 0) return 0;
    int64_t expected = 1;
    for (int i = t->ndim - 1; i >= 0; i--) {
        if (t->shape[i] != 1 && t->strides[i] != expected) {
            return 0;
        }
        expected *= t->shape[i];
    }
    return 1;
}

wowml_status_t wowml_tensor_contiguous(wowml_tensor_t* dst, const wowml_tensor_t* src) {
    WOWML_CHECK(dst != NULL && src != NULL && src->data != NULL, WOWML_ERR_NULL_PTR);
    wowml_status_t s = wowml_tensor_alloc(dst, src->dtype, src->ndim, src->shape);
    if (s != WOWML_OK) return s;

    if (wowml_tensor_is_contiguous(src)) {
        memcpy(dst->data, src->data, (size_t)src->numel * wowml_dtype_size(src->dtype));
        return WOWML_OK;
    }

    size_t elem_size = wowml_dtype_size(src->dtype);
    uint8_t* dst_bytes = (uint8_t*)dst->data;
    const uint8_t* src_bytes = (const uint8_t*)src->data;

    for (int64_t i = 0; i < src->numel; i++) {
        int64_t rem = i;
        int64_t src_off = 0;
        for (int d = src->ndim - 1; d >= 0; d--) {
            int64_t coord = (src->shape[d] > 0) ? (rem % src->shape[d]) : 0;
            rem = (src->shape[d] > 0) ? (rem / src->shape[d]) : 0;
            src_off += coord * src->strides[d];
        }
        memcpy(dst_bytes + (size_t)i * elem_size, src_bytes + (size_t)src_off * elem_size, elem_size);
    }
    return WOWML_OK;
}

wowml_status_t wowml_tensor_reshape(wowml_tensor_t* t, int ndim, const int64_t* new_shape) {
    WOWML_CHECK(t != NULL && new_shape != NULL, WOWML_ERR_NULL_PTR);
    WOWML_CHECK(ndim > 0 && ndim <= WOWML_MAX_DIMS, WOWML_ERR_INVALID_ARGUMENT);
    WOWML_CHECK(wowml_tensor_is_contiguous(t), WOWML_ERR_INVALID_ARGUMENT);

    int64_t resolved[WOWML_MAX_DIMS];
    int neg_idx = -1;
    int64_t known_prod = 1;
    for (int i = 0; i < ndim; i++) {
        if (new_shape[i] == -1) {
            WOWML_CHECK(neg_idx == -1, WOWML_ERR_INVALID_ARGUMENT);
            neg_idx = i;
            resolved[i] = 1;
        } else {
            WOWML_CHECK(new_shape[i] >= 0, WOWML_ERR_INVALID_ARGUMENT);
            resolved[i] = new_shape[i];
            known_prod *= new_shape[i];
        }
    }
    if (neg_idx != -1) {
        WOWML_CHECK(known_prod > 0 && (t->numel % known_prod == 0), WOWML_ERR_SHAPE_MISMATCH);
        resolved[neg_idx] = t->numel / known_prod;
    } else {
        WOWML_CHECK(known_prod == t->numel, WOWML_ERR_SHAPE_MISMATCH);
    }

    t->ndim = ndim;
    for (int i = 0; i < ndim; i++) t->shape[i] = resolved[i];
    for (int i = ndim; i < WOWML_MAX_DIMS; i++) t->shape[i] = 0;
    compute_row_major_strides(ndim, resolved, t->strides);
    return WOWML_OK;
}

wowml_status_t wowml_tensor_transpose(wowml_tensor_t* dst, const wowml_tensor_t* src,
                                      int dim0, int dim1) {
    WOWML_CHECK(dst != NULL && src != NULL, WOWML_ERR_NULL_PTR);
    if (dim0 < 0) dim0 += src->ndim;
    if (dim1 < 0) dim1 += src->ndim;
    WOWML_CHECK(dim0 >= 0 && dim0 < src->ndim && dim1 >= 0 && dim1 < src->ndim,
                WOWML_ERR_INVALID_ARGUMENT);

    *dst = *src;
    dst->owns_data = 0;
    int64_t tmp_s = dst->shape[dim0];
    dst->shape[dim0] = dst->shape[dim1];
    dst->shape[dim1] = tmp_s;

    int64_t tmp_st = dst->strides[dim0];
    dst->strides[dim0] = dst->strides[dim1];
    dst->strides[dim1] = tmp_st;
    return WOWML_OK;
}

wowml_status_t wowml_tensor_slice(wowml_tensor_t* view, const wowml_tensor_t* src,
                                  int dim, int64_t start, int64_t end, int64_t step) {
    WOWML_CHECK(view != NULL && src != NULL, WOWML_ERR_NULL_PTR);
    if (dim < 0) dim += src->ndim;
    WOWML_CHECK(dim >= 0 && dim < src->ndim, WOWML_ERR_INVALID_ARGUMENT);
    WOWML_CHECK(step > 0, WOWML_ERR_INVALID_ARGUMENT);

    int64_t dim_len = src->shape[dim];
    if (start < 0) start += dim_len;
    if (end < 0) end += dim_len;
    if (start < 0) start = 0;
    if (end > dim_len) end = dim_len;
    WOWML_CHECK(start <= end, WOWML_ERR_INVALID_ARGUMENT);

    int64_t new_len = (end - start + step - 1) / step;
    *view = *src;
    view->owns_data = 0;
    view->data = (uint8_t*)src->data + (size_t)(start * src->strides[dim]) * wowml_dtype_size(src->dtype);
    view->shape[dim] = new_len;
    view->strides[dim] = src->strides[dim] * step;
    view->numel = 1;
    for (int i = 0; i < view->ndim; i++) {
        view->numel *= view->shape[i];
    }
    return WOWML_OK;
}

wowml_status_t wowml_broadcast_shapes(int ndim_a, const int64_t* shape_a,
                                      int ndim_b, const int64_t* shape_b,
                                      int* out_ndim, int64_t* out_shape) {
    WOWML_CHECK(shape_a != NULL && shape_b != NULL && out_ndim != NULL && out_shape != NULL,
                WOWML_ERR_NULL_PTR);
    int ndim = (ndim_a > ndim_b) ? ndim_a : ndim_b;
    WOWML_CHECK(ndim > 0 && ndim <= WOWML_MAX_DIMS, WOWML_ERR_INVALID_ARGUMENT);

    for (int i = 0; i < ndim; i++) {
        int ia = ndim_a - 1 - i;
        int ib = ndim_b - 1 - i;
        int io = ndim - 1 - i;
        int64_t da = (ia >= 0) ? shape_a[ia] : 1;
        int64_t db = (ib >= 0) ? shape_b[ib] : 1;
        if (da == db) {
            out_shape[io] = da;
        } else if (da == 1) {
            out_shape[io] = db;
        } else if (db == 1) {
            out_shape[io] = da;
        } else {
            return WOWML_ERR_SHAPE_MISMATCH;
        }
    }
    *out_ndim = ndim;
    return WOWML_OK;
}

wowml_status_t wowml_tensor_broadcast_to(wowml_tensor_t* view, const wowml_tensor_t* src,
                                        int out_ndim, const int64_t* out_shape) {
    WOWML_CHECK(view != NULL && src != NULL && out_shape != NULL, WOWML_ERR_NULL_PTR);
    WOWML_CHECK(out_ndim >= src->ndim && out_ndim <= WOWML_MAX_DIMS, WOWML_ERR_SHAPE_MISMATCH);

    *view = *src;
    view->owns_data = 0;
    view->ndim = out_ndim;
    view->numel = 1;

    int offset = out_ndim - src->ndim;
    for (int i = 0; i < out_ndim; i++) {
        view->shape[i] = out_shape[i];
        view->numel *= out_shape[i];
        if (i < offset) {
            view->strides[i] = 0;
        } else {
            int64_t src_dim = src->shape[i - offset];
            if (src_dim == out_shape[i]) {
                view->strides[i] = src->strides[i - offset];
            } else if (src_dim == 1) {
                view->strides[i] = 0;
            } else {
                return WOWML_ERR_SHAPE_MISMATCH;
            }
        }
    }
    for (int i = out_ndim; i < WOWML_MAX_DIMS; i++) {
        view->shape[i] = 0;
        view->strides[i] = 0;
    }
    return WOWML_OK;
}

void wowml_tensor_print(const wowml_tensor_t* t) {
    if (!t) {
        printf("Tensor(null)\n");
        return;
    }
    printf("Tensor(dtype=%s, ndim=%d, shape=[", wowml_dtype_name(t->dtype), t->ndim);
    for (int i = 0; i < t->ndim; i++) {
        printf("%lld%s", (long long)t->shape[i], i < t->ndim - 1 ? ", " : "");
    }
    printf("], numel=%lld, contiguous=%d)\n", (long long)t->numel, wowml_tensor_is_contiguous(t));
}
