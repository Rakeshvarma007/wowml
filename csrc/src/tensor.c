/**
 * tensor.c — Tensor lifecycle and shape manipulation.
 */
#include "wowml/tensor.h"
#include "wowml/memory.h"
#include <string.h>
#include <stdio.h>

wowml_status_t wowml_tensor_alloc(wowml_tensor_t* t, wowml_dtype_t dtype,
                                  int ndim, const int64_t* shape) {
    if (!t || !shape) return WOWML_ERR_NULL_PTR;
    if (ndim <= 0 || ndim > WOWML_MAX_DIMS) return WOWML_ERR_INVALID_ARGUMENT;

    t->dtype = dtype;
    t->ndim  = ndim;
    t->numel = 1;
    for (int i = 0; i < ndim; i++) {
        t->shape[i] = shape[i];
        t->numel *= shape[i];
    }

    /* Row-major strides */
    t->strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
        t->strides[i] = t->strides[i + 1] * t->shape[i + 1];
    }

    size_t nbytes = (size_t)t->numel * wowml_dtype_size(dtype);
    t->data = wowml_alloc(nbytes);
    if (!t->data) return WOWML_ERR_OUT_OF_MEMORY;

    t->owns_data = 1;
    return WOWML_OK;
}

wowml_status_t wowml_tensor_free(wowml_tensor_t* t) {
    if (!t) return WOWML_ERR_NULL_PTR;
    if (t->owns_data && t->data) {
        wowml_free(t->data);
    }
    t->data = NULL;
    t->numel = 0;
    return WOWML_OK;
}

wowml_status_t wowml_tensor_zeros(wowml_tensor_t* t, wowml_dtype_t dtype,
                                  int ndim, const int64_t* shape) {
    wowml_status_t s = wowml_tensor_alloc(t, dtype, ndim, shape);
    if (s != WOWML_OK) return s;
    memset(t->data, 0, (size_t)t->numel * wowml_dtype_size(dtype));
    return WOWML_OK;
}

wowml_status_t wowml_tensor_ones(wowml_tensor_t* t, wowml_dtype_t dtype,
                                 int ndim, const int64_t* shape) {
    /* TODO: implement */
    return WOWML_OK;
}

wowml_status_t wowml_tensor_fill(wowml_tensor_t* t, double value) {
    /* TODO: implement */
    return WOWML_OK;
}

wowml_status_t wowml_tensor_clone(wowml_tensor_t* dst, const wowml_tensor_t* src) {
    /* TODO: implement */
    return WOWML_OK;
}

wowml_status_t wowml_tensor_set_flat(wowml_tensor_t* t, int64_t idx, double val) {
    /* TODO: implement */
    return WOWML_OK;
}

wowml_status_t wowml_tensor_get_flat(const wowml_tensor_t* t, int64_t idx, double* val) {
    /* TODO: implement */
    return WOWML_OK;
}

wowml_status_t wowml_tensor_reshape(wowml_tensor_t* t, int ndim, const int64_t* new_shape) {
    /* TODO: implement */
    return WOWML_OK;
}

wowml_status_t wowml_tensor_transpose(wowml_tensor_t* dst, const wowml_tensor_t* src) {
    /* TODO: implement */
    return WOWML_OK;
}

void wowml_tensor_print(const wowml_tensor_t* t) {
    if (!t) { printf("Tensor(null)\n"); return; }
    printf("Tensor(dtype=%s, ndim=%d, shape=[", wowml_dtype_name(t->dtype), t->ndim);
    for (int i = 0; i < t->ndim; i++) {
        printf("%lld%s", (long long)t->shape[i], i < t->ndim - 1 ? ", " : "");
    }
    printf("], numel=%lld)\n", (long long)t->numel);
}
