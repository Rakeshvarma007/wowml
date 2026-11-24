/**
 * tensor.h — Core Tensor data structure and lifetime management.
 */
#ifndef WOWML_TENSOR_H
#define WOWML_TENSOR_H

#include "dtype.h"
#include "error.h"
#include <stddef.h>
#include <stdint.h>

#define WOWML_MAX_DIMS 8

typedef struct {
    void*          data;
    wowml_dtype_t  dtype;
    int            ndim;
    int64_t        shape[WOWML_MAX_DIMS];
    int64_t        strides[WOWML_MAX_DIMS];
    int64_t        numel;
    int            owns_data;
} wowml_tensor_t;

/* Lifecycle */
wowml_status_t wowml_tensor_alloc(wowml_tensor_t* t, wowml_dtype_t dtype,
                                  int ndim, const int64_t* shape);
wowml_status_t wowml_tensor_free(wowml_tensor_t* t);

/* Creation helpers */
wowml_status_t wowml_tensor_zeros(wowml_tensor_t* t, wowml_dtype_t dtype,
                                  int ndim, const int64_t* shape);
wowml_status_t wowml_tensor_ones(wowml_tensor_t* t, wowml_dtype_t dtype,
                                 int ndim, const int64_t* shape);
wowml_status_t wowml_tensor_fill(wowml_tensor_t* t, double value);
wowml_status_t wowml_tensor_clone(wowml_tensor_t* dst, const wowml_tensor_t* src);

/* Element access */
wowml_status_t wowml_tensor_set_flat(wowml_tensor_t* t, int64_t idx, double val);
wowml_status_t wowml_tensor_get_flat(const wowml_tensor_t* t, int64_t idx, double* val);

/* Shape manipulation */
wowml_status_t wowml_tensor_reshape(wowml_tensor_t* t, int ndim, const int64_t* new_shape);
wowml_status_t wowml_tensor_transpose(wowml_tensor_t* dst, const wowml_tensor_t* src);

/* Debug */
void wowml_tensor_print(const wowml_tensor_t* t);

#endif /* WOWML_TENSOR_H */
