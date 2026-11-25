/**
 * tensor.h — Core Tensor data structure, views, slicing, and broadcasting.
 */
#ifndef WOWML_TENSOR_H
#define WOWML_TENSOR_H

#include "dtype.h"
#include "error.h"
#include <stddef.h>
#include <stdint.h>

#define WOWML_MAX_DIMS 8

typedef struct {
    void*          data;                    /* Pointer to raw memory buffer */
    wowml_dtype_t  dtype;                   /* Data type (float32, etc.)    */
    int            ndim;                    /* Number of dimensions (<= 8)  */
    int64_t        shape[WOWML_MAX_DIMS];   /* Dimension sizes              */
    int64_t        strides[WOWML_MAX_DIMS]; /* Strides in element counts    */
    int64_t        numel;                   /* Total number of elements     */
    int            owns_data;               /* 1 if tensor owns buffer      */
} wowml_tensor_t;

/* ---- Lifecycle ---- */
wowml_status_t wowml_tensor_alloc(wowml_tensor_t* t, wowml_dtype_t dtype,
                                  int ndim, const int64_t* shape);
wowml_status_t wowml_tensor_free(wowml_tensor_t* t);
wowml_status_t wowml_tensor_from_data(wowml_tensor_t* t, void* data, wowml_dtype_t dtype,
                                      int ndim, const int64_t* shape, const int64_t* strides,
                                      int owns_data);

/* ---- Creation helpers ---- */
wowml_status_t wowml_tensor_zeros(wowml_tensor_t* t, wowml_dtype_t dtype,
                                  int ndim, const int64_t* shape);
wowml_status_t wowml_tensor_ones(wowml_tensor_t* t, wowml_dtype_t dtype,
                                 int ndim, const int64_t* shape);
wowml_status_t wowml_tensor_fill(wowml_tensor_t* t, double value);
wowml_status_t wowml_tensor_clone(wowml_tensor_t* dst, const wowml_tensor_t* src);

/* ---- Indexed Access ---- */
wowml_status_t wowml_tensor_set_flat(wowml_tensor_t* t, int64_t idx, double val);
wowml_status_t wowml_tensor_get_flat(const wowml_tensor_t* t, int64_t idx, double* val);

/* ---- Zero-Copy Views, Strides & Slicing ---- */
int            wowml_tensor_is_contiguous(const wowml_tensor_t* t);
wowml_status_t wowml_tensor_contiguous(wowml_tensor_t* dst, const wowml_tensor_t* src);
wowml_status_t wowml_tensor_reshape(wowml_tensor_t* t, int ndim, const int64_t* new_shape);
wowml_status_t wowml_tensor_transpose(wowml_tensor_t* dst, const wowml_tensor_t* src,
                                      int dim0, int dim1);
wowml_status_t wowml_tensor_slice(wowml_tensor_t* view, const wowml_tensor_t* src,
                                  int dim, int64_t start, int64_t end, int64_t step);

/* ---- Broadcasting Engine ---- */
wowml_status_t wowml_broadcast_shapes(int ndim_a, const int64_t* shape_a,
                                      int ndim_b, const int64_t* shape_b,
                                      int* out_ndim, int64_t* out_shape);
wowml_status_t wowml_tensor_broadcast_to(wowml_tensor_t* view, const wowml_tensor_t* src,
                                        int out_ndim, const int64_t* out_shape);

/* ---- Debug ---- */
void wowml_tensor_print(const wowml_tensor_t* t);

#endif /* WOWML_TENSOR_H */
