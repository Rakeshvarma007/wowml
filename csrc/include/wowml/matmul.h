/**
 * matmul.h — Matrix multiplication routines (pure C11 tiled + BLAS/OpenMP fast paths).
 */
#ifndef WOWML_MATMUL_H
#define WOWML_MATMUL_H

#include "tensor.h"

/* General 2-D matrix multiply: out = A @ B */
wowml_status_t wowml_matmul(wowml_tensor_t* out, const wowml_tensor_t* a, const wowml_tensor_t* b);

/* Batched matrix multiply for 3-D tensors: out[i] = A[i] @ B[i] */
wowml_status_t wowml_bmm(wowml_tensor_t* out, const wowml_tensor_t* a, const wowml_tensor_t* b);

/* Matrix-vector multiply: out = A @ x */
wowml_status_t wowml_matvec(wowml_tensor_t* out, const wowml_tensor_t* a, const wowml_tensor_t* x);

/* Vector dot product: *result = a · b */
wowml_status_t wowml_dot(double* result, const wowml_tensor_t* a, const wowml_tensor_t* b);

/* Explicit backend selectors */
wowml_status_t wowml_matmul_c(wowml_tensor_t* out, const wowml_tensor_t* a, const wowml_tensor_t* b);
wowml_status_t wowml_matmul_blas(wowml_tensor_t* out, const wowml_tensor_t* a, const wowml_tensor_t* b);

#endif /* WOWML_MATMUL_H */
