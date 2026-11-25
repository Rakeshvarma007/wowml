/**
 * matmul.c — Matrix multiplication kernels (tiled C + BLAS fast paths) stubs.
 */
#include "wowml/matmul.h"

wowml_status_t wowml_matmul_c(wowml_tensor_t* out, const wowml_tensor_t* a, const wowml_tensor_t* b) {
    (void)out; (void)a; (void)b;
    /* TODO: Pure C11 tiled matrix multiplication with OpenMP */
    return WOWML_ERR_NOT_IMPLEMENTED;
}

wowml_status_t wowml_matmul_blas(wowml_tensor_t* out, const wowml_tensor_t* a, const wowml_tensor_t* b) {
    (void)out; (void)a; (void)b;
    /* TODO: cblas_dgemm / cblas_sgemm fast path */
    return WOWML_ERR_NOT_IMPLEMENTED;
}

wowml_status_t wowml_matmul(wowml_tensor_t* out, const wowml_tensor_t* a, const wowml_tensor_t* b) {
    (void)out; (void)a; (void)b;
    /* TODO: Dispatch to BLAS if available, else wowml_matmul_c */
    return WOWML_ERR_NOT_IMPLEMENTED;
}

wowml_status_t wowml_bmm(wowml_tensor_t* out, const wowml_tensor_t* a, const wowml_tensor_t* b) {
    (void)out; (void)a; (void)b;
    /* TODO: Batched 3-D matrix multiplication */
    return WOWML_ERR_NOT_IMPLEMENTED;
}

wowml_status_t wowml_matvec(wowml_tensor_t* out, const wowml_tensor_t* a, const wowml_tensor_t* x) {
    (void)out; (void)a; (void)x;
    /* TODO: Matrix-vector product */
    return WOWML_ERR_NOT_IMPLEMENTED;
}

wowml_status_t wowml_dot(double* result, const wowml_tensor_t* a, const wowml_tensor_t* b) {
    (void)result; (void)a; (void)b;
    /* TODO: 1-D vector dot product */
    return WOWML_ERR_NOT_IMPLEMENTED;
}
