/**
 * matmul.c — Matrix multiplication routines.
 */
#include "wowml/matmul.h"
#include "wowml/memory.h"

wowml_status_t wowml_matmul(wowml_tensor_t* out,
                            const wowml_tensor_t* a, const wowml_tensor_t* b) {
    /* TODO: implement naive O(n^3) first, then optimize with tiling/SIMD */
    return WOWML_OK;
}

wowml_status_t wowml_bmm(wowml_tensor_t* out,
                          const wowml_tensor_t* a, const wowml_tensor_t* b) {
    /* TODO: implement batched matmul */
    return WOWML_OK;
}

wowml_status_t wowml_matvec(wowml_tensor_t* out,
                            const wowml_tensor_t* a, const wowml_tensor_t* x) {
    /* TODO: implement matrix-vector multiply */
    return WOWML_OK;
}

wowml_status_t wowml_dot(double* result,
                          const wowml_tensor_t* a, const wowml_tensor_t* b) {
    /* TODO: implement dot product */
    return WOWML_OK;
}
