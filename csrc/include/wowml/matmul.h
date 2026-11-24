/**
 * matmul.h — Matrix multiplication routines.
 */
#ifndef WOWML_MATMUL_H
#define WOWML_MATMUL_H

#include "tensor.h"

wowml_status_t wowml_matmul(wowml_tensor_t* out, const wowml_tensor_t* a, const wowml_tensor_t* b);
wowml_status_t wowml_bmm(wowml_tensor_t* out, const wowml_tensor_t* a, const wowml_tensor_t* b);
wowml_status_t wowml_matvec(wowml_tensor_t* out, const wowml_tensor_t* a, const wowml_tensor_t* x);
wowml_status_t wowml_dot(double* result, const wowml_tensor_t* a, const wowml_tensor_t* b);

#endif /* WOWML_MATMUL_H */
