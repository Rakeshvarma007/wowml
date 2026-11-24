/**
 * linalg.c — Linear algebra decompositions and solvers.
 */
#include "wowml/linalg.h"
#include "wowml/memory.h"
#include <math.h>

/* ---- Decompositions ---- */

wowml_status_t wowml_linalg_lu(wowml_tensor_t* lu, int64_t* piv,
                                const wowml_tensor_t* a) {
    /* TODO: Doolittle LU with partial pivoting */
    return WOWML_OK;
}

wowml_status_t wowml_linalg_qr(wowml_tensor_t* q, wowml_tensor_t* r,
                                const wowml_tensor_t* a) {
    /* TODO: Householder QR */
    return WOWML_OK;
}

wowml_status_t wowml_linalg_svd(wowml_tensor_t* u, wowml_tensor_t* s,
                                 wowml_tensor_t* vt, const wowml_tensor_t* a) {
    /* TODO: Golub-Kahan bidiagonalization + QR iteration */
    return WOWML_OK;
}

wowml_status_t wowml_linalg_cholesky(wowml_tensor_t* l,
                                      const wowml_tensor_t* a) {
    /* TODO: Cholesky-Banachiewicz */
    return WOWML_OK;
}

wowml_status_t wowml_linalg_eigh(wowml_tensor_t* eigenvalues,
                                  wowml_tensor_t* eigenvectors,
                                  const wowml_tensor_t* a) {
    /* TODO: Jacobi eigenvalue algorithm */
    return WOWML_OK;
}

/* ---- Solvers ---- */

wowml_status_t wowml_linalg_solve(wowml_tensor_t* x,
                                   const wowml_tensor_t* a,
                                   const wowml_tensor_t* b) {
    /* TODO: LU-based solver */
    return WOWML_OK;
}

wowml_status_t wowml_linalg_inv(wowml_tensor_t* out,
                                 const wowml_tensor_t* a) {
    /* TODO: Gauss-Jordan or LU-based */
    return WOWML_OK;
}

/* ---- Scalar queries ---- */

wowml_status_t wowml_linalg_det(double* result, const wowml_tensor_t* a) {
    /* TODO: LU-based determinant */
    return WOWML_OK;
}

wowml_status_t wowml_linalg_trace(double* result, const wowml_tensor_t* a) {
    /* TODO: sum of diagonal */
    return WOWML_OK;
}

wowml_status_t wowml_linalg_norm(double* result, const wowml_tensor_t* a, int ord) {
    /* TODO: L1, L2, Linf norms */
    return WOWML_OK;
}
