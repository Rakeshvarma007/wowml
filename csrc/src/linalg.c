/**
 * linalg.c — Linear algebra decompositions, solvers, and norms stubs.
 */
#include "wowml/linalg.h"

wowml_status_t wowml_linalg_lu(wowml_tensor_t* lu, int64_t* piv, const wowml_tensor_t* a) {
    (void)lu; (void)piv; (void)a;
    /* TODO: Doolittle LU / LAPACKE_dgetrf */
    return WOWML_ERR_NOT_IMPLEMENTED;
}

wowml_status_t wowml_linalg_qr(wowml_tensor_t* q, wowml_tensor_t* r, const wowml_tensor_t* a) {
    (void)q; (void)r; (void)a;
    /* TODO: Householder QR / LAPACKE_dgeqrf */
    return WOWML_ERR_NOT_IMPLEMENTED;
}

wowml_status_t wowml_linalg_svd(wowml_tensor_t* u, wowml_tensor_t* s, wowml_tensor_t* vt,
                                const wowml_tensor_t* a) {
    (void)u; (void)s; (void)vt; (void)a;
    /* TODO: Golub-Kahan / LAPACKE_dgesdd */
    return WOWML_ERR_NOT_IMPLEMENTED;
}

wowml_status_t wowml_linalg_cholesky(wowml_tensor_t* l, const wowml_tensor_t* a) {
    (void)l; (void)a;
    /* TODO: Cholesky decomposition / LAPACKE_dpotrf */
    return WOWML_ERR_NOT_IMPLEMENTED;
}

wowml_status_t wowml_linalg_eigh(wowml_tensor_t* eigenvalues, wowml_tensor_t* eigenvectors,
                                const wowml_tensor_t* a) {
    (void)eigenvalues; (void)eigenvectors; (void)a;
    /* TODO: Symmetric eigenvalue decomposition / LAPACKE_dsyev */
    return WOWML_ERR_NOT_IMPLEMENTED;
}

wowml_status_t wowml_linalg_solve(wowml_tensor_t* x, const wowml_tensor_t* a, const wowml_tensor_t* b) {
    (void)x; (void)a; (void)b;
    /* TODO: Linear system solve Ax = b / LAPACKE_dgesv */
    return WOWML_ERR_NOT_IMPLEMENTED;
}

wowml_status_t wowml_linalg_inv(wowml_tensor_t* out, const wowml_tensor_t* a) {
    (void)out; (void)a;
    /* TODO: Matrix inversion / LAPACKE_dgetri */
    return WOWML_ERR_NOT_IMPLEMENTED;
}

wowml_status_t wowml_linalg_det(double* result, const wowml_tensor_t* a) {
    (void)result; (void)a;
    /* TODO: Matrix determinant */
    return WOWML_ERR_NOT_IMPLEMENTED;
}

wowml_status_t wowml_linalg_trace(double* result, const wowml_tensor_t* a) {
    (void)result; (void)a;
    /* TODO: Matrix trace */
    return WOWML_ERR_NOT_IMPLEMENTED;
}

wowml_status_t wowml_linalg_norm(double* result, const wowml_tensor_t* a, int ord) {
    (void)result; (void)a; (void)ord;
    /* TODO: Matrix / vector norm */
    return WOWML_ERR_NOT_IMPLEMENTED;
}
