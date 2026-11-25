/**
 * linalg.h — Linear algebra decompositions, solvers, and queries.
 */
#ifndef WOWML_LINALG_H
#define WOWML_LINALG_H

#include "tensor.h"

/* ---- Matrix Decompositions (Pure C + LAPACKE fast paths) ---- */
wowml_status_t wowml_linalg_lu(wowml_tensor_t* lu, int64_t* piv, const wowml_tensor_t* a);
wowml_status_t wowml_linalg_qr(wowml_tensor_t* q, wowml_tensor_t* r, const wowml_tensor_t* a);
wowml_status_t wowml_linalg_svd(wowml_tensor_t* u, wowml_tensor_t* s, wowml_tensor_t* vt,
                                const wowml_tensor_t* a);
wowml_status_t wowml_linalg_cholesky(wowml_tensor_t* l, const wowml_tensor_t* a);
wowml_status_t wowml_linalg_eigh(wowml_tensor_t* eigenvalues, wowml_tensor_t* eigenvectors,
                                const wowml_tensor_t* a);

/* ---- Linear System Solvers & Inversion ---- */
wowml_status_t wowml_linalg_solve(wowml_tensor_t* x, const wowml_tensor_t* a, const wowml_tensor_t* b);
wowml_status_t wowml_linalg_inv(wowml_tensor_t* out, const wowml_tensor_t* a);

/* ---- Scalar Queries & Norms ---- */
wowml_status_t wowml_linalg_det(double* result, const wowml_tensor_t* a);
wowml_status_t wowml_linalg_trace(double* result, const wowml_tensor_t* a);
wowml_status_t wowml_linalg_norm(double* result, const wowml_tensor_t* a, int ord);

#endif /* WOWML_LINALG_H */
