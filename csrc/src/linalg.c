/**
 * linalg.c — Linear algebra decompositions, solvers, and norms (pure C11 + LAPACKE fast paths).
 */
#include "wowml/linalg.h"
#include "wowml/memory.h"
#include <math.h>
#include <string.h>

static double get_f(const wowml_tensor_t* t, int64_t r, int64_t c) {
    int64_t idx = r * t->strides[0] + c * t->strides[1];
    if (t->dtype == WOWML_FLOAT64) return ((const double*)t->data)[idx];
    return (double)((const float*)t->data)[idx];
}

static void set_f(wowml_tensor_t* t, int64_t r, int64_t c, double v) {
    int64_t idx = r * t->strides[0] + c * t->strides[1];
    if (t->dtype == WOWML_FLOAT64) ((double*)t->data)[idx] = v;
    else ((float*)t->data)[idx] = (float)v;
}

wowml_status_t wowml_linalg_lu(wowml_tensor_t* lu, int64_t* piv, const wowml_tensor_t* a) {
    WOWML_CHECK(lu != NULL && piv != NULL && a != NULL, WOWML_ERR_NULL_PTR);
    WOWML_CHECK(a->ndim == 2 && lu->ndim == 2, WOWML_ERR_INVALID_ARGUMENT);
    int64_t m = a->shape[0], n = a->shape[1];
    WOWML_CHECK(lu->shape[0] == m && lu->shape[1] == n, WOWML_ERR_SHAPE_MISMATCH);
    int64_t kmax = (m < n) ? m : n;

    double* W = (double*)wowml_alloc((size_t)(m * n) * sizeof(double));
    WOWML_CHECK(W != NULL, WOWML_ERR_OUT_OF_MEMORY);
    for (int64_t i = 0; i < m; i++)
        for (int64_t j = 0; j < n; j++)
            W[i * n + j] = get_f(a, i, j);

    for (int64_t k = 0; k < kmax; k++) {
        int64_t p = k;
        double max_v = fabs(W[k * n + k]);
        for (int64_t i = k + 1; i < m; i++) {
            double v = fabs(W[i * n + k]);
            if (v > max_v) { max_v = v; p = i; }
        }
        piv[k] = p;
        if (p != k) {
            for (int64_t j = 0; j < n; j++) {
                double tmp = W[k * n + j];
                W[k * n + j] = W[p * n + j];
                W[p * n + j] = tmp;
            }
        }
        if (fabs(W[k * n + k]) < 1e-15) {
            wowml_free(W);
            return WOWML_ERR_SINGULAR_MATRIX;
        }
        for (int64_t i = k + 1; i < m; i++) {
            W[i * n + k] /= W[k * n + k];
            double mult = W[i * n + k];
            for (int64_t j = k + 1; j < n; j++) {
                W[i * n + j] -= mult * W[k * n + j];
            }
        }
    }

    for (int64_t i = 0; i < m; i++)
        for (int64_t j = 0; j < n; j++)
            set_f(lu, i, j, W[i * n + j]);

    wowml_free(W);
    return WOWML_OK;
}

wowml_status_t wowml_linalg_solve(wowml_tensor_t* x, const wowml_tensor_t* a, const wowml_tensor_t* b) {
    WOWML_CHECK(x != NULL && a != NULL && b != NULL, WOWML_ERR_NULL_PTR);
    WOWML_CHECK(a->ndim == 2 && a->shape[0] == a->shape[1], WOWML_ERR_INVALID_ARGUMENT);
    int64_t n = a->shape[0];
    int64_t nrhs = (b->ndim == 1) ? 1 : b->shape[1];
    WOWML_CHECK(b->shape[0] == n && x->shape[0] == n, WOWML_ERR_SHAPE_MISMATCH);

    wowml_tensor_t lu;
    WOWML_CHECK(wowml_tensor_alloc(&lu, WOWML_FLOAT64, 2, a->shape) == WOWML_OK, WOWML_ERR_OUT_OF_MEMORY);
    int64_t* piv = (int64_t*)wowml_alloc((size_t)n * sizeof(int64_t));
    if (!piv) { wowml_tensor_free(&lu); return WOWML_ERR_OUT_OF_MEMORY; }

    wowml_status_t s = wowml_linalg_lu(&lu, piv, a);
    if (s != WOWML_OK) { wowml_free(piv); wowml_tensor_free(&lu); return s; }

    double* rhs = (double*)wowml_alloc((size_t)(n * nrhs) * sizeof(double));
    if (!rhs) { wowml_free(piv); wowml_tensor_free(&lu); return WOWML_ERR_OUT_OF_MEMORY; }

    for (int64_t i = 0; i < n; i++) {
        for (int64_t c = 0; c < nrhs; c++) {
            double val = 0.0;
            wowml_tensor_get_flat(b, i * nrhs + c, &val);
            rhs[i * nrhs + c] = val;
        }
    }

    for (int64_t k = 0; k < n; k++) {
        if (piv[k] != k) {
            for (int64_t c = 0; c < nrhs; c++) {
                double tmp = rhs[k * nrhs + c];
                rhs[k * nrhs + c] = rhs[piv[k] * nrhs + c];
                rhs[piv[k] * nrhs + c] = tmp;
            }
        }
    }

    const double* LU = (const double*)lu.data;
    for (int64_t c = 0; c < nrhs; c++) {
        for (int64_t i = 0; i < n; i++) {
            double sum = rhs[i * nrhs + c];
            for (int64_t j = 0; j < i; j++) sum -= LU[i * n + j] * rhs[j * nrhs + c];
            rhs[i * nrhs + c] = sum;
        }
        for (int64_t i = n - 1; i >= 0; i--) {
            double sum = rhs[i * nrhs + c];
            for (int64_t j = i + 1; j < n; j++) sum -= LU[i * n + j] * rhs[j * nrhs + c];
            rhs[i * nrhs + c] = sum / LU[i * n + i];
        }
    }

    for (int64_t i = 0; i < n * nrhs; i++) {
        wowml_tensor_set_flat(x, i, rhs[i]);
    }

    wowml_free(rhs);
    wowml_free(piv);
    wowml_tensor_free(&lu);
    return WOWML_OK;
}

wowml_status_t wowml_linalg_inv(wowml_tensor_t* out, const wowml_tensor_t* a) {
    WOWML_CHECK(out != NULL && a != NULL, WOWML_ERR_NULL_PTR);
    WOWML_CHECK(a->ndim == 2 && a->shape[0] == a->shape[1], WOWML_ERR_INVALID_ARGUMENT);
    int64_t n = a->shape[0];
    wowml_tensor_t eye;
    WOWML_CHECK(wowml_tensor_zeros(&eye, a->dtype, 2, a->shape) == WOWML_OK, WOWML_ERR_OUT_OF_MEMORY);
    for (int64_t i = 0; i < n; i++) set_f(&eye, i, i, 1.0);
    wowml_status_t s = wowml_linalg_solve(out, a, &eye);
    wowml_tensor_free(&eye);
    return s;
}

wowml_status_t wowml_linalg_det(double* result, const wowml_tensor_t* a) {
    WOWML_CHECK(result != NULL && a != NULL, WOWML_ERR_NULL_PTR);
    WOWML_CHECK(a->ndim == 2 && a->shape[0] == a->shape[1], WOWML_ERR_INVALID_ARGUMENT);
    int64_t n = a->shape[0];

    wowml_tensor_t lu;
    WOWML_CHECK(wowml_tensor_alloc(&lu, WOWML_FLOAT64, 2, a->shape) == WOWML_OK, WOWML_ERR_OUT_OF_MEMORY);
    int64_t* piv = (int64_t*)wowml_alloc((size_t)n * sizeof(int64_t));
    if (!piv) { wowml_tensor_free(&lu); return WOWML_ERR_OUT_OF_MEMORY; }

    wowml_status_t s = wowml_linalg_lu(&lu, piv, a);
    if (s == WOWML_ERR_SINGULAR_MATRIX) {
        *result = 0.0;
        wowml_free(piv);
        wowml_tensor_free(&lu);
        return WOWML_OK;
    }
    if (s != WOWML_OK) { wowml_free(piv); wowml_tensor_free(&lu); return s; }

    double det = 1.0;
    int swaps = 0;
    const double* LU = (const double*)lu.data;
    for (int64_t i = 0; i < n; i++) {
        det *= LU[i * n + i];
        if (piv[i] != i) swaps++;
    }
    if (swaps % 2 != 0) det = -det;
    *result = det;

    wowml_free(piv);
    wowml_tensor_free(&lu);
    return WOWML_OK;
}

wowml_status_t wowml_linalg_cholesky(wowml_tensor_t* l, const wowml_tensor_t* a) {
    WOWML_CHECK(l != NULL && a != NULL, WOWML_ERR_NULL_PTR);
    WOWML_CHECK(a->ndim == 2 && a->shape[0] == a->shape[1], WOWML_ERR_INVALID_ARGUMENT);
    int64_t n = a->shape[0];
    WOWML_CHECK(l->shape[0] == n && l->shape[1] == n, WOWML_ERR_SHAPE_MISMATCH);

    double* L = (double*)wowml_calloc((size_t)(n * n), sizeof(double));
    WOWML_CHECK(L != NULL, WOWML_ERR_OUT_OF_MEMORY);

    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j <= i; j++) {
            double sum = 0.0;
            for (int64_t k = 0; k < j; k++) sum += L[i * n + k] * L[j * n + k];
            if (i == j) {
                double d = get_f(a, i, i) - sum;
                if (d <= 0.0) { wowml_free(L); return WOWML_ERR_SINGULAR_MATRIX; }
                L[i * n + j] = sqrt(d);
            } else {
                L[i * n + j] = (get_f(a, i, j) - sum) / L[j * n + j];
            }
        }
    }

    for (int64_t i = 0; i < n; i++)
        for (int64_t j = 0; j < n; j++)
            set_f(l, i, j, L[i * n + j]);

    wowml_free(L);
    return WOWML_OK;
}

wowml_status_t wowml_linalg_qr(wowml_tensor_t* q, wowml_tensor_t* r, const wowml_tensor_t* a) {
    WOWML_CHECK(q != NULL && r != NULL && a != NULL, WOWML_ERR_NULL_PTR);
    WOWML_CHECK(a->ndim == 2, WOWML_ERR_INVALID_ARGUMENT);
    int64_t m = a->shape[0], n = a->shape[1];
    int64_t kmax = (m < n) ? m : n;
    WOWML_CHECK(q->shape[0] == m && q->shape[1] == kmax && r->shape[0] == kmax && r->shape[1] == n,
                WOWML_ERR_SHAPE_MISMATCH);

    /* Modified Gram-Schmidt for reduced QR */
    double* Q = (double*)wowml_calloc((size_t)(m * kmax), sizeof(double));
    double* R = (double*)wowml_calloc((size_t)(kmax * n), sizeof(double));
    double* V = (double*)wowml_alloc((size_t)(m * n) * sizeof(double));
    if (!Q || !R || !V) { wowml_free(Q); wowml_free(R); wowml_free(V); return WOWML_ERR_OUT_OF_MEMORY; }

    for (int64_t i = 0; i < m; i++)
        for (int64_t j = 0; j < n; j++)
            V[i * n + j] = get_f(a, i, j);

    for (int64_t k = 0; k < kmax; k++) {
        double norm = 0.0;
        for (int64_t i = 0; i < m; i++) norm += V[i * n + k] * V[i * n + k];
        norm = sqrt(norm);
        R[k * n + k] = norm;
        if (norm > 1e-15) {
            for (int64_t i = 0; i < m; i++) Q[i * kmax + k] = V[i * n + k] / norm;
        }
        for (int64_t j = k + 1; j < n; j++) {
            double dot = 0.0;
            for (int64_t i = 0; i < m; i++) dot += Q[i * kmax + k] * V[i * n + j];
            R[k * n + j] = dot;
            for (int64_t i = 0; i < m; i++) V[i * n + j] -= dot * Q[i * kmax + k];
        }
    }

    for (int64_t i = 0; i < m; i++)
        for (int64_t j = 0; j < kmax; j++)
            set_f(q, i, j, Q[i * kmax + j]);
    for (int64_t i = 0; i < kmax; i++)
        for (int64_t j = 0; j < n; j++)
            set_f(r, i, j, R[i * n + j]);

    wowml_free(Q); wowml_free(R); wowml_free(V);
    return WOWML_OK;
}

wowml_status_t wowml_linalg_eigh(wowml_tensor_t* eigenvalues, wowml_tensor_t* eigenvectors,
                                const wowml_tensor_t* a) {
    WOWML_CHECK(eigenvalues != NULL && eigenvectors != NULL && a != NULL, WOWML_ERR_NULL_PTR);
    WOWML_CHECK(a->ndim == 2 && a->shape[0] == a->shape[1], WOWML_ERR_INVALID_ARGUMENT);
    int64_t n = a->shape[0];

    double* A = (double*)wowml_alloc((size_t)(n * n) * sizeof(double));
    double* V = (double*)wowml_calloc((size_t)(n * n), sizeof(double));
    if (!A || !V) { wowml_free(A); wowml_free(V); return WOWML_ERR_OUT_OF_MEMORY; }

    for (int64_t i = 0; i < n; i++) {
        V[i * n + i] = 1.0;
        for (int64_t j = 0; j < n; j++) A[i * n + j] = get_f(a, i, j);
    }

    /* Classical Jacobi eigenvalue algorithm */
    for (int sweep = 0; sweep < 50; sweep++) {
        double off_diag = 0.0;
        for (int64_t p = 0; p < n - 1; p++) {
            for (int64_t q = p + 1; q < n; q++) {
                off_diag += A[p * n + q] * A[p * n + q];
            }
        }
        if (sqrt(off_diag) < 1e-12) break;

        for (int64_t p = 0; p < n - 1; p++) {
            for (int64_t q = p + 1; q < n; q++) {
                double apq = A[p * n + q];
                if (fabs(apq) < 1e-15) continue;
                double tau = (A[q * n + q] - A[p * n + p]) / (2.0 * apq);
                double t = (tau >= 0.0) ? (1.0 / (tau + sqrt(1.0 + tau * tau)))
                                        : (-1.0 / (-tau + sqrt(1.0 + tau * tau)));
                double c = 1.0 / sqrt(1.0 + t * t);
                double s = t * c;

                A[p * n + p] -= t * apq;
                A[q * n + q] += t * apq;
                A[p * n + q] = 0.0;
                A[q * n + p] = 0.0;

                for (int64_t r = 0; r < n; r++) {
                    if (r != p && r != q) {
                        double arp = A[r * n + p], arq = A[r * n + q];
                        A[r * n + p] = A[p * n + r] = c * arp - s * arq;
                        A[r * n + q] = A[q * n + r] = s * arp + c * arq;
                    }
                    double vrp = V[r * n + p], vrq = V[r * n + q];
                    V[r * n + p] = c * vrp - s * vrq;
                    V[r * n + q] = s * vrp + c * vrq;
                }
            }
        }
    }

    for (int64_t i = 0; i < n; i++) wowml_tensor_set_flat(eigenvalues, i, A[i * n + i]);
    for (int64_t i = 0; i < n; i++)
        for (int64_t j = 0; j < n; j++)
            set_f(eigenvectors, i, j, V[i * n + j]);

    wowml_free(A); wowml_free(V);
    return WOWML_OK;
}

wowml_status_t wowml_linalg_svd(wowml_tensor_t* u, wowml_tensor_t* s, wowml_tensor_t* vt,
                                const wowml_tensor_t* a) {
    WOWML_CHECK(u != NULL && s != NULL && vt != NULL && a != NULL, WOWML_ERR_NULL_PTR);
    WOWML_CHECK(a->ndim == 2, WOWML_ERR_INVALID_ARGUMENT);
    int64_t m = a->shape[0], n = a->shape[1];
    int64_t k = (m < n) ? m : n;

    /* Compute SVD via eigendecomposition of A^T A (for m >= n) or A A^T */
    double* ATA = (double*)wowml_calloc((size_t)(n * n), sizeof(double));
    double* A = (double*)wowml_alloc((size_t)(m * n) * sizeof(double));
    if (!ATA || !A) { wowml_free(ATA); wowml_free(A); return WOWML_ERR_OUT_OF_MEMORY; }

    for (int64_t i = 0; i < m; i++)
        for (int64_t j = 0; j < n; j++)
            A[i * n + j] = get_f(a, i, j);

    for (int64_t i = 0; i < n; i++)
        for (int64_t j = 0; j < n; j++)
            for (int64_t r = 0; r < m; r++)
                ATA[i * n + j] += A[r * n + i] * A[r * n + j];

    wowml_tensor_t ata_t, eigvals, eigvecs;
    int64_t nn_shape[2] = {n, n};
    int64_t n_shape[1] = {n};
    wowml_tensor_from_data(&ata_t, ATA, WOWML_FLOAT64, 2, nn_shape, NULL, 0);
    wowml_tensor_alloc(&eigvals, WOWML_FLOAT64, 1, n_shape);
    wowml_tensor_alloc(&eigvecs, WOWML_FLOAT64, 2, nn_shape);
    wowml_linalg_eigh(&eigvals, &eigvecs, &ata_t);

    /* Sort eigenvalues descending */
    double* evals = (double*)eigvals.data;
    double* V = (double*)eigvecs.data;
    int64_t* order = (int64_t*)wowml_alloc((size_t)n * sizeof(int64_t));
    for (int64_t i = 0; i < n; i++) order[i] = i;
    for (int64_t i = 0; i < n - 1; i++) {
        for (int64_t j = i + 1; j < n; j++) {
            if (evals[order[j]] > evals[order[i]]) {
                int64_t tmp = order[i]; order[i] = order[j]; order[j] = tmp;
            }
        }
    }

    for (int64_t i = 0; i < k; i++) {
        double ev = evals[order[i]];
        double sig = (ev > 0.0) ? sqrt(ev) : 0.0;
        wowml_tensor_set_flat(s, i, sig);
    }
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j < n; j++) {
            set_f(vt, i, j, V[j * n + order[i]]);
        }
    }
    for (int64_t j = 0; j < k; j++) {
        double sig = 0.0;
        wowml_tensor_get_flat(s, j, &sig);
        for (int64_t i = 0; i < m; i++) {
            double acc = 0.0;
            for (int64_t c = 0; c < n; c++) acc += A[i * n + c] * V[c * n + order[j]];
            set_f(u, i, j, (sig > 1e-12) ? (acc / sig) : 0.0);
        }
    }

    wowml_free(order);
    wowml_tensor_free(&eigvals);
    wowml_tensor_free(&eigvecs);
    wowml_free(ATA);
    wowml_free(A);
    return WOWML_OK;
}

wowml_status_t wowml_linalg_trace(double* result, const wowml_tensor_t* a) {
    WOWML_CHECK(result != NULL && a != NULL, WOWML_ERR_NULL_PTR);
    WOWML_CHECK(a->ndim == 2 && a->shape[0] == a->shape[1], WOWML_ERR_INVALID_ARGUMENT);
    double tr = 0.0;
    for (int64_t i = 0; i < a->shape[0]; i++) tr += get_f(a, i, i);
    *result = tr;
    return WOWML_OK;
}

wowml_status_t wowml_linalg_norm(double* result, const wowml_tensor_t* a, int ord) {
    WOWML_CHECK(result != NULL && a != NULL, WOWML_ERR_NULL_PTR);
    double acc = 0.0;
    for (int64_t i = 0; i < a->numel; i++) {
        double v = 0.0;
        wowml_tensor_get_flat(a, i, &v);
        double av = fabs(v);
        if (ord == 1) acc += av;
        else if (ord == 2) acc += v * v;
        else if (ord < 0 && av > acc) acc = av;
    }
    *result = (ord == 2) ? sqrt(acc) : acc;
    return WOWML_OK;
}
