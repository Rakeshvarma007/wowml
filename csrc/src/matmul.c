/**
 * matmul.c — High-speed matrix multiplication: pure C11 tiled+OpenMP + OpenBLAS fast paths.
 */
#include "wowml/matmul.h"
#include "wowml/memory.h"
#include <string.h>

#ifdef WOWML_USE_BLAS
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif

#define TILE_M 64
#define TILE_N 64
#define TILE_K 64

wowml_status_t wowml_matmul_c(wowml_tensor_t* out, const wowml_tensor_t* a, const wowml_tensor_t* b) {
    WOWML_CHECK(out != NULL && a != NULL && b != NULL, WOWML_ERR_NULL_PTR);
    WOWML_CHECK(a->ndim == 2 && b->ndim == 2 && out->ndim == 2, WOWML_ERR_INVALID_ARGUMENT);
    WOWML_CHECK(a->dtype == b->dtype && a->dtype == out->dtype, WOWML_ERR_INVALID_DTYPE);

    int64_t M = a->shape[0];
    int64_t K = a->shape[1];
    int64_t N = b->shape[1];
    WOWML_CHECK(b->shape[0] == K && out->shape[0] == M && out->shape[1] == N, WOWML_ERR_SHAPE_MISMATCH);

    wowml_tensor_t a_cont, b_cont;
    const wowml_tensor_t* ap = a;
    const wowml_tensor_t* bp = b;
    int free_a = 0, free_b = 0;

    if (!wowml_tensor_is_contiguous(a)) {
        WOWML_CHECK(wowml_tensor_contiguous(&a_cont, a) == WOWML_OK, WOWML_ERR_OUT_OF_MEMORY);
        ap = &a_cont;
        free_a = 1;
    }
    if (!wowml_tensor_is_contiguous(b)) {
        WOWML_CHECK(wowml_tensor_contiguous(&b_cont, b) == WOWML_OK, WOWML_ERR_OUT_OF_MEMORY);
        bp = &b_cont;
        free_b = 1;
    }

    if (out->dtype == WOWML_FLOAT32) {
        const float* A = (const float*)ap->data;
        const float* B = (const float*)bp->data;
        float* C = (float*)out->data;
        memset(C, 0, (size_t)(M * N) * sizeof(float));

        #pragma omp parallel for schedule(dynamic) if (M * N * K > 32768)
        for (int i0 = 0; i0 < (int)M; i0 += TILE_M) {
            int64_t imax = ((int64_t)i0 + TILE_M < M) ? ((int64_t)i0 + TILE_M) : M;
            for (int64_t k0 = 0; k0 < K; k0 += TILE_K) {
                int64_t kmax = (k0 + TILE_K < K) ? (k0 + TILE_K) : K;
                for (int64_t j0 = 0; j0 < N; j0 += TILE_N) {
                    int64_t jmax = (j0 + TILE_N < N) ? (j0 + TILE_N) : N;
                    for (int64_t i = i0; i < imax; i++) {
                        float* c_row = C + i * N;
                        for (int64_t k = k0; k < kmax; k++) {
                            float a_ik = A[i * K + k];
                            const float* b_row = B + k * N;
                            for (int64_t j = j0; j < jmax; j++) {
                                c_row[j] += a_ik * b_row[j];
                            }
                        }
                    }
                }
            }
        }
    } else if (out->dtype == WOWML_FLOAT64) {
        const double* A = (const double*)ap->data;
        const double* B = (const double*)bp->data;
        double* C = (double*)out->data;
        memset(C, 0, (size_t)(M * N) * sizeof(double));

        #pragma omp parallel for schedule(dynamic) if (M * N * K > 32768)
        for (int i0 = 0; i0 < (int)M; i0 += TILE_M) {
            int64_t imax = ((int64_t)i0 + TILE_M < M) ? ((int64_t)i0 + TILE_M) : M;
            for (int64_t k0 = 0; k0 < K; k0 += TILE_K) {
                int64_t kmax = (k0 + TILE_K < K) ? (k0 + TILE_K) : K;
                for (int64_t j0 = 0; j0 < N; j0 += TILE_N) {
                    int64_t jmax = (j0 + TILE_N < N) ? (j0 + TILE_N) : N;
                    for (int64_t i = i0; i < imax; i++) {
                        double* c_row = C + i * N;
                        for (int64_t k = k0; k < kmax; k++) {
                            double a_ik = A[i * K + k];
                            const double* b_row = B + k * N;
                            for (int64_t j = j0; j < jmax; j++) {
                                c_row[j] += a_ik * b_row[j];
                            }
                        }
                    }
                }
            }
        }
    } else {
        if (free_a) wowml_tensor_free(&a_cont);
        if (free_b) wowml_tensor_free(&b_cont);
        return WOWML_ERR_INVALID_DTYPE;
    }

    if (free_a) wowml_tensor_free(&a_cont);
    if (free_b) wowml_tensor_free(&b_cont);
    return WOWML_OK;
}

wowml_status_t wowml_matmul_blas(wowml_tensor_t* out, const wowml_tensor_t* a, const wowml_tensor_t* b) {
#ifdef WOWML_USE_BLAS
    WOWML_CHECK(out != NULL && a != NULL && b != NULL, WOWML_ERR_NULL_PTR);
    WOWML_CHECK(a->ndim == 2 && b->ndim == 2 && out->ndim == 2, WOWML_ERR_INVALID_ARGUMENT);
    int M = (int)a->shape[0];
    int K = (int)a->shape[1];
    int N = (int)b->shape[1];
    WOWML_CHECK(b->shape[0] == K && out->shape[0] == M && out->shape[1] == N, WOWML_ERR_SHAPE_MISMATCH);

    wowml_tensor_t a_cont, b_cont;
    const wowml_tensor_t* ap = a;
    const wowml_tensor_t* bp = b;
    int free_a = 0, free_b = 0;
    if (!wowml_tensor_is_contiguous(a)) { wowml_tensor_contiguous(&a_cont, a); ap = &a_cont; free_a = 1; }
    if (!wowml_tensor_is_contiguous(b)) { wowml_tensor_contiguous(&b_cont, b); bp = &b_cont; free_b = 1; }

    if (out->dtype == WOWML_FLOAT32) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f,
                    (const float*)ap->data, K, (const float*)bp->data, N, 0.0f, (float*)out->data, N);
    } else if (out->dtype == WOWML_FLOAT64) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0,
                    (const double*)ap->data, K, (const double*)bp->data, N, 0.0, (double*)out->data, N);
    } else {
        if (free_a) wowml_tensor_free(&a_cont);
        if (free_b) wowml_tensor_free(&b_cont);
        return WOWML_ERR_INVALID_DTYPE;
    }
    if (free_a) wowml_tensor_free(&a_cont);
    if (free_b) wowml_tensor_free(&b_cont);
    return WOWML_OK;
#else
    return wowml_matmul_c(out, a, b);
#endif
}

wowml_status_t wowml_matmul(wowml_tensor_t* out, const wowml_tensor_t* a, const wowml_tensor_t* b) {
#ifdef WOWML_USE_BLAS
    return wowml_matmul_blas(out, a, b);
#else
    return wowml_matmul_c(out, a, b);
#endif
}

wowml_status_t wowml_bmm(wowml_tensor_t* out, const wowml_tensor_t* a, const wowml_tensor_t* b) {
    WOWML_CHECK(out != NULL && a != NULL && b != NULL, WOWML_ERR_NULL_PTR);
    WOWML_CHECK(a->ndim == 3 && b->ndim == 3 && out->ndim == 3, WOWML_ERR_INVALID_ARGUMENT);
    int64_t B = a->shape[0];
    WOWML_CHECK(b->shape[0] == B && out->shape[0] == B, WOWML_ERR_SHAPE_MISMATCH);
    WOWML_CHECK(a->shape[2] == b->shape[1] && out->shape[1] == a->shape[1] && out->shape[2] == b->shape[2],
                WOWML_ERR_SHAPE_MISMATCH);

    size_t es = wowml_dtype_size(a->dtype);
    int64_t a_stride = a->strides[0];
    int64_t b_stride = b->strides[0];
    int64_t o_stride = out->strides[0];

    for (int64_t i = 0; i < B; i++) {
        wowml_tensor_t ai, bi, oi;
        wowml_tensor_from_data(&ai, (uint8_t*)a->data + (size_t)(i * a_stride) * es,
                               a->dtype, 2, a->shape + 1, a->strides + 1, 0);
        wowml_tensor_from_data(&bi, (uint8_t*)b->data + (size_t)(i * b_stride) * es,
                               b->dtype, 2, b->shape + 1, b->strides + 1, 0);
        wowml_tensor_from_data(&oi, (uint8_t*)out->data + (size_t)(i * o_stride) * es,
                               out->dtype, 2, out->shape + 1, out->strides + 1, 0);
        wowml_status_t s = wowml_matmul(&oi, &ai, &bi);
        if (s != WOWML_OK) return s;
    }
    return WOWML_OK;
}

wowml_status_t wowml_matvec(wowml_tensor_t* out, const wowml_tensor_t* a, const wowml_tensor_t* x) {
    WOWML_CHECK(out != NULL && a != NULL && x != NULL, WOWML_ERR_NULL_PTR);
    WOWML_CHECK(a->ndim == 2 && x->ndim == 1 && out->ndim == 1, WOWML_ERR_INVALID_ARGUMENT);
    WOWML_CHECK(a->shape[1] == x->shape[0] && out->shape[0] == a->shape[0], WOWML_ERR_SHAPE_MISMATCH);

    int64_t M = a->shape[0];
    int64_t K = a->shape[1];
    if (a->dtype == WOWML_FLOAT32) {
        const float* A = (const float*)a->data;
        const float* X = (const float*)x->data;
        float* Y = (float*)out->data;
        #pragma omp parallel for if (M * K > 16384)
        for (int i = 0; i < (int)M; i++) {
            float acc = 0.0f;
            for (int64_t k = 0; k < K; k++) acc += A[(int64_t)i * K + k] * X[k];
            Y[i] = acc;
        }
        return WOWML_OK;
    } else if (a->dtype == WOWML_FLOAT64) {
        const double* A = (const double*)a->data;
        const double* X = (const double*)x->data;
        double* Y = (double*)out->data;
        #pragma omp parallel for if (M * K > 16384)
        for (int i = 0; i < (int)M; i++) {
            double acc = 0.0;
            for (int64_t k = 0; k < K; k++) acc += A[(int64_t)i * K + k] * X[k];
            Y[i] = acc;
        }
        return WOWML_OK;
    }
    return WOWML_ERR_INVALID_DTYPE;
}

wowml_status_t wowml_dot(double* result, const wowml_tensor_t* a, const wowml_tensor_t* b) {
    WOWML_CHECK(result != NULL && a != NULL && b != NULL, WOWML_ERR_NULL_PTR);
    WOWML_CHECK(a->ndim == 1 && b->ndim == 1, WOWML_ERR_INVALID_ARGUMENT);
    WOWML_CHECK(a->shape[0] == b->shape[0], WOWML_ERR_SHAPE_MISMATCH);
    WOWML_CHECK(a->dtype == b->dtype, WOWML_ERR_INVALID_DTYPE);

    int64_t n = a->shape[0];
    double acc = 0.0;
    if (a->dtype == WOWML_FLOAT32) {
        const float* pa = (const float*)a->data;
        const float* pb = (const float*)b->data;
        for (int64_t i = 0; i < n; i++) acc += (double)pa[i] * (double)pb[i];
    } else if (a->dtype == WOWML_FLOAT64) {
        const double* pa = (const double*)a->data;
        const double* pb = (const double*)b->data;
        for (int64_t i = 0; i < n; i++) acc += pa[i] * pb[i];
    } else {
        return WOWML_ERR_INVALID_DTYPE;
    }
    *result = acc;
    return WOWML_OK;
}
