#include "wowml/wowml.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define TEST_CHECK(expr) \
    do { if (!(expr)) { fprintf(stderr, "FAILED at line %d: %s\n", __LINE__, #expr); exit(1); } } while (0)

int main(void) {
    printf("=== C Matmul Tests ===\n");

    int64_t sa[2] = {2, 3}, sb[2] = {3, 2}, sc[2] = {2, 2};
    wowml_tensor_t a, b, c;
    TEST_CHECK(wowml_tensor_ones(&a, WOWML_FLOAT32, 2, sa) == WOWML_OK);
    TEST_CHECK(wowml_tensor_alloc(&b, WOWML_FLOAT32, 2, sb) == WOWML_OK);
    TEST_CHECK(wowml_tensor_fill(&b, 2.0) == WOWML_OK);
    TEST_CHECK(wowml_tensor_alloc(&c, WOWML_FLOAT32, 2, sc) == WOWML_OK);

    TEST_CHECK(wowml_matmul(&c, &a, &b) == WOWML_OK);
    double v = 0.0;
    for (int i = 0; i < 4; i++) {
        wowml_tensor_get_flat(&c, i, &v);
        TEST_CHECK(fabs(v - 6.0) < 1e-5);
    }

    /* Dot product */
    int64_t sv[1] = {4};
    wowml_tensor_t v1, v2;
    TEST_CHECK(wowml_tensor_ones(&v1, WOWML_FLOAT64, 1, sv) == WOWML_OK);
    TEST_CHECK(wowml_tensor_alloc(&v2, WOWML_FLOAT64, 1, sv) == WOWML_OK);
    TEST_CHECK(wowml_tensor_fill(&v2, 3.0) == WOWML_OK);
    double dot = 0.0;
    TEST_CHECK(wowml_dot(&dot, &v1, &v2) == WOWML_OK);
    TEST_CHECK(fabs(dot - 12.0) < 1e-9);

    wowml_tensor_free(&v1);
    wowml_tensor_free(&v2);
    wowml_tensor_free(&c);
    wowml_tensor_free(&b);
    wowml_tensor_free(&a);
    printf("All C matmul tests passed!\n");
    return 0;
}
