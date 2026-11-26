#include "wowml/wowml.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define TEST_CHECK(expr)                                                       \
  do {                                                                         \
    if (!(expr)) {                                                             \
      fprintf(stderr, "FAILED at line %d: %s\n", __LINE__, #expr);             \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

int main(void) {
  printf("=== C Linalg Tests ===\n");

  int64_t s2[2] = {2, 2}, s1[1] = {2};
  wowml_tensor_t A, b, x, L;
  TEST_CHECK(wowml_tensor_alloc(&A, WOWML_FLOAT64, 2, s2) == WOWML_OK);
  TEST_CHECK(wowml_tensor_alloc(&b, WOWML_FLOAT64, 1, s1) == WOWML_OK);
  TEST_CHECK(wowml_tensor_alloc(&x, WOWML_FLOAT64, 1, s1) == WOWML_OK);
  TEST_CHECK(wowml_tensor_alloc(&L, WOWML_FLOAT64, 2, s2) == WOWML_OK);

  /* SPD matrix A = [[4, 2], [2, 3]], b = [6, 5] -> x = [1, 1] */
  wowml_tensor_set_flat(&A, 0, 4.0);
  wowml_tensor_set_flat(&A, 1, 2.0);
  wowml_tensor_set_flat(&A, 2, 2.0);
  wowml_tensor_set_flat(&A, 3, 3.0);
  wowml_tensor_set_flat(&b, 0, 6.0);
  wowml_tensor_set_flat(&b, 1, 5.0);

  TEST_CHECK(wowml_linalg_solve(&x, &A, &b) == WOWML_OK);
  double x0 = 0.0, x1 = 0.0;
  wowml_tensor_get_flat(&x, 0, &x0);
  wowml_tensor_get_flat(&x, 1, &x1);
  TEST_CHECK(fabs(x0 - 1.0) < 1e-9 && fabs(x1 - 1.0) < 1e-9);

  double det = 0.0, tr = 0.0;
  TEST_CHECK(wowml_linalg_det(&det, &A) == WOWML_OK);
  TEST_CHECK(fabs(det - 8.0) < 1e-9);
  TEST_CHECK(wowml_linalg_trace(&tr, &A) == WOWML_OK);
  TEST_CHECK(fabs(tr - 7.0) < 1e-9);

  TEST_CHECK(wowml_linalg_cholesky(&L, &A) == WOWML_OK);
  double l00 = 0.0;
  wowml_tensor_get_flat(&L, 0, &l00);
  TEST_CHECK(fabs(l00 - 2.0) < 1e-9);

  wowml_tensor_free(&L);
  wowml_tensor_free(&x);
  wowml_tensor_free(&b);
  wowml_tensor_free(&A);
  printf("All C linalg tests passed!\n");
  return 0;
}
