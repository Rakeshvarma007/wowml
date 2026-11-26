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
  printf("=== C Ops & Fused Kernel Tests ===\n");

  int64_t shape[2] = {2, 3};
  wowml_tensor_t a, b, out;
  TEST_CHECK(wowml_tensor_ones(&a, WOWML_FLOAT32, 2, shape) == WOWML_OK);
  TEST_CHECK(wowml_tensor_alloc(&b, WOWML_FLOAT32, 2, shape) == WOWML_OK);
  TEST_CHECK(wowml_tensor_fill(&b, 2.0) == WOWML_OK);
  TEST_CHECK(wowml_tensor_alloc(&out, WOWML_FLOAT32, 2, shape) == WOWML_OK);

  TEST_CHECK(wowml_add(&out, &a, &b) == WOWML_OK);
  double v = 0.0;
  wowml_tensor_get_flat(&out, 0, &v);
  TEST_CHECK(fabs(v - 3.0) < 1e-6);

  TEST_CHECK(wowml_mul(&out, &out, &b) == WOWML_OK);
  wowml_tensor_get_flat(&out, 0, &v);
  TEST_CHECK(fabs(v - 6.0) < 1e-6);

  /* Reduction */
  int64_t s1[1] = {1};
  wowml_tensor_t sum_out;
  TEST_CHECK(wowml_tensor_alloc(&sum_out, WOWML_FLOAT32, 1, s1) == WOWML_OK);
  TEST_CHECK(wowml_sum(&sum_out, &out, -1, 0) == WOWML_OK);
  wowml_tensor_get_flat(&sum_out, 0, &v);
  TEST_CHECK(fabs(v - 36.0) < 1e-5);

  /* Fused linear + bias + relu */
  int64_t wx_shape[2] = {2, 3};
  wowml_tensor_t w, bias, fout;
  int64_t bshape[1] = {2}, oshape[2] = {2, 2};
  TEST_CHECK(wowml_tensor_ones(&w, WOWML_FLOAT32, 2, wx_shape) == WOWML_OK);
  TEST_CHECK(wowml_tensor_ones(&bias, WOWML_FLOAT32, 1, bshape) == WOWML_OK);
  TEST_CHECK(wowml_tensor_alloc(&fout, WOWML_FLOAT32, 2, oshape) == WOWML_OK);
  TEST_CHECK(wowml_fused_linear_bias_relu(&fout, &a, &w, &bias) == WOWML_OK);
  wowml_tensor_get_flat(&fout, 0, &v);
  TEST_CHECK(fabs(v - 4.0) < 1e-5);

  wowml_tensor_free(&fout);
  wowml_tensor_free(&bias);
  wowml_tensor_free(&w);
  wowml_tensor_free(&sum_out);
  wowml_tensor_free(&out);
  wowml_tensor_free(&b);
  wowml_tensor_free(&a);
  printf("All C ops tests passed!\n");
  return 0;
}
