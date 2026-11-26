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
  printf("=== C Tensor Tests ===\n");

  int64_t shape[2] = {3, 4};
  wowml_tensor_t a, b, view;
  TEST_CHECK(wowml_tensor_ones(&a, WOWML_FLOAT32, 2, shape) == WOWML_OK);
  TEST_CHECK(a.numel == 12);
  TEST_CHECK(wowml_tensor_is_contiguous(&a) == 1);

  double v = 0.0;
  TEST_CHECK(wowml_tensor_get_flat(&a, 5, &v) == WOWML_OK);
  TEST_CHECK(fabs(v - 1.0) < 1e-6);

  /* Zero-copy transpose */
  TEST_CHECK(wowml_tensor_transpose(&view, &a, 0, 1) == WOWML_OK);
  TEST_CHECK(view.shape[0] == 4 && view.shape[1] == 3);
  TEST_CHECK(wowml_tensor_is_contiguous(&view) == 0);
  TEST_CHECK(wowml_tensor_contiguous(&b, &view) == WOWML_OK);
  TEST_CHECK(wowml_tensor_is_contiguous(&b) == 1);

  /* Zero-copy slice */
  wowml_tensor_t sl;
  TEST_CHECK(wowml_tensor_slice(&sl, &a, 1, 1, 3, 1) == WOWML_OK);
  TEST_CHECK(sl.shape[0] == 3 && sl.shape[1] == 2);

  /* Broadcast shape check */
  int64_t sa[2] = {3, 1}, sb[2] = {1, 4}, so[WOWML_MAX_DIMS];
  int ndim_o = 0;
  TEST_CHECK(wowml_broadcast_shapes(2, sa, 2, sb, &ndim_o, so) == WOWML_OK);
  TEST_CHECK(ndim_o == 2 && so[0] == 3 && so[1] == 4);

  wowml_tensor_free(&sl);
  wowml_tensor_free(&view);
  wowml_tensor_free(&b);
  wowml_tensor_free(&a);
  printf("All C tensor tests passed!\n");
  return 0;
}
