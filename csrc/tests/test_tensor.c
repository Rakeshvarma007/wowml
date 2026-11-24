/**
 * test_tensor.c — Basic tensor tests.
 */
#include "wowml/wowml.h"
#include <stdio.h>
#include <assert.h>

int main(void) {
    printf("=== Tensor Tests ===\n");

    wowml_tensor_t t;
    int64_t shape[] = {3, 4};
    wowml_status_t s = wowml_tensor_zeros(&t, WOWML_FLOAT32, 2, shape);
    assert(s == WOWML_OK);
    assert(t.numel == 12);
    assert(t.ndim == 2);
    wowml_tensor_print(&t);
    wowml_tensor_free(&t);

    printf("All tensor tests passed.\n");
    return 0;
}
