/**
 * dtype.h — Supported data types and helpers.
 */
#ifndef WOWML_DTYPE_H
#define WOWML_DTYPE_H

#include <stddef.h>

typedef enum {
    WOWML_FLOAT32 = 0,
    WOWML_FLOAT64,
    WOWML_INT32,
    WOWML_INT64,
} wowml_dtype_t;

size_t wowml_dtype_size(wowml_dtype_t dtype);
const char* wowml_dtype_name(wowml_dtype_t dtype);

#endif /* WOWML_DTYPE_H */
