/**
 * dtype.c — Data type helpers.
 */
#include "wowml/dtype.h"

size_t wowml_dtype_size(wowml_dtype_t dtype) {
    switch (dtype) {
        case WOWML_FLOAT32: return 4;
        case WOWML_FLOAT64: return 8;
        case WOWML_INT32:   return 4;
        case WOWML_INT64:   return 8;
        default:            return 0;
    }
}

const char* wowml_dtype_name(wowml_dtype_t dtype) {
    switch (dtype) {
        case WOWML_FLOAT32: return "float32";
        case WOWML_FLOAT64: return "float64";
        case WOWML_INT32:   return "int32";
        case WOWML_INT64:   return "int64";
        default:            return "unknown";
    }
}
