/**
 * error.h — Error codes and error-handling macros.
 */
#ifndef WOWML_ERROR_H
#define WOWML_ERROR_H

typedef enum {
    WOWML_OK = 0,
    WOWML_ERR_NULL_PTR,
    WOWML_ERR_SHAPE_MISMATCH,
    WOWML_ERR_OUT_OF_MEMORY,
    WOWML_ERR_INVALID_DTYPE,
    WOWML_ERR_SINGULAR_MATRIX,
    WOWML_ERR_INVALID_ARGUMENT,
} wowml_status_t;

const char* wowml_status_string(wowml_status_t status);

typedef void (*wowml_error_handler_t)(wowml_status_t status, const char* msg);
void wowml_set_error_handler(wowml_error_handler_t handler);

#endif /* WOWML_ERROR_H */
