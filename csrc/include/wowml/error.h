/**
 * error.h — Error codes and assertions for the WowML C kernel.
 */
#ifndef WOWML_ERROR_H
#define WOWML_ERROR_H

typedef enum {
    WOWML_OK = 0,
    WOWML_ERR_NULL_PTR = 1,
    WOWML_ERR_SHAPE_MISMATCH = 2,
    WOWML_ERR_OUT_OF_MEMORY = 3,
    WOWML_ERR_INVALID_DTYPE = 4,
    WOWML_ERR_SINGULAR_MATRIX = 5,
    WOWML_ERR_INVALID_ARGUMENT = 6,
    WOWML_ERR_NOT_IMPLEMENTED = 7
} wowml_status_t;

/** Return a human-readable string for a status code. */
const char* wowml_status_string(wowml_status_t status);

/** Set a custom error handler callback. */
typedef void (*wowml_error_handler_t)(wowml_status_t status, const char* msg);
void wowml_set_error_handler(wowml_error_handler_t handler);

/** Internal assertion macro returning status on error */
#define WOWML_CHECK(expr, status) \
    do {                          \
        if (!(expr)) {            \
            return (status);      \
        }                         \
    } while (0)

#endif /* WOWML_ERROR_H */
