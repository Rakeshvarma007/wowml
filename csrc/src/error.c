/**
 * error.c — Error handling and status string implementation.
 */
#include "wowml/error.h"
#include <stdio.h>

static wowml_error_handler_t g_error_handler = NULL;

static const char* status_strings[] = {
    "OK",
    "Null pointer",
    "Shape mismatch",
    "Out of memory",
    "Invalid dtype",
    "Singular matrix",
    "Invalid argument",
    "Not implemented"
};

const char* wowml_status_string(wowml_status_t status) {
    if (status < 0 || status > WOWML_ERR_NOT_IMPLEMENTED) {
        return "Unknown";
    }
    return status_strings[status];
}

void wowml_set_error_handler(wowml_error_handler_t handler) {
    g_error_handler = handler;
}
