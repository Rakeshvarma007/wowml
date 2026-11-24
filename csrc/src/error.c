/**
 * error.c — Error handling implementation.
 */
#include "wowml/error.h"
#include <stdio.h>
#include <stdlib.h>

static wowml_error_handler_t g_error_handler = NULL;

static const char* status_strings[] = {
    "OK",
    "Null pointer",
    "Shape mismatch",
    "Out of memory",
    "Invalid dtype",
    "Singular matrix",
    "Invalid argument",
};

const char* wowml_status_string(wowml_status_t status) {
    if (status < 0 || status > WOWML_ERR_INVALID_ARGUMENT) return "Unknown";
    return status_strings[status];
}

void wowml_set_error_handler(wowml_error_handler_t handler) {
    g_error_handler = handler;
}
