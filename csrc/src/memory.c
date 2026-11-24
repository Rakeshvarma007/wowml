/**
 * memory.c — Memory allocator implementation.
 */
#include "wowml/memory.h"
#include <stdlib.h>
#include <string.h>

static wowml_alloc_fn g_alloc = malloc;
static wowml_free_fn  g_free  = free;

void* wowml_alloc(size_t size) {
    return g_alloc(size);
}

void* wowml_calloc(size_t count, size_t size) {
    void* ptr = g_alloc(count * size);
    if (ptr) memset(ptr, 0, count * size);
    return ptr;
}

void wowml_free(void* ptr) {
    g_free(ptr);
}

void wowml_set_allocator(wowml_alloc_fn alloc_fn, wowml_free_fn free_fn) {
    g_alloc = alloc_fn ? alloc_fn : malloc;
    g_free  = free_fn  ? free_fn  : free;
}
