/**
 * memory.c — Memory allocation, aligned allocation, and arena stubs.
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

void* wowml_aligned_alloc(size_t alignment, size_t size) {
    (void)alignment;
    (void)size;
    /* TODO: Implement 64-byte aligned allocation via _aligned_malloc or posix_memalign */
    return NULL;
}

void wowml_aligned_free(void* ptr) {
    (void)ptr;
    /* TODO: Implement aligned free */
}

wowml_status_t wowml_arena_init(wowml_arena_t* arena, size_t capacity) {
    (void)arena;
    (void)capacity;
    /* TODO: Initialize arena buffer */
    return WOWML_ERR_NOT_IMPLEMENTED;
}

void* wowml_arena_alloc(wowml_arena_t* arena, size_t size, size_t alignment) {
    (void)arena;
    (void)size;
    (void)alignment;
    /* TODO: Bump-allocate from arena */
    return NULL;
}

void wowml_arena_reset(wowml_arena_t* arena) {
    (void)arena;
    /* TODO: Reset offset */
}

void wowml_arena_destroy(wowml_arena_t* arena) {
    (void)arena;
    /* TODO: Free arena buffer */
}
