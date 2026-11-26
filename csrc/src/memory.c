/**
 * memory.c — Memory allocation, 64-byte SIMD alignment, and Arena allocator.
 */
#include "wowml/memory.h"
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <malloc.h>
#endif

static wowml_alloc_fn g_alloc = malloc;
static wowml_free_fn  g_free  = free;

void* wowml_alloc(size_t size) {
    if (size == 0) size = 1;
    return g_alloc(size);
}

void* wowml_calloc(size_t count, size_t size) {
    size_t total = count * size;
    if (total == 0) total = 1;
    void* ptr = g_alloc(total);
    if (ptr) {
        memset(ptr, 0, total);
    }
    return ptr;
}

void wowml_free(void* ptr) {
    if (ptr) {
        g_free(ptr);
    }
}

void wowml_set_allocator(wowml_alloc_fn alloc_fn, wowml_free_fn free_fn) {
    g_alloc = alloc_fn ? alloc_fn : malloc;
    g_free  = free_fn  ? free_fn  : free;
}

void* wowml_aligned_alloc(size_t alignment, size_t size) {
    if (alignment < sizeof(void*)) {
        alignment = sizeof(void*);
    }
    if (size == 0) {
        size = alignment;
    }
#ifdef _WIN32
    return _aligned_malloc(size, alignment);
#else
    void* ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return NULL;
    }
    return ptr;
#endif
}

void wowml_aligned_free(void* ptr) {
    if (!ptr) return;
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

wowml_status_t wowml_arena_init(wowml_arena_t* arena, size_t capacity) {
    WOWML_CHECK(arena != NULL, WOWML_ERR_NULL_PTR);
    WOWML_CHECK(capacity > 0, WOWML_ERR_INVALID_ARGUMENT);

    arena->buffer = (uint8_t*)wowml_aligned_alloc(WOWML_DEFAULT_ALIGNMENT, capacity);
    if (!arena->buffer) {
        arena->capacity = 0;
        arena->offset = 0;
        return WOWML_ERR_OUT_OF_MEMORY;
    }
    arena->capacity = capacity;
    arena->offset = 0;
    return WOWML_OK;
}

void* wowml_arena_alloc(wowml_arena_t* arena, size_t size, size_t alignment) {
    if (!arena || !arena->buffer || size == 0) return NULL;
    if (alignment == 0) alignment = WOWML_DEFAULT_ALIGNMENT;

    size_t mask = alignment - 1;
    size_t aligned_offset = (arena->offset + mask) & ~mask;
    if (aligned_offset + size > arena->capacity) {
        return NULL;
    }
    void* ptr = arena->buffer + aligned_offset;
    arena->offset = aligned_offset + size;
    return ptr;
}

void wowml_arena_reset(wowml_arena_t* arena) {
    if (arena) {
        arena->offset = 0;
    }
}

void wowml_arena_destroy(wowml_arena_t* arena) {
    if (arena) {
        wowml_aligned_free(arena->buffer);
        arena->buffer = NULL;
        arena->capacity = 0;
        arena->offset = 0;
    }
}
