/**
 * memory.h — Memory allocator interface, SIMD alignment, and Arena allocator.
 */
#ifndef WOWML_MEMORY_H
#define WOWML_MEMORY_H

#include <stddef.h>
#include <stdint.h>
#include "error.h"

#define WOWML_DEFAULT_ALIGNMENT 64 /* 64-byte alignment for AVX-512 / AVX2 cache lines */

/* Standard allocator wrappers */
void* wowml_alloc(size_t size);
void* wowml_calloc(size_t count, size_t size);
void  wowml_free(void* ptr);

/* SIMD 64-byte aligned allocation */
void* wowml_aligned_alloc(size_t alignment, size_t size);
void  wowml_aligned_free(void* ptr);

/* Swappable allocator hooks */
typedef void* (*wowml_alloc_fn)(size_t);
typedef void  (*wowml_free_fn)(void*);
void wowml_set_allocator(wowml_alloc_fn alloc_fn, wowml_free_fn free_fn);

/* Lightweight Arena / Memory Pool allocator for temporary buffers */
typedef struct {
    uint8_t* buffer;
    size_t   capacity;
    size_t   offset;
} wowml_arena_t;

wowml_status_t wowml_arena_init(wowml_arena_t* arena, size_t capacity);
void*          wowml_arena_alloc(wowml_arena_t* arena, size_t size, size_t alignment);
void           wowml_arena_reset(wowml_arena_t* arena);
void           wowml_arena_destroy(wowml_arena_t* arena);

#endif /* WOWML_MEMORY_H */
