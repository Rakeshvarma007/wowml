/**
 * memory.h — Memory allocator interface.
 */
#ifndef WOWML_MEMORY_H
#define WOWML_MEMORY_H

#include <stddef.h>

void* wowml_alloc(size_t size);
void* wowml_calloc(size_t count, size_t size);
void  wowml_free(void* ptr);

typedef void* (*wowml_alloc_fn)(size_t);
typedef void  (*wowml_free_fn)(void*);
void wowml_set_allocator(wowml_alloc_fn alloc_fn, wowml_free_fn free_fn);

#endif /* WOWML_MEMORY_H */
