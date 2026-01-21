// #include <stdlib.h> // no stdlib in guest

#ifndef NULL
#define NULL ((void*)0)
#endif

// Declare malloc and free, which are provided by the Jolt runtime
typedef unsigned long size_t;
void* malloc(size_t size);
void free(void* ptr);

int* alloc_and_set(int val) {
    int* ptr = (int*)malloc(sizeof(int));
    if (ptr != NULL) {
        *ptr = val;
    }
    return ptr;
}

void free_me(int* ptr) {
    free(ptr);
}
