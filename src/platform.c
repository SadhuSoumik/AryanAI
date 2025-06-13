/**
 * @file platform.c
 * @brief Implementation of platform abstraction layer
 */

#include "platform.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef PLATFORM_WINDOWS
    #include <windows.h>
    #include <malloc.h>
#else
    #include <sys/time.h>
    #include <errno.h>
    #include <unistd.h>
    
    /* For posix_memalign on older systems */
    #ifdef _GNU_SOURCE
        #include <malloc.h>
    #endif
#endif

/* ============================================================================
 * MEMORY ALLOCATION IMPLEMENTATION
 * ============================================================================ */

void* platform_aligned_malloc(size_t size, size_t alignment) {
    /* Ensure alignment is a power of 2 and at least sizeof(void*) */
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        return NULL;
    }
    
    if (alignment < sizeof(void*)) {
        alignment = sizeof(void*);
    }

#ifdef PLATFORM_WINDOWS
    /* Use Windows _aligned_malloc */
    return _aligned_malloc(size, alignment);
    
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
    /* Use C11 aligned_alloc if available */
    /* Note: aligned_alloc requires size to be a multiple of alignment */
    size_t aligned_size = ((size + alignment - 1) / alignment) * alignment;
    return aligned_alloc(alignment, aligned_size);
    
#else
    /* Use POSIX posix_memalign */
    void* ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) == 0) {
        return ptr;
    }
    return NULL;
#endif
}

void platform_aligned_free(void* ptr) {
    if (ptr == NULL) return;
    
#ifdef PLATFORM_WINDOWS
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

/* ============================================================================
 * FILE SYSTEM IMPLEMENTATION
 * ============================================================================ */

#ifdef PLATFORM_WINDOWS
/* Windows implementation of getline() */
ssize_t platform_getline(char **lineptr, size_t *n, FILE *stream) {
    if (lineptr == NULL || n == NULL || stream == NULL) {
        return -1;
    }
    
    size_t pos = 0;
    int c;
    
    /* Allocate initial buffer if needed */
    if (*lineptr == NULL || *n == 0) {
        *n = 128;
        *lineptr = (char*)malloc(*n);
        if (*lineptr == NULL) {
            return -1;
        }
    }
    
    /* Read characters until newline or EOF */
    while ((c = fgetc(stream)) != EOF) {
        /* Ensure we have space for character and null terminator */
        if (pos + 1 >= *n) {
            size_t new_size = *n * 2;
            char* new_ptr = (char*)realloc(*lineptr, new_size);
            if (new_ptr == NULL) {
                return -1;
            }
            *lineptr = new_ptr;
            *n = new_size;
        }
        
        (*lineptr)[pos++] = (char)c;
        
        /* Stop at newline */
        if (c == '\n') {
            break;
        }
    }
    
    /* Handle EOF with no characters read */
    if (pos == 0 && c == EOF) {
        return -1;
    }
    
    /* Null-terminate the string */
    if (pos < *n) {
        (*lineptr)[pos] = '\0';
    }
    
    return (ssize_t)pos;
}
#endif

/* ============================================================================
 * TIMING IMPLEMENTATION
 * ============================================================================ */

uint64_t platform_get_time_microseconds(void) {
#ifdef PLATFORM_WINDOWS
    LARGE_INTEGER frequency, counter;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    
    /* Convert to microseconds */
    return (counter.QuadPart * 1000000ULL) / frequency.QuadPart;
    
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000000ULL + (uint64_t)tv.tv_usec;
#endif
}

/* ============================================================================
 * ERROR HANDLING IMPLEMENTATION
 * ============================================================================ */

const char* platform_get_last_error(void) {
    static char error_buffer[256];
    
#ifdef PLATFORM_WINDOWS
    DWORD error_code = GetLastError();
    FormatMessageA(
        FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL,
        error_code,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        error_buffer,
        sizeof(error_buffer) - 1,
        NULL
    );
    
    /* Remove trailing newline if present */
    size_t len = strlen(error_buffer);
    if (len > 0 && error_buffer[len - 1] == '\n') {
        error_buffer[len - 1] = '\0';
        if (len > 1 && error_buffer[len - 2] == '\r') {
            error_buffer[len - 2] = '\0';
        }
    }
    
    return error_buffer;
    
#else
    strncpy(error_buffer, strerror(errno), sizeof(error_buffer) - 1);
    error_buffer[sizeof(error_buffer) - 1] = '\0';
    return error_buffer;
#endif
}
