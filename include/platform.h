/**
 * @file platform.h
 * @brief Platform abstraction layer for cross-platform compatibility
 * 
 * This header provides a unified interface for platform-specific functionality
 * such as threading, memory allocation, and file operations. It abstracts away
 * the differences between Windows, macOS, and Linux to make the codebase portable.
 */

#ifndef PLATFORM_H
#define PLATFORM_H

/* Define feature test macros before any includes */
#ifndef _WIN32
    #define _GNU_SOURCE 1            /* Enable all GNU/Linux extensions */
    #define _POSIX_C_SOURCE 200809L  /* For POSIX.1-2008 features */
    #define _DEFAULT_SOURCE 1        /* For additional compatibility */
#endif

#include <stddef.h>
#include <stdint.h>

/* ============================================================================
 * FORMAT SPECIFIERS FOR CROSS-PLATFORM PRINTF
 * ============================================================================ */

#ifdef PLATFORM_WINDOWS
    // Windows/MinGW doesn't support %zu for size_t
    #define SIZE_T_FMT "%Iu"
    #define SSIZE_T_FMT "%Id" 
#else
    // POSIX systems support %zu
    #define SIZE_T_FMT "%zu"
    #define SSIZE_T_FMT "%zd"
#endif

/* ============================================================================
 * COMPILER AND PLATFORM DETECTION
 * ============================================================================ */

#ifdef _WIN32
    #define PLATFORM_WINDOWS 1
    #ifdef _WIN64
        #define PLATFORM_WINDOWS_64 1
    #else
        #define PLATFORM_WINDOWS_32 1
    #endif
#elif defined(__APPLE__)
    #define PLATFORM_MACOS 1
    #include <TargetConditionals.h>
#elif defined(__linux__)
    #define PLATFORM_LINUX 1
#else
    #define PLATFORM_UNKNOWN 1
#endif

/* ============================================================================
 * THREADING ABSTRACTION
 * ============================================================================ */

#ifdef PLATFORM_WINDOWS
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
    #include <process.h>
    #include <stdio.h>
    #include <synchapi.h>  /* For SRW lock functions */
    
    /* Mutex types */
    typedef CRITICAL_SECTION platform_mutex_t;
    typedef SRWLOCK platform_rwlock_t;
    
    /* Mutex operations */
    #define platform_mutex_init(m) InitializeCriticalSection(m)
    #define platform_mutex_destroy(m) DeleteCriticalSection(m)
    #define platform_mutex_lock(m) EnterCriticalSection(m)
    #define platform_mutex_unlock(m) LeaveCriticalSection(m)
    
    /* Read-write lock operations */
    #define platform_rwlock_init(rw) InitializeSRWLock(rw)
    #define platform_rwlock_destroy(rw) /* No cleanup needed for SRWLOCK */
    #define platform_rwlock_rdlock(rw) AcquireSRWLockShared((PSRWLOCK)(rw))
    #define platform_rwlock_wrlock(rw) AcquireSRWLockExclusive((PSRWLOCK)(rw))
    #define platform_rwlock_unlock_rd(rw) ReleaseSRWLockShared((PSRWLOCK)(rw))
    #define platform_rwlock_unlock_wr(rw) ReleaseSRWLockExclusive((PSRWLOCK)(rw))
    
    /* Atomic operations */
    typedef volatile LONG platform_atomic_int_t;
    #define platform_atomic_load(ptr) InterlockedCompareExchange((LONG*)ptr, 0, 0)
    #define platform_atomic_store(ptr, val) InterlockedExchange((LONG*)ptr, val)
    #define platform_atomic_increment(ptr) InterlockedIncrement((LONG*)ptr)
    #define platform_atomic_decrement(ptr) InterlockedDecrement((LONG*)ptr)
    #define platform_atomic_decrement_fetch(ptr) (InterlockedDecrement((LONG*)ptr))

#else
    /* POSIX systems (Linux, macOS, etc.) */
    #include <unistd.h>
    #include <pthread.h>
    #include <stdatomic.h>
    
    /* Mutex types */
    typedef pthread_mutex_t platform_mutex_t;
    typedef pthread_rwlock_t platform_rwlock_t;
    
    /* Mutex operations */
    #define platform_mutex_init(m) pthread_mutex_init(m, NULL)
    #define platform_mutex_destroy(m) pthread_mutex_destroy(m)
    #define platform_mutex_lock(m) pthread_mutex_lock(m)
    #define platform_mutex_unlock(m) pthread_mutex_unlock(m)
    
    /* Read-write lock operations - cast away const for pthread compatibility */
    #define platform_rwlock_init(rw) pthread_rwlock_init(rw, NULL)
    #define platform_rwlock_destroy(rw) pthread_rwlock_destroy(rw)
    #define platform_rwlock_rdlock(rw) pthread_rwlock_rdlock((pthread_rwlock_t*)(rw))
    #define platform_rwlock_wrlock(rw) pthread_rwlock_wrlock((pthread_rwlock_t*)(rw))
    #define platform_rwlock_unlock_rd(rw) pthread_rwlock_unlock((pthread_rwlock_t*)(rw))
    #define platform_rwlock_unlock_wr(rw) pthread_rwlock_unlock((pthread_rwlock_t*)(rw))
    
    /* Atomic operations */
    typedef atomic_int platform_atomic_int_t;
    #define platform_atomic_load(ptr) atomic_load(ptr)
    #define platform_atomic_store(ptr, val) atomic_store(ptr, val)
    #define platform_atomic_increment(ptr) ((void)atomic_fetch_add(ptr, 1))
    #define platform_atomic_decrement(ptr) (atomic_fetch_sub(ptr, 1) - 1)
    #define platform_atomic_increment_fetch(ptr) (atomic_fetch_add(ptr, 1) + 1)
    #define platform_atomic_decrement_fetch(ptr) (atomic_fetch_sub(ptr, 1) - 1)
#endif

/* ============================================================================
 * MEMORY ALLOCATION ABSTRACTION
 * ============================================================================ */

/**
 * Allocate aligned memory that's suitable for SIMD operations
 * 
 * @param size: Number of bytes to allocate
 * @param alignment: Alignment requirement (must be power of 2)
 * @return: Pointer to aligned memory, or NULL on failure
 */
void* platform_aligned_malloc(size_t size, size_t alignment);

/**
 * Free memory allocated by platform_aligned_malloc
 * 
 * @param ptr: Pointer to free (can be NULL)
 */
void platform_aligned_free(void* ptr);

/* ============================================================================
 * FILE SYSTEM ABSTRACTION
 * ============================================================================ */

#ifdef PLATFORM_WINDOWS
    #define PLATFORM_PATH_SEPARATOR '\\'
    #define PLATFORM_PATH_SEPARATOR_STR "\\"
#else
    #define PLATFORM_PATH_SEPARATOR '/'
    #define PLATFORM_PATH_SEPARATOR_STR "/"
#endif

/**
 * Cross-platform getline implementation
 * Works like POSIX getline() but available on all platforms
 * 
 * @param lineptr: Pointer to buffer (will be allocated/reallocated as needed)
 * @param n: Pointer to buffer size
 * @param stream: File stream to read from
 * @return: Number of characters read, or -1 on error/EOF
 */
#ifdef PLATFORM_WINDOWS
ssize_t platform_getline(char **lineptr, size_t *n, FILE *stream);
#else
#define platform_getline getline
#endif

/* ============================================================================
 * TIMING ABSTRACTION
 * ============================================================================ */

/**
 * Get high-resolution timestamp in microseconds
 * Useful for performance measurement
 * 
 * @return: Current time in microseconds since some epoch
 */
uint64_t platform_get_time_microseconds(void);

/* ============================================================================
 * ERROR HANDLING
 * ============================================================================ */

/**
 * Get the last system error as a string
 * 
 * @return: String describing the last system error
 */
const char* platform_get_last_error(void);

#endif /* PLATFORM_H */
