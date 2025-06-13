# Cross-Platform Compatibility Guide

## Overview
This guide explains how to build and run the C Transformer project on different operating systems, including Windows, macOS, and various Linux distributions.

## Current Platform Dependencies

### GNU/Linux Dependencies
- **pthread library**: Used for thread-safe tokenizer operations
- **OpenMP**: Used for parallel matrix operations
- **GNU extensions**: `_GNU_SOURCE`, `getline()`, `posix_memalign()`
- **GCC compiler**: With GNU-specific flags

### Identified Issues for Windows

#### 1. Threading Library (pthread)
**Problem**: Windows doesn't natively support POSIX threads (pthreads)
**Files affected**: 
- `src/tokenizer.c` - Uses `pthread_rwlock_t`, `pthread_mutex_t`
- `main.c` - Includes `<pthread.h>`
- `include/tokenizer.h` - Uses `pthread_rwlock_t` in struct

**Solutions**:
- **Option A**: Use pthreads-win32 library
- **Option B**: Replace with Windows native threading (CreateThread, etc.)
- **Option C**: Use C11 threads (requires C11 compiler support)

#### 2. Memory Alignment (posix_memalign)
**Problem**: `posix_memalign()` is not available on Windows
**Files affected**: `src/matix.c`

**Solutions**:
- Use `_aligned_malloc()` on Windows
- Use `aligned_alloc()` (C11 standard, if available)
- Implement fallback using manual alignment

#### 3. GNU Extensions
**Problem**: `_GNU_SOURCE`, `getline()` not available on Windows
**Files affected**: 
- `src/tokenizer.c` - Uses `getline()`
- Multiple files use `_GNU_SOURCE`

**Solutions**:
- Implement custom `getline()` for Windows
- Use alternative approaches for file reading

#### 4. Build System
**Problem**: Unix Makefile won't work on Windows
**Solutions**:
- Create Windows batch files or PowerShell scripts
- Use CMake for cross-platform builds
- Provide Visual Studio project files

## Recommended Solutions

### Option 1: Minimal Changes (Recommended)
Keep the current Linux-focused approach and provide Windows build instructions using:
- **MinGW-w64** or **MSYS2** for GCC on Windows
- **pthreads-win32** library for threading support

### Option 2: Cross-Platform Refactoring
Modify the codebase to use cross-platform alternatives:
- Replace pthreads with C11 threads or custom abstraction layer
- Replace GNU extensions with standard C alternatives
- Use CMake for cross-platform building

### Option 3: Docker/Container Approach
Provide Docker containers that work consistently across platforms

## Build Instructions by Platform

### Windows (MinGW-w64/MSYS2)

#### Prerequisites
1. Install MSYS2 from https://www.msys2.org/
2. Install required packages:
```bash
pacman -S mingw-w64-x86_64-gcc
pacman -S mingw-w64-x86_64-make
pacman -S mingw-w64-x86_64-pthreads-w32
```

#### Building
```bash
# In MSYS2 MinGW64 terminal
make CC=gcc LDFLAGS="-lm -lpthread"
```

### Windows (Visual Studio)

#### Prerequisites
1. Install Visual Studio 2019 or later with C++ tools
2. Install vcpkg for package management
3. Install pthreads: `vcpkg install pthreads:x64-windows`

#### Building
Create `CMakeLists.txt` (see CMake section below)

### macOS

#### Prerequisites
```bash
brew install gcc
brew install libomp  # For OpenMP support
```

#### Building
```bash
make CC=gcc-12 LDFLAGS="-lm -lomp"
```

### Linux (Various Distributions)

#### Ubuntu/Debian
```bash
sudo apt-get install build-essential libpthread-stubs0-dev
make
```

#### CentOS/RHEL/Fedora
```bash
sudo yum install gcc make  # or dnf on newer versions
make
```

## CMake Configuration (Cross-Platform Solution)

Create `CMakeLists.txt` for cross-platform building:

```cmake
cmake_minimum_required(VERSION 3.12)
project(c_transformer C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Source files
file(GLOB SOURCES "src/*.c" "main.c")

# Include directories
include_directories(include)

# Create executable
add_executable(c_transformer ${SOURCES})

# Platform-specific configurations
if(WIN32)
    # Windows-specific settings
    target_compile_definitions(c_transformer PRIVATE _CRT_SECURE_NO_WARNINGS)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(PTHREAD REQUIRED pthread)
    target_link_libraries(c_transformer ${PTHREAD_LIBRARIES})
    target_compile_options(c_transformer PRIVATE ${PTHREAD_CFLAGS_OTHER})
else()
    # Unix-like systems
    target_compile_definitions(c_transformer PRIVATE _GNU_SOURCE)
    find_package(Threads REQUIRED)
    target_link_libraries(c_transformer Threads::Threads)
endif()

# Math library
target_link_libraries(c_transformer m)

# OpenMP (optional)
find_package(OpenMP)
if(OpenMP_C_FOUND)
    target_link_libraries(c_transformer OpenMP::OpenMP_C)
endif()

# Compiler flags
if(CMAKE_C_COMPILER_ID STREQUAL "GNU" OR CMAKE_C_COMPILER_ID STREQUAL "Clang")
    target_compile_options(c_transformer PRIVATE 
        -Wall -Wextra -O3 
        -Wno-missing-braces 
        -Wno-unused-function 
        -Wno-unused-parameter
    )
endif()
```

## Code Modifications for Better Portability

### 1. Threading Abstraction
Create `src/platform.h` to abstract threading:

```c
#ifndef PLATFORM_H
#define PLATFORM_H

#ifdef _WIN32
    #include <windows.h>
    typedef CRITICAL_SECTION mutex_t;
    typedef SRWLOCK rwlock_t;
    #define mutex_init(m) InitializeCriticalSection(m)
    #define mutex_destroy(m) DeleteCriticalSection(m)
    #define mutex_lock(m) EnterCriticalSection(m)
    #define mutex_unlock(m) LeaveCriticalSection(m)
#else
    #include <pthread.h>
    typedef pthread_mutex_t mutex_t;
    typedef pthread_rwlock_t rwlock_t;
    #define mutex_init(m) pthread_mutex_init(m, NULL)
    #define mutex_destroy(m) pthread_mutex_destroy(m)
    #define mutex_lock(m) pthread_mutex_lock(m)
    #define mutex_unlock(m) pthread_mutex_unlock(m)
#endif

#endif // PLATFORM_H
```

### 2. Memory Alignment Abstraction
Create portable memory alignment:

```c
void* aligned_malloc(size_t size, size_t alignment) {
#ifdef _WIN32
    return _aligned_malloc(size, alignment);
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
    return aligned_alloc(alignment, size);
#else
    void* ptr;
    if (posix_memalign(&ptr, alignment, size) == 0) {
        return ptr;
    }
    return NULL;
#endif
}

void aligned_free(void* ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}
```

### 3. getline() Replacement
For Windows compatibility, implement custom getline():

```c
#ifdef _WIN32
ssize_t getline(char **lineptr, size_t *n, FILE *stream) {
    // Custom implementation for Windows
    // ... (implementation details)
}
#endif
```

## Testing Cross-Platform Builds

### Automated Testing
Consider using GitHub Actions or similar CI/CD to test builds on:
- Ubuntu 20.04/22.04
- macOS (latest)
- Windows Server 2019/2022

### Manual Testing
1. Verify compilation succeeds without errors
2. Test basic functionality (tokenizer, matrix operations)
3. Test file I/O operations
4. Verify threading works correctly

## Performance Considerations

### Windows-Specific
- Consider using Intel MKL or OpenBLAS for optimized matrix operations
- Windows Subsystem for Linux (WSL) might provide better performance for GNU-specific code

### macOS-Specific
- Use Accelerate framework for optimized BLAS operations
- Consider Metal Performance Shaders for GPU acceleration

## Conclusion

For immediate Windows compatibility with minimal code changes:
1. Use MSYS2 with MinGW-w64
2. Install pthreads-win32
3. Use the existing Makefile with minor modifications

For long-term maintainability:
1. Implement the CMake build system
2. Add platform abstraction layers
3. Replace GNU-specific extensions with portable alternatives
