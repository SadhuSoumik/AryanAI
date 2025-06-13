/**
 * @file utils.h
 * @brief Utility functions for memory management, error handling, and mathematical operations
 * 
 * This header file contains essential utility functions used throughout the transformer project.
 * These functions provide safe memory allocation, error reporting, and mathematical operations
 * that are commonly needed in machine learning applications.
 * 
 * 
 * - Activation functions: Mathematical functions used in neural networks
 */

#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>  // For standard input/output functions like printf, fprintf
#include <stdlib.h> // For memory allocation functions like malloc, calloc, free
#include <string.h> // For string manipulation functions like strlen, strcpy

/**
 * @section Memory Management Functions
 * These functions provide "safe" versions of standard memory allocation.
 * They automatically check if allocation succeeded and terminate the program
 * if memory cannot be allocated (which prevents crashes from null pointers).
 */

/**
 * Safely allocates memory with automatic error checking.
 * This is a wrapper around malloc() that ensures the allocation succeeded.
 * If allocation fails, the program will terminate with an error message.
 * 
 * @param size: Number of bytes to allocate
 * @return: Pointer to allocated memory (guaranteed non-NULL)
 * 
 * Example usage:
 *   int* numbers = (int*)safe_malloc(10 * sizeof(int)); // Allocate space for 10 integers
 */
void* safe_malloc(size_t size);

/**
 * 
 * @param num: Number of elements to allocate
 * @param size: Size of each element in bytes
 * @return: Pointer to allocated and zeroed memory (guaranteed non-NULL)
 * 
 * Example usage:
 *   float* matrix = (float*)safe_calloc(100, sizeof(float)); // 100 floats, all set to 0.0
 */
void* safe_calloc(size_t num, size_t size);

/**
 * @section Error Handling Functions
 */

/**
 * Prints an error message and terminates the program.
 * This function should be called when encountering unrecoverable errors.
 * It will print the error message to stderr and exit with status code 1.
 * 
 * @param message: Error message to display before terminating
 * 
 * Example usage:
 *   if (file == NULL) {
 *       die("Failed to open required configuration file");
 *   }
 */
void die(const char* message);

/**
 * @section Mathematical Functions (Activation Functions)
 * These are mathematical functions commonly used in neural networks.
 * They are currently commented out but serve as examples of functions
 * you might want to implement for neural network computations.
 */

// Rectified Linear Unit (ReLU): Returns max(0, x)
// This is one of the most common activation functions in deep learning
// It outputs the input if positive, otherwise outputs zero
// float relu(float x);

// Gaussian Error Linear Unit (GELU): A smooth approximation to ReLU
// GELU is often used in transformer models and provides better gradients
// than ReLU in some cases
// float gelu(float x);

/**
 * @section Usage Notes for Beginners
 * 
 * Memory Management Best Practices:
 * 1. Always use safe_malloc/safe_calloc instead of malloc/calloc
 * 2. For every allocation, ensure there's a corresponding free() call
 * 3. Set pointers to NULL after freeing them to avoid double-free errors
 * 
 * Error Handling Best Practices:
 * 1. Check return values of functions that can fail
 * 2. Use die() for unrecoverable errors (like out of memory)
 * 3. Use fprintf(stderr, ...) for warnings that don't require termination
 * 
 * Common Memory Allocation Patterns:
 * - Single values: int* x = (int*)safe_malloc(sizeof(int));
 * - Arrays: float* arr = (float*)safe_malloc(count * sizeof(float));
 * - Structures: MyStruct* s = (MyStruct*)safe_malloc(sizeof(MyStruct));
 * - Zero-initialized: float* zeros = (float*)safe_calloc(count, sizeof(float));
 */

#endif // UTILS_H