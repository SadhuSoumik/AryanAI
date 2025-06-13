/**
 * @file utils.c
 * @brief Implementation of utility functions for memory management and error handling
 * 
 * This file contains the implementations of essential utility functions that provide
 * safe memory allocation and error handling throughout the transformer project.
 * These functions are critical for preventing common C programming errors like
 * null pointer dereferencing and memory leaks.
 */

#include "utils.h"
#include <math.h> // For mathematical functions like erf, tanh, etc.

/**
 * Safe memory allocation with automatic error checking.
 * This function wraps the standard malloc() function and adds error checking.
 * If memory allocation fails (returns NULL), the program terminates gracefully.
 * 
 * Why this is important:
 * - malloc() can fail if system runs out of memory
 * - Failing to check malloc() return value leads to crashes
 * - This wrapper ensures we never continue with NULL pointers
 * 
 * @param size: Number of bytes to allocate
 * @return: Valid pointer to allocated memory (never NULL)
 */
void* safe_malloc(size_t size) {
    // Attempt to allocate memory using standard malloc
    void* ptr = malloc(size);
    
    // Check if allocation succeeded
    if (ptr == NULL) {
        // If allocation failed, terminate the program with error message
        die("Memory allocation failed in safe_malloc");
    }
    
    // Return the valid pointer
    return ptr;
}

/**
 * Safe zero-initialized memory allocation with automatic error checking.
 * This function wraps calloc() which allocates memory and initializes it to zero.
 * This is particularly useful for arrays and structures that should start clean.
 * 
 * Difference from safe_malloc:
 * - malloc: allocates uninitialized memory (contains random data)
 * - calloc: allocates zero-initialized memory (all bytes set to 0)
 * 
 * @param num: Number of elements to allocate
 * @param size: Size of each element in bytes
 * @return: Valid pointer to zero-initialized memory (never NULL)
 */
void* safe_calloc(size_t num, size_t size) {
    // Allocate zero-initialized memory
    void* ptr = calloc(num, size);
    
    // Check if allocation succeeded
    if (ptr == NULL) {
        // If allocation failed, terminate with error message
        die("Memory allocation failed in safe_calloc");
    }
    
    // Return the valid pointer to zeroed memory
    return ptr;
}

/**
 * Terminates the program with an error message.
 * This function should be called when encountering unrecoverable errors
 * that prevent the program from continuing safely.
 * 
 * What this function does:
 * 1. Prints the error message to stderr (standard error stream)
 * 2. If applicable, prints system error details (via perror)
 * 3. Exits the program with failure status code
 * 
 * @param message: Descriptive error message to display
 */
void die(const char* message) {
    // Print error message and any system error details to stderr
    perror(message);
    
    // Exit the program with failure status (1 indicates error)
    // EXIT_FAILURE is a standard constant for error exit status
    exit(EXIT_FAILURE);
}

/**
 * @section Commented Activation Functions
 * 
 * The following functions are examples of mathematical activation functions
 * commonly used in neural networks. They are commented out but serve as
 * templates for implementing activation functions when needed.
 */

/**
 * Gaussian Error Linear Unit (GELU) activation function.
 * GELU is a smooth activation function that's commonly used in transformer models.
 * It's smoother than ReLU and can provide better gradients for training.
 * 
 * Mathematical formula: GELU(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
 * 
 * Uncomment to use:
 */
// float gelu(float x) {
//     // GELU approximation using tanh (faster than erf-based version)
//     return 0.5f * x * (1.0f + tanhf(sqrtf(2.0f / M_PI) * (x + 0.044715f * powf(x, 3.0f))));
// }

/**
 * Rectified Linear Unit (ReLU) activation function.
 * ReLU is the most common activation function in deep learning.
 * It's simple, fast, and helps with the vanishing gradient problem.
 * 
 * Mathematical formula: ReLU(x) = max(0, x)
 * 
 * Uncomment to use:
 */
// float relu(float x) {
//     return (x > 0.0f) ? x : 0.0f;
// }