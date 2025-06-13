/**
 * @file matrix.c
 * @brief Implementation of matrix operations for neural network computations
 * 
 * BEGINNER'S GUIDE TO THIS FILE
 * ============================
 * 
 * This file contains the "engine" that powers all the math in our neural network.
 * Every operation here is optimized for speed because neural networks do
 * millions of these calculations!
 * 
 * PERFORMANCE OPTIMIZATIONS USED:
 * - Memory alignment: Ensures data is arranged for fast CPU/GPU access
 * - OpenMP: Parallel processing across multiple CPU cores
 * - BLAS: Optimized math libraries (when available)
 * - SIMD: Single instruction, multiple data operations
 * 
 * LEARNING TIP: Don't worry about understanding every optimization.
 * Focus on the core algorithm logic first, then explore optimizations later.
 */

#include "matrix.h"    // Matrix structure and function declarations
#include "platform.h"  // Cross-platform memory allocation functions
#include <stdio.h>     // For printf, fprintf (debugging output)
#include <stdlib.h>    // For malloc, free (memory management)
#include <string.h>    // For memcpy, memset (memory operations)
#include <math.h>      // For expf, sqrtf (mathematical functions)
#include <float.h>     // For FLT_MAX, FLT_MIN (floating point limits)
#include <assert.h>    // For assert() (debugging checks)
#include <limits.h>    // For INT_MAX (maximum integer value)
#include <stdint.h>    // For SIZE_MAX (maximum size_t value)
#include <omp.h>       // For OpenMP parallel processing

/**
 * OPTIONAL BLAS SUPPORT
 * ====================
 * BLAS (Basic Linear Algebra Subprograms) is a highly optimized library
 * for matrix operations. If available, we use it for speed.
 * If not available, we fall back to our own implementations.
 */
#ifdef USE_BLAS
#include <cblas.h>     // CBLAS interface for optimized matrix operations
#endif

/**
 * Mathematical constants that might not be defined on all systems
 */
#ifndef M_PI
#define M_PI 3.14159265358979323846  // π (pi) for mathematical calculations
#endif

/**
 * MEMORY ALIGNMENT FOR PERFORMANCE
 * ===============================
 * Modern CPUs work fastest when data is aligned to specific byte boundaries.
 * 32-byte alignment works well for most SIMD (Single Instruction, Multiple Data) operations.
 */
#define ALIGNMENT 32

// ===== BASIC MATRIX OPERATIONS =====

Matrix* create_matrix(int rows, int cols) {
    // Validate input parameters
    if (rows <= 0 || cols <= 0) {
        fprintf(stderr, "Error: create_matrix called with invalid dimensions: %dx%d\n", rows, cols);
        return NULL;
    }
    
    // Check for potential overflow
    if (rows > INT_MAX / cols || (size_t)rows * cols > SIZE_MAX / sizeof(float)) {
        fprintf(stderr, "Error: create_matrix dimensions too large: %dx%d\n", rows, cols);
        return NULL;
    }
    
    Matrix* mat = (Matrix*)malloc(sizeof(Matrix));
    if (!mat) {
        fprintf(stderr, "Error: Failed to allocate memory for matrix structure\n");
        return NULL;
    }
    
    mat->rows = rows;
    mat->cols = cols;
    
    // Align memory for better SIMD performance
    size_t size = rows * cols * sizeof(float);
    mat->data = (float*)platform_aligned_malloc(size, ALIGNMENT);
    
    if (!mat->data) {
        fprintf(stderr, "Error: Failed to allocate memory for matrix data (%dx%d = " SIZE_T_FMT " bytes)\n", 
                rows, cols, size);
        free(mat);
        return NULL;
    }
    
    return mat;
}

Matrix* create_matrix_like(const Matrix* template_matrix) {
    if (!template_matrix) {
        fprintf(stderr, "Error: create_matrix_like called with NULL template\n");
        return NULL;
    }
    return create_matrix(template_matrix->rows, template_matrix->cols);
}

void free_matrix(Matrix* mat) {
    if (mat) {
        platform_aligned_free(mat->data);
        free(mat);
    }
}

void fill_matrix(Matrix* mat, float value) {
    if (!mat || !mat->data) return;
    
    int total = mat->rows * mat->cols;
    #pragma omp parallel for simd
    for (int i = 0; i < total; ++i) {
        mat->data[i] = value;
    }
}

void copy_matrix(Matrix* dst, const Matrix* src) {
    assert(dst && src);
    
    if (dst->rows != src->rows || dst->cols != src->cols) {
        fprintf(stderr, "copy_matrix dimension mismatch: dst(%d,%d) != src(%d,%d)\n", 
                dst->rows, dst->cols, src->rows, src->cols);
        
        // Print detailed debugging information
        fprintf(stderr, "ERROR: Matrix dimension mismatch in copy_matrix\n");
        fprintf(stderr, "  Expected dst dimensions: (%d, %d)\n", dst->rows, dst->cols);
        fprintf(stderr, "  Actual src dimensions: (%d, %d)\n", src->rows, src->cols);
        
        // Analyze the pattern - is it Q,K,V concatenation?
        if (src->cols % dst->cols == 0) {
            int factor = src->cols / dst->cols;
            fprintf(stderr, "  src->cols = %d * %d (factor of %d)\n", factor, dst->cols, factor);
            if (factor == 3) {
                fprintf(stderr, "  DETECTED: 3x factor suggests Q,K,V concatenation in attention!\n");
            }
        }
        
        fflush(stderr);
    }
    assert(dst->rows == src->rows && dst->cols == src->cols);
    
    size_t size = src->rows * src->cols * sizeof(float);
    memcpy(dst->data, src->data, size);
}

// ===== MATRIX MULTIPLICATION =====

void mat_mul(Matrix* out, const Matrix* a, const Matrix* b) {
    assert(a && b && out);
    assert(a->cols == b->rows);
    assert(out->rows == a->rows && out->cols == b->cols);

#ifdef USE_BLAS
    // Use optimized BLAS implementation
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                a->rows, b->cols, a->cols, 1.0f,
                a->data, a->cols, b->data, b->cols, 0.0f,
                out->data, out->cols);
#else
    // Optimized manual implementation with blocking and vectorization
    const int BLOCK_SIZE = 64; // Cache-friendly block size
    
    fill_matrix(out, 0.0f);
    
    #pragma omp parallel for collapse(2)
    for (int ii = 0; ii < a->rows; ii += BLOCK_SIZE) {
        for (int jj = 0; jj < b->cols; jj += BLOCK_SIZE) {
            for (int kk = 0; kk < a->cols; kk += BLOCK_SIZE) {
                // Process block
                int i_end = (ii + BLOCK_SIZE < a->rows) ? ii + BLOCK_SIZE : a->rows;
                int j_end = (jj + BLOCK_SIZE < b->cols) ? jj + BLOCK_SIZE : b->cols;
                int k_end = (kk + BLOCK_SIZE < a->cols) ? kk + BLOCK_SIZE : a->cols;
                
                for (int i = ii; i < i_end; ++i) {
                    for (int k = kk; k < k_end; ++k) {
                        float a_ik = a->data[i * a->cols + k];
                        if (fabsf(a_ik) < 1e-8f) continue; // to Skip near-zero values
                        
                        #pragma omp simd
                        for (int j = jj; j < j_end; ++j) {
                            out->data[i * out->cols + j] += a_ik * b->data[k * b->cols + j];
                        }
                    }
                }
            }
        }
    }
#endif
}

// backward pass for matrix multiplication
void mat_mul_backward(
    Matrix* grad_a_acc, Matrix* grad_b_acc,
    const Matrix* grad_out, const Matrix* a, const Matrix* b
) {
    assert(grad_out->rows == a->rows && grad_out->cols == b->cols);
    assert(grad_a_acc->rows == a->rows && grad_a_acc->cols == a->cols);
    assert(grad_b_acc->rows == b->rows && grad_b_acc->cols == b->cols);

#ifdef USE_BLAS
    // dL/dA += grad_out @ B^T
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                grad_out->rows, b->rows, grad_out->cols, 1.0f,
                grad_out->data, grad_out->cols, b->data, b->cols, 1.0f,
                grad_a_acc->data, grad_a_acc->cols);
    
    // dL/dB += A^T @ grad_out
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                a->cols, grad_out->cols, a->rows, 1.0f,
                a->data, a->cols, grad_out->data, grad_out->cols, 1.0f,
                grad_b_acc->data, grad_b_acc->cols);
#else
    // Manual implementation with optimizations
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < a->rows; ++i) {
        for (int j = 0; j < a->cols; ++j) {
            float sum = 0.0f;
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < grad_out->cols; ++k) {
                sum += grad_out->data[i * grad_out->cols + k] * b->data[j * b->cols + k];
            }
            grad_a_acc->data[i * a->cols + j] += sum;
        }
    }

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < b->rows; ++i) {
        for (int j = 0; j < b->cols; ++j) {
            float sum = 0.0f;
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < a->rows; ++k) {
                sum += a->data[k * a->cols + i] * grad_out->data[k * grad_out->cols + j];
            }
            grad_b_acc->data[i * b->cols + j] += sum;
        }
    }
#endif
}

// ===== ACTIVATION FUNCTIONS =====

// Optimized GELU activation with better numerical stability
void gelu_activation(Matrix* mat) {
    if (!mat || !mat->data) return;
    
    const float sqrt_2_over_pi = 0.7978845608028654f; // sqrt(2/π)
    const float coeff = 0.044715f;
    
    int total = mat->rows * mat->cols;
    #pragma omp parallel for simd
    for (int i = 0; i < total; ++i) {
        float x = mat->data[i];
        float x_cubed = x * x * x;
        float inner = sqrt_2_over_pi * (x + coeff * x_cubed);
        mat->data[i] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

void gelu_activation_backward(Matrix* grad_input_acc, const Matrix* input_forward, const Matrix* grad_output) {
    assert(grad_input_acc->rows == input_forward->rows && grad_input_acc->cols == input_forward->cols);
    assert(grad_output->rows == input_forward->rows && grad_output->cols == input_forward->cols);

    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;
    
    int total = input_forward->rows * input_forward->cols;
    #pragma omp parallel for simd
    for (int i = 0; i < total; ++i) {
        float x = input_forward->data[i];
        float x_squared = x * x;
        float x_cubed = x_squared * x;
        
        float inner = sqrt_2_over_pi * (x + coeff * x_cubed);
        float tanh_val = tanhf(inner);
        float sech_squared = 1.0f - tanh_val * tanh_val;
        
        float d_inner_dx = sqrt_2_over_pi * (1.0f + 3.0f * coeff * x_squared);
        float gelu_prime = 0.5f * (1.0f + tanh_val) + 0.5f * x * sech_squared * d_inner_dx;

        grad_input_acc->data[i] += grad_output->data[i] * gelu_prime;
    }
}

// ===== SOFTMAX AND CROSS-ENTROPY =====

// Numerically stable row-wise softmax
void softmax_row_wise(Matrix* mat) {
    if (!mat || !mat->data) return;
    
    #pragma omp parallel for
    for (int i = 0; i < mat->rows; ++i) {
        float* row = &mat->data[i * mat->cols];
        
        // Find maximum for numerical stability
        float max_val = -FLT_MAX;
        for (int j = 0; j < mat->cols; ++j) {
            if (row[j] > max_val) max_val = row[j];
        }
        
        // Compute exponentials and sum
        float sum_exp = 0.0f;
        #pragma omp simd reduction(+:sum_exp)
        for (int j = 0; j < mat->cols; ++j) {
            row[j] = expf(row[j] - max_val);
            sum_exp += row[j];
        }
        
        // Normalize
        float inv_sum = 1.0f / (sum_exp + 1e-8f); // Add small epsilon
        #pragma omp simd
        for (int j = 0; j < mat->cols; ++j) {
            row[j] *= inv_sum;
        }
    }
}

// softmax cross-entropy with numerical stability
float softmax_cross_entropy_loss_and_backward(
    Matrix* grad_logits_out,
    const Matrix* logits,
    const int* targets_flat,
    int num_tokens
) {
    assert(logits->rows == num_tokens && grad_logits_out->rows == num_tokens);
    assert(logits->cols == grad_logits_out->cols);
    
    int vocab_size = logits->cols;
    double total_loss = 0.0;
    int valid_tokens = 0;

    #pragma omp parallel for reduction(+:total_loss, valid_tokens)
    for (int i = 0; i < num_tokens; ++i) {
        const float* logit_row = &logits->data[i * vocab_size];
        float* grad_row = &grad_logits_out->data[i * vocab_size];
        int target_idx = targets_flat[i];
        
        // Skip padding tokens
        if (target_idx < 0 || target_idx >= vocab_size) {
            memset(grad_row, 0, vocab_size * sizeof(float));
            continue;
        }
        
        // Find maximum for numerical stability
        float max_logit = -FLT_MAX;
        for (int j = 0; j < vocab_size; ++j) {
            if (logit_row[j] > max_logit) max_logit = logit_row[j];
        }
        
        // Compute softmax probabilities
        float sum_exp = 0.0f;
        for (int j = 0; j < vocab_size; ++j) {
            grad_row[j] = expf(logit_row[j] - max_logit);
            sum_exp += grad_row[j];
        }
        
        // Normalize and compute loss/gradients
        float inv_sum = 1.0f / sum_exp;
        float target_prob = grad_row[target_idx] * inv_sum;
        
        // Clamp probability to avoid log(0)
        target_prob = fmaxf(target_prob, 1e-8f);
        total_loss += -logf(target_prob);
        valid_tokens++;
        
        // Compute gradients: softmax(x) - one_hot(target)
        for (int j = 0; j < vocab_size; ++j) {
            grad_row[j] *= inv_sum;
            if (j == target_idx) {
                grad_row[j] -= 1.0f;
            }
        }
    }
    
    return valid_tokens > 0 ? (float)(total_loss / valid_tokens) : 0.0f;
}

// ===== LAYER NORMALIZATION =====

// Forward pass with stored statistics for backward pass
void layer_norm_forward(
    Matrix* output, Matrix* means, Matrix* inv_stddevs,
    const Matrix* input, const Matrix* gain, const Matrix* bias, float eps
) {
    int num_tokens = input->rows;
    int d_model = input->cols;
    
    #pragma omp parallel for
    for (int i = 0; i < num_tokens; ++i) {
        const float* input_row = &input->data[i * d_model];
        float* output_row = &output->data[i * d_model];
        
        // Compute mean
        float mean = 0.0f;
        #pragma omp simd reduction(+:mean)
        for (int j = 0; j < d_model; ++j) {
            mean += input_row[j];
        }
        mean /= d_model;
        means->data[i] = mean;
        
        // Compute variance
        float variance = 0.0f;
        #pragma omp simd reduction(+:variance)
        for (int j = 0; j < d_model; ++j) {
            float diff = input_row[j] - mean;
            variance += diff * diff;
        }
        variance /= d_model;
        
        float inv_stddev = 1.0f / sqrtf(variance + eps);
        inv_stddevs->data[i] = inv_stddev;
        
        // Normalize and scale/shift
        #pragma omp simd
        for (int j = 0; j < d_model; ++j) {
            float normalized = (input_row[j] - mean) * inv_stddev;
            output_row[j] = normalized * (gain ? gain->data[j] : 1.0f) + (bias ? bias->data[j] : 0.0f);
        }
    }
}

// layer normalization backward pass
void layer_norm_backward(
    Matrix* grad_input_acc, Matrix* grad_gain_acc, Matrix* grad_bias_acc,
    const Matrix* grad_output, const Matrix* input_forward,
    const Matrix* gain_forward, const Matrix* means_forward, 
    const Matrix* inv_stddevs_forward, float eps
) {
    // Add null pointer checks for defensive programming
    if (!grad_input_acc || !grad_output || !input_forward || 
        !means_forward || !inv_stddevs_forward) {
        fprintf(stderr, "Error: NULL pointer passed to layer_norm_backward\n");
        return;
    }
    
    // Validate matrix dimensions
    if (grad_input_acc->rows != input_forward->rows || 
        grad_input_acc->cols != input_forward->cols ||
        grad_output->rows != input_forward->rows ||
        grad_output->cols != input_forward->cols) {
        fprintf(stderr, "Error: Matrix dimension mismatch in layer_norm_backward\n");
        return;
    }
    
    int num_tokens = input_forward->rows;
    int d_model = input_forward->cols;
    
    // Accumulate bias gradients (simple sum)
    if (grad_bias_acc) {
        #pragma omp parallel for
        for (int j = 0; j < d_model; ++j) {
            float sum = 0.0f;
            #pragma omp simd reduction(+:sum)
            for (int i = 0; i < num_tokens; ++i) {
                sum += grad_output->data[i * d_model + j];
            }
            grad_bias_acc->data[j] += sum;
        }
    }
    
    // Process each token
    #pragma omp parallel for
    for (int i = 0; i < num_tokens; ++i) {
        const float* grad_out_row = &grad_output->data[i * d_model];
        const float* input_row = &input_forward->data[i * d_model];
        float* grad_input_row = &grad_input_acc->data[i * d_model];
        
        float mean = means_forward->data[i];
        float inv_stddev = inv_stddevs_forward->data[i];
        
        // Compute intermediate sums
        float sum_grad_out = 0.0f;
        float sum_grad_out_centered = 0.0f;
        
        for (int j = 0; j < d_model; ++j) {
            float centered = input_row[j] - mean;
            float normalized = centered * inv_stddev;
            
            sum_grad_out += grad_out_row[j];
            sum_grad_out_centered += grad_out_row[j] * normalized;
            
            // Accumulate gain gradients
            if (grad_gain_acc) {
                grad_gain_acc->data[j] += grad_out_row[j] * normalized;
            }
        }
        
        // Compute input gradients
        float inv_d_model = 1.0f / d_model;
        for (int j = 0; j < d_model; ++j) {
            float centered = input_row[j] - mean;
            // Note: normalized variable not used in this loop iteration,
            // but computed for clarity and potential debugging
            float normalized = centered * inv_stddev;
            (void)normalized; // Suppress unused variable warning
            float gain_val = gain_forward ? gain_forward->data[j] : 1.0f;
            
            float grad_normalized = grad_out_row[j] * gain_val;
            float grad_var = -0.5f * inv_stddev * inv_stddev * inv_stddev * sum_grad_out_centered;
            float grad_mean = -inv_stddev * sum_grad_out * inv_d_model - 2.0f * grad_var * centered * inv_d_model;
            
            grad_input_row[j] += grad_normalized * inv_stddev + grad_var * 2.0f * centered * inv_d_model + grad_mean;
        }
    }
}

// Simple wrapper for layer_norm that doesn't return statistics
void layer_norm(Matrix* out, const Matrix* input, const Matrix* gain, const Matrix* bias, float eps) {
    // Create temporary matrices for means and inv_stddevs (not used)
    Matrix* means = create_matrix(input->rows, 1);
    Matrix* inv_stddevs = create_matrix(input->rows, 1);
    
    layer_norm_forward(out, means, inv_stddevs, input, gain, bias, eps);
    
    // Free temporary matrices
    free_matrix(means);
    free_matrix(inv_stddevs);
}

// ===== UTILITY FUNCTIONS =====

// Add matrices element-wise with optional scaling
void add_matrices(Matrix* result, const Matrix* a, const Matrix* b, float scale_b) {
    assert(result && a && b);
    assert(a->rows == b->rows && a->cols == b->cols);
    assert(result->rows == a->rows && result->cols == a->cols);
    
    int total = a->rows * a->cols;
    #pragma omp parallel for simd
    for (int i = 0; i < total; ++i) {
        result->data[i] = a->data[i] + scale_b * b->data[i];
    }
}

// Scale matrix by constant
void scale_matrix(Matrix* mat, float scale) {
    if (!mat || !mat->data) return;
    
    int total = mat->rows * mat->cols;
    #pragma omp parallel for simd
    for (int i = 0; i < total; ++i) {
        mat->data[i] *= scale;
    }
}

// Compute matrix norm (Frobenius norm)
float matrix_norm(const Matrix* mat) {
    if (!mat || !mat->data) return 0.0f;
    
    double sum_squares = 0.0;
    int total = mat->rows * mat->cols;
    
    #pragma omp parallel for reduction(+:sum_squares)
    for (int i = 0; i < total; ++i) {
        double val = mat->data[i];
        sum_squares += val * val;
    }
    
    return sqrtf((float)sum_squares);
}

// Clip gradients to prevent exploding gradients
void clip_gradients(Matrix* grad, float max_norm) {
    float norm = matrix_norm(grad);
    if (norm > max_norm) {
        float scale = max_norm / norm;
        scale_matrix(grad, scale);
    }
}

// ===== ADDITIONAL BASIC MATRIX OPERATIONS =====

// Matrix addition: out = a + b
void mat_add(Matrix* out, const Matrix* a, const Matrix* b) {
    assert(out && a && b);
    assert(a->rows == b->rows && a->cols == b->cols);
    assert(out->rows == a->rows && out->cols == b->cols);
    
    int total = a->rows * a->cols;
    #pragma omp parallel for simd
    for (int i = 0; i < total; ++i) {
        out->data[i] = a->data[i] + b->data[i];
    }
}

// Matrix scalar multiplication: m *= scalar
void mat_mul_scalar(Matrix* m, float scalar) {
    if (!m || !m->data) return;
    
    int total = m->rows * m->cols;
    #pragma omp parallel for simd
    for (int i = 0; i < total; ++i) {
        m->data[i] *= scalar;
    }
}

// Matrix transpose: out = m^T
void mat_transpose(Matrix* out, const Matrix* m) {
    // Enhanced validation with detailed error messages
    if (!out) {
        fprintf(stderr, "Error: mat_transpose received NULL output matrix\n");
        return;
    }
    if (!m) {
        fprintf(stderr, "Error: mat_transpose received NULL input matrix\n");
        return;
    }
    if (!out->data) {
        fprintf(stderr, "Error: mat_transpose output matrix has NULL data pointer\n");
        return;
    }
    if (!m->data) {
        fprintf(stderr, "Error: mat_transpose input matrix has NULL data pointer\n");
        return;
    }
    
    // Validate dimensions
    if (out->rows != m->cols || out->cols != m->rows) {
        fprintf(stderr, "Error: mat_transpose dimension mismatch\n");
        fprintf(stderr, "  Expected output: %dx%d, but got: %dx%d\n", 
                m->cols, m->rows, out->rows, out->cols);
        return;
    }
    
    // Validate non-negative dimensions
    if (m->rows <= 0 || m->cols <= 0 || out->rows <= 0 || out->cols <= 0) {
        fprintf(stderr, "Error: mat_transpose invalid dimensions - input: %dx%d, output: %dx%d\n",
                m->rows, m->cols, out->rows, out->cols);
        return;
    }
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < m->rows; ++i) {
        for (int j = 0; j < m->cols; ++j) {
            out->data[j * out->cols + i] = m->data[i * m->cols + j];
        }
    }
}

// Create causal mask matrix for transformer attention
Matrix* create_causal_mask(int seq_len) {
    Matrix* mask = create_matrix(seq_len, seq_len);
    if (!mask) return NULL;
    
    // Set upper triangle to -inf (or very large negative number)
    const float neg_inf = -1e9f;
    
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < seq_len; ++j) {
            if (j > i) {
                mask->data[i * seq_len + j] = neg_inf;
            } else {
                mask->data[i * seq_len + j] = 0.0f;
            }
        }
    }
    
    return mask;
}