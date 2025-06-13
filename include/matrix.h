#ifndef MATRIX_H
#define MATRIX_H

#include <stddef.h> // For size_t

typedef struct {
    int rows;
    int cols;
    float* data;
} Matrix;

// Matrix creation and destruction
Matrix* create_matrix(int rows, int cols);
Matrix* create_matrix_like(const Matrix* template_matrix); // Create matrix with same dimensions
void free_matrix(Matrix* m);
void fill_matrix(Matrix* m, float val);
void copy_matrix(Matrix* dest, const Matrix* src);
Matrix* slice_matrix_rows(const Matrix* m, int start_row, int num_rows);
Matrix* create_causal_mask(int seq_len); // Added declaration


// Basic operations
void mat_mul(Matrix* out, const Matrix* a, const Matrix* b); // out = a * b
void mat_add(Matrix* out, const Matrix* a, const Matrix* b); // out = a + b
void mat_add_scalar(Matrix* m, float scalar);
void mat_mul_scalar(Matrix* m, float scalar);
void mat_transpose(Matrix* out, const Matrix* m);

// Activation and normalization
void softmax_row_wise(Matrix* m); // Applies softmax to each row
void layer_norm(Matrix* out, const Matrix* input, const Matrix* gain, const Matrix* bias, float eps);
void layer_norm_forward(Matrix* output, Matrix* means, Matrix* inv_stddevs,
                       const Matrix* input, const Matrix* gain, const Matrix* bias, float eps);
void gelu_activation(Matrix* m); // Applies GELU element-wise

// Utility
void print_matrix(const Matrix* m, const char* name);

#endif // MATRIX_H