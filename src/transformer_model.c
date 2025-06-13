/**
 * @file transformer_model.c
 * @brief Core implementation of a Transformer neural network model for language models.
 * 
 * KEY CONCEPTS FOR Devlets:
 * ---------------------------
 * 1. TOKENS: Words or subwords that the model processes (e.g., "hello" might be one token)
 * 2. EMBEDDINGS: Mathematical representations of tokens as vectors of numbers
 * 3. ATTENTION: A mechanism that helps the model understand relationships between tokens
 * 4. LAYERS: The model is built from multiple processing layers stacked together
 * 5. FORWARD PASS: Processing input through the model to get predictions
 * 6. BACKWARD PASS: Computing gradients to train the model (learning from mistakes)
 * 
 * TRANSFORMER ARCHITECTURE OVERVIEW:
 * ---------------------------------
 * Input Text -> Token IDs -> Embeddings -> Transformer Layers -> Output Probabilities
 * 
 * Each transformer layer contains:
 * - Multi-Head Attention (understands relationships between tokens)
 * - Feed-Forward Network (processes each token independently)
 * - Layer Normalization (stabilizes training)
 * - Residual Connections (helps with training deep networks)
 * 
 * This file implements both training (learning from data) and inference (generating text).
 */

#include "transformer_model.h"
#include "utils.h"
#include <stdio.h> // For file ops, printf
#include <stdint.h> // For uint32_t in file format
#include <math.h>  // For sqrtf in grad clipping
#include <assert.h>
#include <float.h> // For FLT_MAX in causal mask

/* ==============================================
 * KV CACHE HELPER FUNCTIONS
 * =====================================================
 * These functions support efficient autoregressive generation by implementing
 * Key-Value caching to avoid recomputing attention for previously processed tokens.
 */

/**
 * Copies a row from a source matrix to a specific row in the destination matrix.
 * it's used to update KV caches with new token representations.
 * 
 * @param dst_matrix: Destination matrix (e.g., key_cache or value_cache)
 * @param dst_row: Row index in destination matrix to update
 * @param src_matrix: Source matrix (e.g., computed K or V for current token)
 * @param src_row: Row index in source matrix to copy from
 */
static void copy_row(Matrix* dst_matrix, int dst_row, const Matrix* src_matrix, int src_row) {
    if (!dst_matrix || !src_matrix) {
        fprintf(stderr, "Error: NULL pointer in copy_row\n");
        return;
    }
    
    if (dst_row >= dst_matrix->rows || src_row >= src_matrix->rows) {
        fprintf(stderr, "Error: Row index out of bounds in copy_row\n");
        return;
    }
    
    if (dst_matrix->cols != src_matrix->cols) {
        fprintf(stderr, "Error: Column dimension mismatch in copy_row\n");
        return;
    }
    
    // Copy the entire row
    int cols = dst_matrix->cols;
    for (int j = 0; j < cols; j++) {
        dst_matrix->data[dst_row * cols + j] = src_matrix->data[src_row * cols + j];
    }
}

/**
 * Computes attention using cached K, V matrices and current query.
 * This is the core function for efficient autoregressive generation.
 * 
 * @param output: Output attention result (1 x d_model)
 * @param q: Query matrix for current token (1 x d_model)
 * @param cached_k: Cached key matrix (max_seq_len x d_model)
 * @param cached_v: Cached value matrix (max_seq_len x d_model)
 * @param valid_seq_len: Number of valid tokens in cache (including current)
 */
static void compute_cached_attention(Matrix* output, const Matrix* q, 
                                   const Matrix* cached_k, const Matrix* cached_v, 
                                   int valid_seq_len) {
    if (!output || !q || !cached_k || !cached_v) {
        fprintf(stderr, "Error: NULL pointer in compute_cached_attention\n");
        return;
    }
    
    if (valid_seq_len <= 0 || valid_seq_len > cached_k->rows) {
        fprintf(stderr, "Error: Invalid sequence length in compute_cached_attention\n");
        return;
    }
    
    int d_model = q->cols;
    if (d_model != cached_k->cols || d_model != cached_v->cols || d_model != output->cols) {
        fprintf(stderr, "Error: Dimension mismatch in compute_cached_attention\n");
        return;
    }
    
    // Create a view of the valid portion of cached K and V
    Matrix* valid_k = create_matrix(valid_seq_len, d_model);
    Matrix* valid_v = create_matrix(valid_seq_len, d_model);
    
    // Copy valid rows from cache
    for (int i = 0; i < valid_seq_len; i++) {
        for (int j = 0; j < d_model; j++) {
            valid_k->data[i * d_model + j] = cached_k->data[i * d_model + j];
            valid_v->data[i * d_model + j] = cached_v->data[i * d_model + j];
        }
    }
    
    // Compute attention scores: Q @ K^T
    Matrix* k_t = create_matrix(d_model, valid_seq_len);
    mat_transpose(k_t, valid_k);
    
    Matrix* attention_scores = create_matrix(1, valid_seq_len);
    mat_mul(attention_scores, q, k_t);
    
    // Scale by sqrt(d_model)
    float scale = 1.0f / sqrtf((float)d_model);
    mat_mul_scalar(attention_scores, scale);
    
    // Apply causal masking - current token can only attend to previous + current tokens
    // For autoregressive generation, all positions up to current are valid
    // (no masking needed since we only have valid_seq_len tokens)
    
    // Apply softmax to attention scores
    softmax_row_wise(attention_scores);
    
    // Compute final output: attention_scores @ V
    mat_mul(output, attention_scores, valid_v);
    
    // Cleanup
    free_matrix(valid_k);
    free_matrix(valid_v);
    free_matrix(k_t);
    free_matrix(attention_scores);
}

/* ===================================================
 * FORWARD DECLARATIONS FOR INTERNAL FUNCTIONS
 * ========================================================
 * These functions are used internally for the backward pass (training).
 * They compute gradients - the mathematical derivatives that tell us how to
 * update the model's parameters to improve performance.
 */

/**
 * Computes gradients for a linear layer (matrix multiplication + optional bias).
 * Linear layers are the building blocks of neural networks - they transform
 * input vectors through learned weight matrices.
 * 
 * MATHEMATICAL BACKGROUND:
 * A linear layer computes: Y = X * W + B
 * where X=input, W=weights, B=bias, Y=output
 * 
 * During backpropagation, we need gradients:
 * - dL/dX (how loss changes with respect to input)
 * - dL/dW (how loss changes with respect to weights)
 * - dL/dB (how loss changes with respect to bias)
 */
static void linear_layer_backward(Matrix* grad_input_acc, Matrix* grad_W_acc, Matrix* grad_b_acc,
                                  const Matrix* grad_output, const Matrix* input_forward, const Matrix* W_forward);

/**
 * Computes gradients for the attention mechanism.
 * Attention is the key innovation of transformers - it allows the model to
 * "pay attention" to different parts of the input sequence when processing each token.
 */
static void attention_layer_backward(
    Matrix* grad_input_acc,           // Output: dL/d(input to attention block)
    AttentionWeights* grad_attn_weights_acc, // Output: gradients for attention weights
    const Matrix* grad_output,        // Input: dL/d(output of attention block)
    const Matrix* input_forward,      // Input: stored input from forward pass
    const AttentionWeights* weights_forward, // Input: attention weights from forward pass
    const TransformerConfig* config
);

/**
 * Computes gradients for the feed-forward network.
 * Feed-forward networks process each token independently through two linear
 * transformations with a non-linear activation (GELU) in between.
 */
static void ffn_layer_backward(
    Matrix* grad_input_acc, // dL/d(input to FFN block)
    Matrix* grad_W1_acc, Matrix* grad_W2_acc, // Grads for weights
    Matrix* grad_b1_acc, Matrix* grad_b2_acc, // Grads for biases (optional)
    const Matrix* grad_output, // dL/d(output of FFN block)
    const Matrix* input_forward, // Input to FFN block (e.g. output of LN2)
    const Matrix* ffn_hidden_forward, // Output of GELU(input @ W1 + b1)
    const Matrix* W1_forward, const Matrix* W2_forward, // Weights
    const TransformerConfig* config
);

/**
 * Computes gradients for the embedding layer.
 * Embeddings convert discrete token IDs into continuous vector representations
 * that the neural network can process effectively.
 */
static void embedding_layer_backward(Matrix* grad_token_emb_acc, Matrix* grad_pos_emb_acc,
                                     const Matrix* grad_output_emb_sum, // dL/d(token_emb + pos_emb)
                                     const Matrix* input_token_ids, // num_tokens x 1 or flattened batch_size*seq_len
                                     int current_seq_len);

/* ============================================================================
 * TRANSFORMER WEIGHTS MANAGEMENT
 * ============================================================================
 * These functions handle creating, destroying, and initializing the model's
 * learnable parameters (weights and biases).
 */

/**
 * Creates and allocates memory for all transformer model weights.
 * 
 * WEIGHT ORGANIZATION:
 * -------------------
 * 1. Token Embeddings: Convert token IDs to vectors
 * 2. Positional Embeddings: Add position information to tokens
 * 3. Layer Weights: Multiple transformer layers, each containing:
 *    - Attention weights (Q, K, V, O projections)
 *    - Feed-forward weights (two linear transformations)
 *    - Layer normalization parameters (gamma and beta)
 * 4. Final Layer Norm: Normalizes before output projection
 * 5. Output Projection: Maps hidden state to vocabulary probabilities
 * 
 * @param config: Model configuration specifying dimensions and layer count
 * @return: Newly allocated TransformerWeights structure, or NULL on failure
 */
TransformerWeights* create_transformer_weights(const TransformerConfig* config) {
    TransformerWeights* weights = (TransformerWeights*)safe_malloc(sizeof(TransformerWeights));
    weights->config = config;
    
    // Token and positional embeddings
    weights->token_embeddings = create_matrix(config->vocab_size, config->d_model);
    weights->positional_embeddings = create_matrix(config->max_seq_len, config->d_model);
    
    // Layer weights
    weights->layers = (TransformerLayerWeights*)safe_malloc(config->n_layers * sizeof(TransformerLayerWeights));
    
    for (int i = 0; i < config->n_layers; i++) {
        // Attention weights
        weights->layers[i].attn_weights.q_proj = create_matrix(config->d_model, config->d_model);
        weights->layers[i].attn_weights.k_proj = create_matrix(config->d_model, config->d_model);
        weights->layers[i].attn_weights.v_proj = create_matrix(config->d_model, config->d_model);
        weights->layers[i].attn_weights.o_proj = create_matrix(config->d_model, config->d_model);
        
        // Feed-forward weights
        weights->layers[i].ff_weights.fc1 = create_matrix(config->d_model, config->d_ff);
        weights->layers[i].ff_weights.fc2 = create_matrix(config->d_ff, config->d_model);
        
        // Layer norm weights
        weights->layers[i].ln1_gamma = create_matrix(1, config->d_model);
        weights->layers[i].ln1_beta = create_matrix(1, config->d_model);
        weights->layers[i].ln2_gamma = create_matrix(1, config->d_model);
        weights->layers[i].ln2_beta = create_matrix(1, config->d_model);
    }
    
    // Final layer norm and output projection
    weights->final_ln_gamma = create_matrix(1, config->d_model);
    weights->final_ln_beta = create_matrix(1, config->d_model);
    weights->output_projection = create_matrix(config->d_model, config->vocab_size);
    
    return weights;
}

/**
 * Frees all memory associated with transformer model weights.
 * This is crucial to prevent memory leaks - always call this for any allocated
 * TransformerWeights structure when it's no longer needed.
 * 
 * @param weights: The TransformerWeights to free
 */
void free_transformer_weights(TransformerWeights* weights) {
    if (!weights) return;
    
    free_matrix(weights->token_embeddings);
    free_matrix(weights->positional_embeddings);
    
    for (int i = 0; i < weights->config->n_layers; i++) {
        free_matrix(weights->layers[i].attn_weights.q_proj);
        free_matrix(weights->layers[i].attn_weights.k_proj);
        free_matrix(weights->layers[i].attn_weights.v_proj);
        free_matrix(weights->layers[i].attn_weights.o_proj);
        
        free_matrix(weights->layers[i].ff_weights.fc1);
        free_matrix(weights->layers[i].ff_weights.fc2);
        
        free_matrix(weights->layers[i].ln1_gamma);
        free_matrix(weights->layers[i].ln1_beta);
        free_matrix(weights->layers[i].ln2_gamma);
        free_matrix(weights->layers[i].ln2_beta);
    }
    
    free(weights->layers);
    free_matrix(weights->final_ln_gamma);
    free_matrix(weights->final_ln_beta);
    free_matrix(weights->output_projection);
    free(weights);
}

/**
 * Initializes transformer model weights to dummy values.
 * This is useful for testing and debugging - it allows you to create a model
 * with predictable, constant values.
 * 
 * @param weights: The TransformerWeights to initialize
 */
void init_dummy_weights(TransformerWeights* weights) {
    if (!weights) return;
    
    // Initialize with small random values (for now, use constant values)
    fill_matrix(weights->token_embeddings, 0.01f);
    fill_matrix(weights->positional_embeddings, 0.01f);
    
    for (int i = 0; i < weights->config->n_layers; i++) {
        fill_matrix(weights->layers[i].attn_weights.q_proj, 0.01f);
        fill_matrix(weights->layers[i].attn_weights.k_proj, 0.01f);
        fill_matrix(weights->layers[i].attn_weights.v_proj, 0.01f);
        fill_matrix(weights->layers[i].attn_weights.o_proj, 0.01f);
        
        fill_matrix(weights->layers[i].ff_weights.fc1, 0.01f);
        fill_matrix(weights->layers[i].ff_weights.fc2, 0.01f);
        
        // Initialize layer norm gamma to 1, beta to 0
        fill_matrix(weights->layers[i].ln1_gamma, 1.0f);
        fill_matrix(weights->layers[i].ln1_beta, 0.0f);
        fill_matrix(weights->layers[i].ln2_gamma, 1.0f);
        fill_matrix(weights->layers[i].ln2_beta, 0.0f);
    }
    
    fill_matrix(weights->final_ln_gamma, 1.0f);
    fill_matrix(weights->final_ln_beta, 0.0f);
    fill_matrix(weights->output_projection, 0.01f);
}

/**
 * Saves transformer model weights to a file.
 * This is a placeholder function - the actual implementation would serialize
 * the weights to a binary or text format for storage.
 * 
 * @param weights: The TransformerWeights to save
 * @param filepath: The file path where weights should be saved
 * @return: 0 on success, non-zero on failure
 */
// Helper function to save a matrix to file
static int save_matrix_to_file(FILE* file, const Matrix* mat, const char* name);
// Helper function to load a matrix from file
static int load_matrix_from_file(FILE* file, Matrix** mat, const char* name);

static int save_matrix_to_file(FILE* file, const Matrix* mat, const char* name) {
    if (!mat) {
        printf("Warning: Matrix %s is NULL, skipping\n", name);
        int zero = 0;
        return fwrite(&zero, sizeof(int), 1, file) == 1; // Write 0 to indicate NULL
    }
    
    // Write 1 to indicate matrix exists
    int one = 1;
    if (fwrite(&one, sizeof(int), 1, file) != 1) return 0;
    
    // Write dimensions
    if (fwrite(&mat->rows, sizeof(int), 1, file) != 1) return 0;
    if (fwrite(&mat->cols, sizeof(int), 1, file) != 1) return 0;
    
    // Write data
    size_t size = (size_t)mat->rows * mat->cols;
    if (fwrite(mat->data, sizeof(float), size, file) != size) return 0;
    
    return 1;
}

int save_transformer_weights(const TransformerWeights* weights, const char* filepath) {
    if (!weights || !filepath) {
        printf("Error: NULL pointer passed to save_transformer_weights\n");
        return 1;
    }
    
    FILE* file = fopen(filepath, "wb");
    if (!file) {
        printf("Error: Cannot open file %s for writing\n", filepath);
        return 1;
    }
    
    // Write magic number and version for file format validation
    const uint32_t magic_number = 0x54524E53; // "TRNS" in ASCII
    const uint32_t version = 1;
    if (fwrite(&magic_number, sizeof(uint32_t), 1, file) != 1 ||
        fwrite(&version, sizeof(uint32_t), 1, file) != 1) {
        printf("Error: Failed to write file header\n");
        fclose(file);
        return 1;
    }
    
    const TransformerConfig* config = weights->config;
    
    // Save config first
    if (fwrite(config, sizeof(TransformerConfig), 1, file) != 1) {
        printf("Error: Failed to write config\n");
        fclose(file);
        return 1;
    }
    
    // Save token embeddings
    if (!save_matrix_to_file(file, weights->token_embeddings, "token_embeddings")) {
        printf("Error: Failed to save token_embeddings\n");
        fclose(file);
        return 1;
    }
    
    // Save positional embeddings
    if (!save_matrix_to_file(file, weights->positional_embeddings, "positional_embeddings")) {
        printf("Error: Failed to save positional_embeddings\n");
        fclose(file);
        return 1;
    }
    
    // Save layer weights
    for (int l = 0; l < config->n_layers; l++) {
        const TransformerLayerWeights* layer = &weights->layers[l];
        
        // Attention weights
        if (!save_matrix_to_file(file, layer->attn_weights.q_proj, "q_proj") ||
            !save_matrix_to_file(file, layer->attn_weights.k_proj, "k_proj") ||
            !save_matrix_to_file(file, layer->attn_weights.v_proj, "v_proj") ||
            !save_matrix_to_file(file, layer->attn_weights.o_proj, "o_proj")) {
            printf("Error: Failed to save attention weights for layer %d\n", l);
            fclose(file);
            return 1;
        }
        
        // Feed forward weights
        if (!save_matrix_to_file(file, layer->ff_weights.fc1, "fc1") ||
            !save_matrix_to_file(file, layer->ff_weights.fc2, "fc2")) {
            printf("Error: Failed to save feed forward weights for layer %d\n", l);
            fclose(file);
            return 1;
        }
        
        // Layer norm weights
        if (!save_matrix_to_file(file, layer->ln1_gamma, "ln1_gamma") ||
            !save_matrix_to_file(file, layer->ln1_beta, "ln1_beta") ||
            !save_matrix_to_file(file, layer->ln2_gamma, "ln2_gamma") ||
            !save_matrix_to_file(file, layer->ln2_beta, "ln2_beta")) {
            printf("Error: Failed to save layer norm weights for layer %d\n", l);
            fclose(file);
            return 1;
        }
    }
    
    // Save final layer norm and output projection
    if (!save_matrix_to_file(file, weights->final_ln_gamma, "final_ln_gamma") ||
        !save_matrix_to_file(file, weights->final_ln_beta, "final_ln_beta") ||
        !save_matrix_to_file(file, weights->output_projection, "output_projection")) {
        printf("Error: Failed to save final weights\n");
        fclose(file);
        return 1;
    }
    
    fclose(file);
    printf("Successfully saved transformer weights to %s\n", filepath);
    return 0;
}

// Helper function to load a matrix from file
static int load_matrix_from_file(FILE* file, Matrix** mat, const char* name) {
    int exists;
    if (fread(&exists, sizeof(int), 1, file) != 1) return 0;
    
    if (!exists) {
        *mat = NULL;
        return 1; // NULL matrix is valid
    }
    
    int rows, cols;
    if (fread(&rows, sizeof(int), 1, file) != 1) return 0;
    if (fread(&cols, sizeof(int), 1, file) != 1) return 0;
    
    *mat = create_matrix(rows, cols);
    if (!*mat) {
        printf("Error: Failed to allocate matrix %s (%dx%d)\n", name, rows, cols);
        return 0;
    }
    
    size_t size = (size_t)rows * cols;
    if (fread((*mat)->data, sizeof(float), size, file) != size) {
        printf("Error: Failed to read matrix data for %s\n", name);
        free_matrix(*mat);
        *mat = NULL;
        return 0;
    }
    
    return 1;
}

/**
 * Loads transformer model weights from a file.
 * 
 * @param weights: The TransformerWeights to load into (must be pre-allocated)
 * @param filepath: The file path from where weights should be loaded
 * @return: 0 on success, non-zero on failure
 */
int load_transformer_weights(TransformerWeights* weights, const char* filepath) {
    if (!weights || !filepath) {
        printf("Error: NULL pointer passed to load_transformer_weights\n");
        return 1;
    }
    
    FILE* file = fopen(filepath, "rb");
    if (!file) {
        printf("Error: Cannot open file %s for reading\n", filepath);
        return 1;
    }
    
    // Read and validate magic number and version
    uint32_t magic_number, version;
    if (fread(&magic_number, sizeof(uint32_t), 1, file) != 1 ||
        fread(&version, sizeof(uint32_t), 1, file) != 1) {
        printf("Error: Failed to read file header\n");
        fclose(file);
        return 1;
    }
    
    if (magic_number != 0x54524E53) { // "TRNS"
        printf("Error: Invalid file format (wrong magic number)\n");
        fclose(file);
        return 1;
    }
    
    if (version != 1) {
        printf("Error: Unsupported file version %u\n", version);
        fclose(file);
        return 1;
    }
    
    // Load config
    TransformerConfig* config = (TransformerConfig*)malloc(sizeof(TransformerConfig));
    if (!config || fread(config, sizeof(TransformerConfig), 1, file) != 1) {
        printf("Error: Failed to load config\n");
        free(config);
        fclose(file);
        return 1;
    }
    weights->config = config;
    
    // Load token embeddings
    if (!load_matrix_from_file(file, &weights->token_embeddings, "token_embeddings")) {
        printf("Error: Failed to load token_embeddings\n");
        fclose(file);
        return 1;
    }
    
    // Load positional embeddings
    if (!load_matrix_from_file(file, &weights->positional_embeddings, "positional_embeddings")) {
        printf("Error: Failed to load positional_embeddings\n");
        fclose(file);
        return 1;
    }
    
    // Allocate layer weights
    weights->layers = (TransformerLayerWeights*)malloc(config->n_layers * sizeof(TransformerLayerWeights));
    if (!weights->layers) {
        printf("Error: Failed to allocate layer weights\n");
        fclose(file);
        return 1;
    }
    
    // Load layer weights
    for (int l = 0; l < config->n_layers; l++) {
        TransformerLayerWeights* layer = &weights->layers[l];
        
        // Attention weights
        if (!load_matrix_from_file(file, &layer->attn_weights.q_proj, "q_proj") ||
            !load_matrix_from_file(file, &layer->attn_weights.k_proj, "k_proj") ||
            !load_matrix_from_file(file, &layer->attn_weights.v_proj, "v_proj") ||
            !load_matrix_from_file(file, &layer->attn_weights.o_proj, "o_proj")) {
            printf("Error: Failed to load attention weights for layer %d\n", l);
            fclose(file);
            return 1;
        }
        
        // Feed forward weights
        if (!load_matrix_from_file(file, &layer->ff_weights.fc1, "fc1") ||
            !load_matrix_from_file(file, &layer->ff_weights.fc2, "fc2")) {
            printf("Error: Failed to load feed forward weights for layer %d\n", l);
            fclose(file);
            return 1;
        }
        
        // Layer norm weights
        if (!load_matrix_from_file(file, &layer->ln1_gamma, "ln1_gamma") ||
            !load_matrix_from_file(file, &layer->ln1_beta, "ln1_beta") ||
            !load_matrix_from_file(file, &layer->ln2_gamma, "ln2_gamma") ||
            !load_matrix_from_file(file, &layer->ln2_beta, "ln2_beta")) {
            printf("Error: Failed to load layer norm weights for layer %d\n", l);
            fclose(file);
            return 1;
        }
    }
    
    // Load final layer norm and output projection
    if (!load_matrix_from_file(file, &weights->final_ln_gamma, "final_ln_gamma") ||
        !load_matrix_from_file(file, &weights->final_ln_beta, "final_ln_beta") ||
        !load_matrix_from_file(file, &weights->output_projection, "output_projection")) {
        printf("Error: Failed to load final weights\n");
        fclose(file);
        return 1;
    }
    
    fclose(file);
    printf("Successfully loaded transformer weights from %s\n", filepath);
    return 0;
}

/* ============================================================================
 * GRADIENT UTILITY FUNCTIONS
 * ============================================================================
 * These functions manage the gradients - the values that are computed during
 * backpropagation to update the model's weights.
 */

/**
 * Creates and allocates memory for all transformer model gradients.
 * This reuses the allocation logic from create_transformer_weights, as the
 * structures are similar.
 * 
 * @param config: Model configuration specifying dimensions and layer count
 * @return: Newly allocated TransformerGradients structure, or NULL on failure
 */
TransformerGradients* create_transformer_gradients(const TransformerConfig* config) {
    return create_transformer_weights(config); // Reuses allocation logic
}

/**
 * Frees all memory associated with transformer model gradients.
 * This uses the same free logic as for weights, since the structures are similar.
 * 
 * @param grads: The TransformerGradients to free
 */
void free_transformer_gradients(TransformerGradients* grads) {
    free_transformer_weights(grads); // Reuses free logic
}

/**
 * Zeros out all gradients in the TransformerGradients structure.
 * This is important before starting a new training step - it clears old gradients
 * that would otherwise accumulate.
 * 
 * @param grads: The TransformerGradients to zero out
 */
void zero_gradients(TransformerGradients* grads) {
    if (!grads) return;
    // Iterate through all grad matrices and fill with 0.0f
    // This is tedious; a helper macro or function to iterate fields would be good.
    // Example for one: fill_matrix(grads->token_embeddings, 0.0f);
    // ... Do this for ALL matrices in TransformerGradients ...
    // Assuming a helper:
    void _fill_all_matrices_in_weights_struct(TransformerWeights* s, float val) {
        if (!s) return;
        
        // Fill embeddings
        if(s->token_embeddings) fill_matrix(s->token_embeddings, val);
        if(s->positional_embeddings) fill_matrix(s->positional_embeddings, val);
        
        // Fill per-layer weights
        for (int i = 0; i < s->config->n_layers; ++i) {
            // Attention weights
            if(s->layers[i].attn_weights.q_proj) fill_matrix(s->layers[i].attn_weights.q_proj, val);
            if(s->layers[i].attn_weights.k_proj) fill_matrix(s->layers[i].attn_weights.k_proj, val);
            if(s->layers[i].attn_weights.v_proj) fill_matrix(s->layers[i].attn_weights.v_proj, val);
            if(s->layers[i].attn_weights.o_proj) fill_matrix(s->layers[i].attn_weights.o_proj, val);
            
            // Feed-forward weights
            if(s->layers[i].ff_weights.fc1) fill_matrix(s->layers[i].ff_weights.fc1, val);
            if(s->layers[i].ff_weights.fc2) fill_matrix(s->layers[i].ff_weights.fc2, val);
            
            // Layer norm weights
            if(s->layers[i].ln1_gamma) fill_matrix(s->layers[i].ln1_gamma, val);
            if(s->layers[i].ln1_beta) fill_matrix(s->layers[i].ln1_beta, val);
            if(s->layers[i].ln2_gamma) fill_matrix(s->layers[i].ln2_gamma, val);
            if(s->layers[i].ln2_beta) fill_matrix(s->layers[i].ln2_beta, val);
        }
        
        // Fill final weights
        if(s->final_ln_gamma) fill_matrix(s->final_ln_gamma, val);
        if(s->final_ln_beta) fill_matrix(s->final_ln_beta, val);
        if(s->output_projection) fill_matrix(s->output_projection, val);
    }
    _fill_all_matrices_in_weights_struct(grads, 0.0f);
}


/**
 * Calculates the total L2 norm of all gradients.
 * This is used for gradient clipping - a technique to prevent too-large updates
 * that can destabilize training.
 * 
 * @param grads: The TransformerGradients containing the gradients
 * @return: The total L2 norm (squared sum, then square root)
 */
static float _calculate_total_gradient_norm_sq(TransformerGradients* grads) {
    float total_norm_sq = 0.0f;
    // Helper macro to add norm sq of a matrix if it exists
    #define ADD_NORM_SQ(matrix_ptr) \
        if (matrix_ptr) { \
            for (int k = 0; k < (matrix_ptr)->rows * (matrix_ptr)->cols; ++k) { \
                float val = (matrix_ptr)->data[k]; \
                total_norm_sq += val * val; \
            }\
        }

    ADD_NORM_SQ(grads->token_embeddings);
    ADD_NORM_SQ(grads->positional_embeddings);

    for (int i = 0; i < grads->config->n_layers; ++i) { // Corrected: Use -> for pointer
        ADD_NORM_SQ(grads->layers[i].attn_weights.q_proj); // Corrected: att_weights to attn_weights
        ADD_NORM_SQ(grads->layers[i].attn_weights.k_proj); // Corrected: att_weights to attn_weights
        ADD_NORM_SQ(grads->layers[i].attn_weights.v_proj); // Corrected: att_weights to attn_weights
        ADD_NORM_SQ(grads->layers[i].attn_weights.o_proj); // Corrected: att_weights to attn_weights
        // Add biases if they exist in struct
        ADD_NORM_SQ(grads->layers[i].ln1_gamma); // Corrected: norm1_gain to ln1_gamma
        ADD_NORM_SQ(grads->layers[i].ln1_beta);  // Corrected: norm1_bias to ln1_beta
        ADD_NORM_SQ(grads->layers[i].ff_weights.fc1);    // Corrected: ffn_weights to ff_weights
        ADD_NORM_SQ(grads->layers[i].ff_weights.fc2);    // Corrected: ffn_weights to ff_weights
        // Add biases if they exist
        ADD_NORM_SQ(grads->layers[i].ln2_gamma); // Corrected: norm2_gain to ln2_gamma
        ADD_NORM_SQ(grads->layers[i].ln2_beta);  // Corrected: norm2_bias to ln2_beta
    }

    ADD_NORM_SQ(grads->final_ln_gamma); // Corrected: final_norm_gain to final_ln_gamma
    ADD_NORM_SQ(grads->final_ln_beta);  // Corrected: final_norm_bias to final_ln_beta
    ADD_NORM_SQ(grads->output_projection); // Corrected: output_projection_W to output_projection
    // Add output_projection_bias if it exists

    #undef ADD_NORM_SQ
    return total_norm_sq;
}

/**
 * Clips gradients by scaling them down if the total norm exceeds a maximum value.
 * This prevents excessively large updates to the model weights, which can destabilize training.
 * 
 * @param grads: The TransformerGradients containing the gradients to clip
 * @param max_norm: The maximum allowed norm. Gradients are scaled by (max_norm / total_norm)
 *                   if total_norm exceeds max_norm.
 */
void clip_gradients_global_norm(TransformerGradients* grads, float max_norm) {
    if (max_norm <= 0.0f) return; // No clipping if max_norm is non-positive

    float total_norm_sq = _calculate_total_gradient_norm_sq(grads);
    float total_norm = sqrtf(total_norm_sq);

    if (total_norm > max_norm) {
        float scale_factor = max_norm / total_norm;
        // Helper macro to scale a gradient matrix if it exists
        #define SCALE_GRAD_MATRIX(matrix_ptr) \
            if (matrix_ptr) mat_mul_scalar(matrix_ptr, scale_factor);

        SCALE_GRAD_MATRIX(grads->token_embeddings);
        SCALE_GRAD_MATRIX(grads->positional_embeddings);

        for (int i = 0; i < grads->config->n_layers; ++i) { // Corrected: Use -> for pointer
            SCALE_GRAD_MATRIX(grads->layers[i].attn_weights.q_proj); // Corrected
            SCALE_GRAD_MATRIX(grads->layers[i].attn_weights.k_proj); // Corrected
            SCALE_GRAD_MATRIX(grads->layers[i].attn_weights.v_proj); // Corrected
            SCALE_GRAD_MATRIX(grads->layers[i].attn_weights.o_proj); // Corrected
            SCALE_GRAD_MATRIX(grads->layers[i].ln1_gamma);     // Corrected
            SCALE_GRAD_MATRIX(grads->layers[i].ln1_beta);      // Corrected
            SCALE_GRAD_MATRIX(grads->layers[i].ff_weights.fc1);    // Corrected
            SCALE_GRAD_MATRIX(grads->layers[i].ff_weights.fc2);    // Corrected
            SCALE_GRAD_MATRIX(grads->layers[i].ln2_gamma);     // Corrected
            SCALE_GRAD_MATRIX(grads->layers[i].ln2_beta);      // Corrected
        }

        SCALE_GRAD_MATRIX(grads->final_ln_gamma);          // Corrected
        SCALE_GRAD_MATRIX(grads->final_ln_beta);           // Corrected
        SCALE_GRAD_MATRIX(grads->output_projection);      // Corrected

        #undef SCALE_GRAD_MATRIX
    }
}


/* =============================================
 * TRAINING BATCH FORWARD/BACKWARD PASS
 * ==============================================
 * This is the core of the training process - it performs a forward pass
 * (computing predictions) and a backward pass (updating weights).
 */

/**
 * Performs a single training step on a batch of data.
 * This function assumes the existence of a valid TransformerWeights, TransformerGradients,
 * and TrainingActivationsCache structures, properly configured and allocated.
 * 
 * @param weights: The model weights
 * @param grads: The gradient accumulators
 * @param activations_cache: Cache for storing activations (outputs of layers) during forward pass
 * @param input_token_ids_batch: Batch of input token IDs (flattened)
 * @param target_token_ids_batch: Batch of target token IDs (flattened, for computing loss)
 * @param causal_mask: Mask to apply causality in attention (prevents attending to future tokens)
 * @param config: Transformer configuration (contains hyperparameters like batch size, etc.)
 * @param is_first_batch_of_epoch: Flag indicating if this is the first batch in the current epoch
 * @return: The computed loss for the batch
 */
float transformer_train_batch(
    TransformerWeights* weights,
    TransformerGradients* grads,
    TrainingActivationsCache* activations_cache, // Changed parameter name to match header
    const Matrix* input_token_ids_batch, const Matrix* target_token_ids_batch,
    Matrix* causal_mask, const TransformerConfig* config, int is_first_batch_of_epoch) {
    int batch_size = config->batch_size; // from config
    int current_seq_len = input_token_ids_batch->rows / batch_size; // Assuming it's a multiple
    int num_tokens_in_batch = batch_size * current_seq_len;
    int d_model = config->d_model;
    int d_ff = config->d_ff;
    int vocab_size = config->vocab_size;

    // --- FORWARD PASS ---
    // 0. Reset/Prepare Activations Cache (if needed per batch)

    // 1. Embedding Layer
    // x_embedded: (num_tokens_in_batch, d_model)
    Matrix* x_embedded = activations_cache->x_embedding_output_batch; // Get from cache or create
    fill_matrix(x_embedded, 0.0f); // Zero out
    for (int i = 0; i < num_tokens_in_batch; ++i) {
        int token_id = (int)input_token_ids_batch->data[i];
        int seq_pos = i % current_seq_len; // Position within its sequence
        if (token_id < 0 || token_id >= vocab_size) token_id = 0; // Safety for UNK/PAD
        if (seq_pos < 0 || seq_pos >= config->max_seq_len) seq_pos = 0; // Safety

        // Add token embedding
        for (int d = 0; d < d_model; ++d) {
            x_embedded->data[i * d_model + d] += weights->token_embeddings->data[token_id * d_model + d];
        }
        // Add positional embedding
        for (int d = 0; d < d_model; ++d) {
            x_embedded->data[i * d_model + d] += weights->positional_embeddings->data[seq_pos * d_model + d];
        }
    }

    Matrix* current_x = create_matrix(num_tokens_in_batch, d_model);
    copy_matrix(current_x, x_embedded); // Input to the first layer

    // Store inputs/outputs for LayerNorm backward pass
    // This is simplified. In reality, TrainingActivationsCache would handle this.
    Matrix** layer_inputs_ln1 = (Matrix**)safe_malloc(config->n_layers * sizeof(Matrix*));
    Matrix** layer_inputs_ln2 = (Matrix**)safe_malloc(config->n_layers * sizeof(Matrix*));
    // ... (allocate and store all necessary intermediate activations)

    // 2. Transformer Layers
    for (int l = 0; l < config->n_layers; ++l) {
        // Store input to LayerNorm1 for this layer
        layer_inputs_ln1[l] = create_matrix(num_tokens_in_batch, d_model);
        copy_matrix(layer_inputs_ln1[l], current_x);

        Matrix* norm1_out = create_matrix(num_tokens_in_batch, d_model);
        layer_norm_forward(norm1_out, activations_cache->ln1_means_batch[l], activations_cache->ln1_inv_stddevs_batch[l],
                          current_x, weights->layers[l].ln1_gamma, weights->layers[l].ln1_beta, config->layer_norm_eps);

        Matrix* mha_out = create_matrix(num_tokens_in_batch, d_model);
        // Simplified Multi-Head Attention Forward Pass
        // This is a basic implementation that performs the core attention computation
        // without full multi-head splitting for simplicity
        
        // Compute Q, K, V projections
        Matrix* Q = create_matrix(num_tokens_in_batch, d_model);
        Matrix* K = create_matrix(num_tokens_in_batch, d_model);
        Matrix* V = create_matrix(num_tokens_in_batch, d_model);
        
        mat_mul(Q, norm1_out, weights->layers[l].attn_weights.q_proj);
        mat_mul(K, norm1_out, weights->layers[l].attn_weights.k_proj);
        mat_mul(V, norm1_out, weights->layers[l].attn_weights.v_proj);
        
        // Compute attention scores: Q @ K^T
        Matrix* K_T = create_matrix(d_model, num_tokens_in_batch);
        mat_transpose(K_T, K);
        
        Matrix* attention_scores = create_matrix(num_tokens_in_batch, num_tokens_in_batch);
        mat_mul(attention_scores, Q, K_T);
        
        // Scale by sqrt(d_model)
        float scale = 1.0f / sqrtf((float)d_model);
        mat_mul_scalar(attention_scores, scale);
        
        // Apply causal mask (only for autoregressive training)
        if (causal_mask && causal_mask->rows == current_seq_len && causal_mask->cols == current_seq_len) {
            // Apply mask to each sequence in the batch
            for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
                for (int i = 0; i < current_seq_len; i++) {
                    for (int j = 0; j < current_seq_len; j++) {
                        int batch_i = seq_idx * current_seq_len + i;
                        int batch_j = seq_idx * current_seq_len + j;
                        if (batch_i < num_tokens_in_batch && batch_j < num_tokens_in_batch) {
                            attention_scores->data[batch_i * num_tokens_in_batch + batch_j] += 
                                causal_mask->data[i * current_seq_len + j];
                        }
                    }
                }
            }
        }
        
        // Apply softmax to attention scores
        softmax_row_wise(attention_scores);
        
        // Compute attention output: attention_scores @ V
        Matrix* attention_output = create_matrix(num_tokens_in_batch, d_model);
        mat_mul(attention_output, attention_scores, V);
        
        // Output projection
        mat_mul(mha_out, attention_output, weights->layers[l].attn_weights.o_proj);
        
        // Cleanup
        free_matrix(Q);
        free_matrix(K);
        free_matrix(V);
        free_matrix(K_T);
        free_matrix(attention_scores);
        free_matrix(attention_output);

        // Add residual (current_x was input to norm1)
        mat_add(current_x, current_x, mha_out); // current_x = current_x (residual) + mha_out
        free_matrix(norm1_out);
        free_matrix(mha_out);

        // Store input to LayerNorm2
        layer_inputs_ln2[l] = create_matrix(num_tokens_in_batch, d_model);
        copy_matrix(layer_inputs_ln2[l], current_x);

        Matrix* norm2_out = create_matrix(num_tokens_in_batch, d_model);
        layer_norm_forward(norm2_out, activations_cache->ln2_means_batch[l], activations_cache->ln2_inv_stddevs_batch[l],
                          current_x, weights->layers[l].ln2_gamma, weights->layers[l].ln2_beta, config->layer_norm_eps);

        Matrix* ffn_out = create_matrix(num_tokens_in_batch, d_model);
        // ffn_forward_batched(ffn_out, norm2_out, &weights->layers[l].ffn_weights, config, activations_cache, l);
        // STUB:
        mat_mul(activations_cache->ffn_hidden_state[l], norm2_out, weights->layers[l].ff_weights.fc1); // Corrected: ffn_weights to ff_weights
        // NOTE: No bias b1 in current implementation (omitted for simplicity as per design)
        gelu_activation(activations_cache->ffn_hidden_state[l]);
        mat_mul(ffn_out, activations_cache->ffn_hidden_state[l], weights->layers[l].ff_weights.fc2); // Corrected: ffn_weights to ff_weights
        // NOTE: No bias b2 in current implementation (omitted for simplicity as per design)


        // Add residual
        mat_add(current_x, current_x, ffn_out); // current_x = current_x (residual) + ffn_out
        free_matrix(norm2_out);
        free_matrix(ffn_out);
    }

    // 3. Final LayerNorm
    Matrix* input_final_ln = create_matrix(num_tokens_in_batch, d_model);
    copy_matrix(input_final_ln, current_x); // Store for backward

    Matrix* final_norm_out = create_matrix(num_tokens_in_batch, d_model);
    layer_norm_forward(final_norm_out, activations_cache->final_ln_means_batch, activations_cache->final_ln_inv_stddevs_batch,
                      current_x, weights->final_ln_gamma, weights->final_ln_beta, config->layer_norm_eps);
    free_matrix(current_x); // current_x no longer needed for fwd pass

    // 4. Output Projection (Linear layer to vocab_size)
    Matrix* logits = create_matrix(num_tokens_in_batch, vocab_size);
    mat_mul(logits, final_norm_out, weights->output_projection); // Corrected: output_projection_W to output_projection
    // NOTE: No output_projection_b in current implementation (omitted for simplicity as per design)

    // --- LOSS CALCULATION & INITIAL GRADIENT ---
    Matrix* grad_logits = create_matrix(num_tokens_in_batch, vocab_size); // dL/dLogits
    // Flatten target_token_ids_batch_flat (already flat if input is (N*S)x1)
    int* targets_flat_int_array = (int*)safe_malloc(num_tokens_in_batch * sizeof(int));
    for(int i=0; i<num_tokens_in_batch; ++i) targets_flat_int_array[i] = (int)target_token_ids_batch->data[i];

    float loss = softmax_cross_entropy_loss_and_backward(grad_logits, logits, targets_flat_int_array, num_tokens_in_batch, config->vocab_size);
    free(targets_flat_int_array);


    // --- BACKWARD PASS ---
    // Gradients are accumulated in `grads` struct.
    // Start with grad_logits (dL/dLogits) and propagate backwards.
    Matrix* current_grad_x = create_matrix(num_tokens_in_batch, d_model); // dL/dX for current component's input

    // 4. Output Projection Backward
    linear_layer_backward(current_grad_x, grads->output_projection, NULL /*grad_bias*/,
                          grad_logits, final_norm_out, weights->output_projection); // Corrected field names
    free_matrix(grad_logits);
    // free_matrix(final_norm_out); // final_norm_out is needed as input_forward for next step

    // 3. Final LayerNorm Backward
    Matrix* grad_input_to_final_ln = create_matrix(num_tokens_in_batch, d_model);
    layer_norm_backward(grad_input_to_final_ln, grads->final_ln_gamma, grads->final_ln_beta,
                        current_grad_x, input_final_ln, weights->final_ln_gamma, 
                        activations_cache->final_ln_means_batch, activations_cache->final_ln_inv_stddevs_batch, 
                        config->layer_norm_eps); // Corrected field names

    // current_grad_x now holds dL/d(output_of_last_transformer_layer)
    copy_matrix(current_grad_x, grad_input_to_final_ln); 
    free_matrix(grad_input_to_final_ln);
    free_matrix(input_final_ln);
    free_matrix(final_norm_out);


    // 2. Transformer Layers Backward (in reverse order)
    for (int l = config->n_layers - 1; l >= 0; --l) {
        // Add & Norm residual backward (FFN part)
        // Gradient from Add splits. current_grad_x is dL/d(Output_of_Add)
        Matrix* grad_ffn_out_acc = create_matrix(num_tokens_in_batch, d_model); // Temp for grad w.r.t. FFN output
        copy_matrix(grad_ffn_out_acc, current_grad_x); // grad flows to FFN output
        // grad also flows to residual connection (which was input to LayerNorm2 for FFN)
        // So, current_grad_x now holds dL/d(Input_to_LayerNorm2_for_FFN) *before* FFN was added.

        // FFN Backward
        Matrix* grad_input_to_ffn = create_matrix(num_tokens_in_batch, d_model);
        ffn_layer_backward(grad_input_to_ffn, 
                          grads->layers[l].ff_weights.fc1, grads->layers[l].ff_weights.fc2,
                          NULL, NULL, // No biases in current implementation
                          current_grad_x, // grad_output
                          layer_inputs_ln2[l], // input to FFN (norm2_out)
                          activations_cache->ffn_hidden_state[l], // GELU output
                          weights->layers[l].ff_weights.fc1, weights->layers[l].ff_weights.fc2,
                          config);

        copy_matrix(current_grad_x, grad_input_to_ffn); // current_grad_x = dL/d(Output_of_LayerNorm2)
        free_matrix(grad_ffn_out_acc);
        free_matrix(grad_input_to_ffn);


        // LayerNorm2 Backward
        Matrix* grad_input_to_ln2 = create_matrix(num_tokens_in_batch, d_model);
        layer_norm_backward(grad_input_to_ln2, grads->layers[l].ln2_gamma, grads->layers[l].ln2_beta,
                            current_grad_x, layer_inputs_ln2[l], weights->layers[l].ln2_gamma,
                            activations_cache->ln2_means_batch[l], activations_cache->ln2_inv_stddevs_batch[l],
                            config->layer_norm_eps); // Corrected field names
        copy_matrix(current_grad_x, grad_input_to_ln2);
        free_matrix(grad_input_to_ln2);
        free_matrix(layer_inputs_ln2[l]);


        // Add & Norm residual backward (MHA part) - similar logic
        Matrix* grad_mha_out_acc = create_matrix(num_tokens_in_batch, d_model);
        copy_matrix(grad_mha_out_acc, current_grad_x);

        // MHA Backward
        Matrix* grad_input_to_mha = create_matrix(num_tokens_in_batch, d_model); // dL/d(Output_of_LayerNorm1)
        attention_layer_backward(grad_input_to_mha, &grads->layers[l].attn_weights, 
                                grad_mha_out_acc, layer_inputs_ln1[l], 
                                &weights->layers[l].attn_weights, config);

        copy_matrix(current_grad_x, grad_input_to_mha);
        free_matrix(grad_mha_out_acc);
        free_matrix(grad_input_to_mha);

        // LayerNorm1 Backward
        Matrix* grad_input_to_ln1 = create_matrix(num_tokens_in_batch, d_model);
        layer_norm_backward(grad_input_to_ln1, grads->layers[l].ln1_gamma, grads->layers[l].ln1_beta,
                            current_grad_x, layer_inputs_ln1[l], weights->layers[l].ln1_gamma,
                            activations_cache->ln1_means_batch[l], activations_cache->ln1_inv_stddevs_batch[l],
                            config->layer_norm_eps); // Corrected field names

        // current_grad_x now holds dL/d(output_of_embedding_layer_plus_positional_encoding)
        copy_matrix(current_grad_x, grad_input_to_ln1);
        free_matrix(grad_input_to_ln1);
        free_matrix(layer_inputs_ln1[l]);
    } // End layers backward loop

    // 1. Embedding Layer Backward
    embedding_layer_backward(grads->token_embeddings, grads->positional_embeddings,
                             current_grad_x, input_token_ids_batch, current_seq_len);

    // --- Cleanup ---
    free_matrix(current_grad_x);
    free_matrix(logits);
    // Free layer_inputs_ln1, layer_inputs_ln2 arrays
    free(layer_inputs_ln1);
    free(layer_inputs_ln2);
    // free_training_activations_cache(activations_cache, config); // Or managed outside per epoch

    return loss;
}

// --- Backward pass for a simple linear layer: Y = XW + B ---
// grad_input_acc: accumulates dL/dX
// grad_W_acc: accumulates dL/dW
// grad_b_acc: accumulates dL/dB (optional, if NULL, bias grad not computed)
// grad_output: dL/dY
// input_forward: X from forward pass
// W_forward: W from forward pass
static void linear_layer_backward(Matrix* grad_input_acc, Matrix* grad_W_acc, Matrix* grad_b_acc,
                                  const Matrix* grad_output, const Matrix* input_forward, const Matrix* W_forward) {
    // Add null pointer checks
    if (!grad_input_acc || !grad_W_acc || !grad_output || !input_forward || !W_forward) {
        fprintf(stderr, "Error: NULL pointer passed to linear_layer_backward\n");
        return;
    }
    
    // Validate dimensions for backward pass
    // For Y = X @ W, backward pass computes:
    // dL/dX = dL/dY @ W^T  =>  grad_output @ W^T
    // dL/dW = X^T @ dL/dY  =>  input^T @ grad_output
    if (grad_output->cols != W_forward->cols || input_forward->cols != W_forward->rows) {
        fprintf(stderr, "Error: Matrix dimension mismatch in linear_layer_backward\n");
        fprintf(stderr, "  grad_output: %dx%d, W_forward: %dx%d, input_forward: %dx%d\n",
                grad_output->rows, grad_output->cols, W_forward->rows, W_forward->cols,
                input_forward->rows, input_forward->cols);
        fprintf(stderr, "  Expected: grad_output->cols (%d) == W_forward->cols (%d) && input_forward->cols (%d) == W_forward->rows (%d)\n",
                grad_output->cols, W_forward->cols, input_forward->cols, W_forward->rows);
        fprintf(stderr, "  This is for Y = X @ W where X:%dx%d, W:%dx%d, Y:%dx%d\n",
                input_forward->rows, input_forward->cols, W_forward->rows, W_forward->cols,
                grad_output->rows, grad_output->cols);
        return;
    }
    
    // dL/dX += dL/dY @ W^T
    Matrix* W_T = create_matrix(W_forward->cols, W_forward->rows);
    if (!W_T) {
        fprintf(stderr, "Error: Failed to allocate memory for W_T in linear_layer_backward\n");
        return;
    }
    
    // Debug information before transpose
    // printf("Debug: About to transpose W_forward (%dx%d) into W_T (%dx%d)\n",
    //        W_forward->rows, W_forward->cols, W_T->rows, W_T->cols);
    
    mat_transpose(W_T, W_forward);
    mat_mul_add(grad_input_acc, grad_output, W_T); // grad_input_acc += grad_output @ W_T
    free_matrix(W_T);

    // dL/dW += X^T @ dL/dY
    Matrix* input_T = create_matrix(input_forward->cols, input_forward->rows);
    if (!input_T) {
        fprintf(stderr, "Error: Failed to allocate memory for input_T in linear_layer_backward\n");
        return;
    }
    
    // Debug information before transpose
    // printf("Debug: About to transpose input_forward (%dx%d) into input_T (%dx%d)\n",
    //        input_forward->rows, input_forward->cols, input_T->rows, input_T->cols);
    
    mat_transpose(input_T, input_forward);
    mat_mul_add(grad_W_acc, input_T, grad_output); // grad_W_acc += input_T @ grad_output
    free_matrix(input_T);

    // dL/dB += sum(dL/dY, axis=0) (sum over batch dimension)
    if (grad_b_acc) {
        assert(grad_b_acc->rows == 1 && grad_b_acc->cols == grad_output->cols);
        for (int j = 0; j < grad_output->cols; ++j) {
            float sum_col_grad = 0.0f;
            for (int i = 0; i < grad_output->rows; ++i) { // Summing rows (batch items)
                sum_col_grad += grad_output->data[i * grad_output->cols + j];
            }
            grad_b_acc->data[j] += sum_col_grad;
        }
    }
}

// Helper for mat_mul_add: C += A @ B
void mat_mul_add(Matrix* C_acc, const Matrix* A, const Matrix* B) {
    // Ensure dimensions are compatible for C_acc += A @ B
    // A: m x k, B: k x n, C_acc: m x n
    if (!A || !B || !C_acc) {
        fprintf(stderr, "Error: NULL pointer passed to mat_mul_add\n");
        return;
    }
    
    // Print dimension information for debugging
    // printf("Debug mat_mul_add: C_acc(%dx%d) += A(%dx%d) @ B(%dx%d)\n",
    //        C_acc->rows, C_acc->cols, A->rows, A->cols, B->rows, B->cols);
    
    if (A->cols != B->rows) {
        fprintf(stderr, "Error: mat_mul_add dimension mismatch - A->cols (%d) != B->rows (%d)\n",
                A->cols, B->rows);
        return;
    }
    if (C_acc->rows != A->rows) {
        fprintf(stderr, "Error: mat_mul_add dimension mismatch - C_acc->rows (%d) != A->rows (%d)\n",
                C_acc->rows, A->rows);
        return;
    }
    if (C_acc->cols != B->cols) {
        fprintf(stderr, "Error: mat_mul_add dimension mismatch - C_acc->cols (%d) != B->cols (%d)\n",
                C_acc->cols, B->cols);
        return;
    }

    // This is a simple, unoptimized implementation.
    // For performance, consider BLAS (sgemm with beta=1.0) or optimized loops.
    for (int i = 0; i < A->rows; ++i) {
        for (int j = 0; j < B->cols; ++j) {
            float sum = 0.0f;
            for (int l = 0; l < A->cols; ++l) {
                sum += A->data[i * A->cols + l] * B->data[l * B->cols + j];
            }
            C_acc->data[i * C_acc->cols + j] += sum;
        }
    }
}


// Placeholder for embedding backward pass
static void embedding_layer_backward(Matrix* grad_token_emb_acc, Matrix* grad_pos_emb_acc,
                                     const Matrix* grad_output_emb_sum, // dL/d(token_emb + pos_emb)
                                     const Matrix* input_token_ids_flat, // (batch_size*seq_len) x 1
                                     int current_seq_len) {
    // Add null pointer checks
    if (!grad_token_emb_acc || !grad_pos_emb_acc || !grad_output_emb_sum || !input_token_ids_flat) {
        fprintf(stderr, "Error: NULL pointer passed to embedding_layer_backward\n");
        return;
    }
    
    int num_tokens_in_batch = grad_output_emb_sum->rows;
    int d_model = grad_output_emb_sum->cols;

    for (int i = 0; i < num_tokens_in_batch; ++i) {
        int token_id = (int)input_token_ids_flat->data[i];
        int seq_pos = i % current_seq_len;

        // Gradient flows to both token and positional embeddings
        // Accumulate gradients for the specific token_id row in token_embeddings
        if (token_id >= 0 && token_id < grad_token_emb_acc->rows) {
            for (int d = 0; d < d_model; ++d) {
                #pragma omp atomic // If parallelizing this loop (outer loop over i)
                grad_token_emb_acc->data[token_id * d_model + d] += grad_output_emb_sum->data[i * d_model + d];
            }
        }
        // Accumulate gradients for the specific seq_pos row in positional_embeddings
        if (seq_pos >= 0 && seq_pos < grad_pos_emb_acc->rows) {
            for (int d = 0; d < d_model; ++d) {
                 #pragma omp atomic
                grad_pos_emb_acc->data[seq_pos * d_model + d] += grad_output_emb_sum->data[i * d_model + d];
            }
        }
    }
}

// Stubs for complex backward passes (MHA, FFN)
// These would be very involved, each decomposing into smaller backward steps.
static void attention_layer_backward(
    Matrix* grad_input_acc,           // Output: dL/d(input to attention block)
    AttentionWeights* grad_attn_weights_acc, // Output: gradients for attention weights
    const Matrix* grad_output,        // Input: dL/d(output of attention block)
    const Matrix* input_forward,      // Input: stored input from forward pass
    const AttentionWeights* weights_forward, // Input: attention weights from forward pass
    const TransformerConfig* config
) {
    // Simplified attention backward pass
    // This implements a basic version of multi-head attention backward pass
    // Note: This is a simplified implementation for educational purposes
    
    int num_tokens = input_forward->rows;
    int d_model = input_forward->cols;
    
    // Step 1: Backward through output projection (O matrix)
    // grad_attention_output = grad_output @ O^T
    Matrix* grad_attention_output = create_matrix(num_tokens, d_model);
    linear_layer_backward(grad_attention_output, grad_attn_weights_acc->o_proj, NULL,
                         grad_output, input_forward, weights_forward->o_proj);
    
    // For a full implementation, we would need to:
    // 1. Backward through attention matrix multiplication (attention_scores @ V)
    // 2. Backward through softmax
    // 3. Backward through scaled dot-product attention (Q @ K^T / sqrt(d_model))
    // 4. Backward through Q, K, V projections
    
    // Simplified approach: approximate the complex attention backward pass
    // by distributing gradients equally among Q, K, V projections
    
    // Create temporary matrices for Q, K, V gradients
    Matrix* grad_Q = create_matrix(num_tokens, d_model);
    Matrix* grad_K = create_matrix(num_tokens, d_model);
    Matrix* grad_V = create_matrix(num_tokens, d_model);
    
    // Simplified gradient distribution (1/3 each for Q, K, V paths)
    float scale = 1.0f / 3.0f;
    for (int i = 0; i < num_tokens * d_model; i++) {
        grad_Q->data[i] = grad_attention_output->data[i] * scale;
        grad_K->data[i] = grad_attention_output->data[i] * scale;
        grad_V->data[i] = grad_attention_output->data[i] * scale;
    }
    
    // Backward through Q, K, V projections
    Matrix* grad_input_Q = create_matrix(num_tokens, d_model);
    Matrix* grad_input_K = create_matrix(num_tokens, d_model);
    Matrix* grad_input_V = create_matrix(num_tokens, d_model);
    
    linear_layer_backward(grad_input_Q, grad_attn_weights_acc->q_proj, NULL,
                         grad_Q, input_forward, weights_forward->q_proj);
    linear_layer_backward(grad_input_K, grad_attn_weights_acc->k_proj, NULL,
                         grad_K, input_forward, weights_forward->k_proj);
    linear_layer_backward(grad_input_V, grad_attn_weights_acc->v_proj, NULL,
                         grad_V, input_forward, weights_forward->v_proj);
    
    // Accumulate gradients for input
    for (int i = 0; i < num_tokens * d_model; i++) {
        grad_input_acc->data[i] += grad_input_Q->data[i] + grad_input_K->data[i] + grad_input_V->data[i];
    }
    
    // Cleanup
    free_matrix(grad_attention_output);
    free_matrix(grad_Q);
    free_matrix(grad_K);
    free_matrix(grad_V);
}

static void ffn_layer_backward(
    Matrix* grad_input_acc, // dL/d(input to FFN block)
    Matrix* grad_W1_acc, Matrix* grad_W2_acc, // Grads for weights
    Matrix* grad_b1_acc, Matrix* grad_b2_acc, // Grads for biases (optional)
    const Matrix* grad_output, // dL/d(output of FFN block)
    const Matrix* input_forward, // Input to FFN block (e.g. output of LN2)
    const Matrix* ffn_hidden_forward, // Output of GELU(input @ W1 + b1)
    const Matrix* W1_forward, const Matrix* W2_forward, // Weights
    const TransformerConfig* config
) {
    // Simplified FFN backward pass implementation
    // FFN: output = (input @ W1 + b1) -> GELU -> @ W2 + b2
    
    // Step 1: Backward through second linear layer (W2)
    // grad_ffn_hidden_acc = grad_output @ W2^T
    Matrix* grad_ffn_hidden_acc = create_matrix_like(ffn_hidden_forward);
    linear_layer_backward(grad_ffn_hidden_acc, grad_W2_acc, grad_b2_acc, 
                         grad_output, ffn_hidden_forward, W2_forward);
    
    // Step 2: Backward through GELU activation
    // We need the input to GELU (output of first linear layer)
    Matrix* fc1_output = create_matrix(input_forward->rows, config->d_ff);
    mat_mul(fc1_output, input_forward, W1_forward);
    // Note: No bias addition since current implementation omits biases
    
    Matrix* grad_fc1_output_acc = create_matrix_like(fc1_output);
    gelu_activation_backward(grad_fc1_output_acc, fc1_output, grad_ffn_hidden_acc);
    
    // Step 3: Backward through first linear layer (W1)
    linear_layer_backward(grad_input_acc, grad_W1_acc, grad_b1_acc,
                         grad_fc1_output_acc, input_forward, W1_forward);
    
    // Cleanup
    free_matrix(grad_ffn_hidden_acc);
    free_matrix(fc1_output);
    free_matrix(grad_fc1_output_acc);
}


/* ===========================================
 * TRAINING ACTIVATIONS CACHE MANAGEMENT
 * =============================================
 * These functions manage the cache of activations - the outputs of each layer
 * during the forward pass. This cache is used during the backward pass to compute
 * gradients.
 */

/**
 * Creates a cache for storing activations during training.
 * This is crucial for the backward pass - we need to remember the output of
 * each layer for each input in the batch, so we can compute gradients.
 * 
 * @param config: Model configuration specifying dimensions and layer count
 * @param current_batch_size: The batch size for the current training step
 * @param current_seq_len: The sequence length for the current training step
 * @return: Newly allocated TrainingActivationsCache, or NULL on failure
 */
TrainingActivationsCache* create_training_activations_cache(const TransformerConfig* config, int current_batch_size, int current_seq_len) {
    TrainingActivationsCache* cache = (TrainingActivationsCache*)safe_malloc(sizeof(TrainingActivationsCache));
    int num_tokens = current_batch_size * current_seq_len;
    cache->x_embedding_output_batch = create_matrix(num_tokens, config->d_model);

    cache->ln1_means_batch = (Matrix**)safe_calloc(config->n_layers, sizeof(Matrix*));
    cache->ln1_inv_stddevs_batch = (Matrix**)safe_calloc(config->n_layers, sizeof(Matrix*));
    cache->ln2_means_batch = (Matrix**)safe_calloc(config->n_layers, sizeof(Matrix*));
    cache->ln2_inv_stddevs_batch = (Matrix**)safe_calloc(config->n_layers, sizeof(Matrix*));
    cache->ffn_hidden_state = (Matrix**)safe_calloc(config->n_layers, sizeof(Matrix*));
    
    // Allocate enhanced attention activation arrays
    cache->q_projections = (Matrix**)safe_calloc(config->n_layers, sizeof(Matrix*));
    cache->k_projections = (Matrix**)safe_calloc(config->n_layers, sizeof(Matrix*));
    cache->v_projections = (Matrix**)safe_calloc(config->n_layers, sizeof(Matrix*));
    cache->attention_scores = (Matrix**)safe_calloc(config->n_layers, sizeof(Matrix*));
    cache->attention_weights = (Matrix**)safe_calloc(config->n_layers, sizeof(Matrix*));
    cache->attention_outputs = (Matrix**)safe_calloc(config->n_layers, sizeof(Matrix*));
    cache->layer_inputs_ln1 = (Matrix**)safe_calloc(config->n_layers, sizeof(Matrix*));
    cache->layer_inputs_ln2 = (Matrix**)safe_calloc(config->n_layers, sizeof(Matrix*));

    for(int i=0; i < config->n_layers; ++i) {
        // These should store means/inv_stddevs *per token* for the whole batch
        // So, num_tokens x 1 if LayerNorm is applied row-wise.
        cache->ln1_means_batch[i] = create_matrix(num_tokens, 1);
        cache->ln1_inv_stddevs_batch[i] = create_matrix(num_tokens, 1);
        cache->ln2_means_batch[i] = create_matrix(num_tokens, 1);
        cache->ln2_inv_stddevs_batch[i] = create_matrix(num_tokens, 1);
        
        // Allocate FFN hidden state cache
        cache->ffn_hidden_state[i] = create_matrix(num_tokens, config->d_ff);
        
        // Allocate enhanced attention activation caches
        cache->q_projections[i] = create_matrix(num_tokens, config->d_model);
        cache->k_projections[i] = create_matrix(num_tokens, config->d_model);
        cache->v_projections[i] = create_matrix(num_tokens, config->d_model);
        cache->attention_scores[i] = create_matrix(num_tokens, num_tokens);
        cache->attention_weights[i] = create_matrix(num_tokens, num_tokens);
        cache->attention_outputs[i] = create_matrix(num_tokens, config->d_model);
        cache->layer_inputs_ln1[i] = create_matrix(num_tokens, config->d_model);
        cache->layer_inputs_ln2[i] = create_matrix(num_tokens, config->d_model);
    }
    cache->final_ln_means_batch = create_matrix(num_tokens, 1);
    cache->final_ln_inv_stddevs_batch = create_matrix(num_tokens, 1);

    
    return cache;
}

/**
 * Frees all memory associated with a TrainingActivationsCache.
 * This is crucial to prevent memory leaks - always call this for any allocated
 * TrainingActivationsCache structure when it's no longer needed.
 * 
 * @param cache: The TrainingActivationsCache to free
 * @param config: Model configuration (used to determine number of layers)
 */
void free_training_activations_cache(TrainingActivationsCache* cache, const TransformerConfig* config) {
    if (!cache) return;
    free_matrix(cache->x_embedding_output_batch);
    for(int i=0; i < config->n_layers; ++i) {
        free_matrix(cache->ln1_means_batch[i]);
        free_matrix(cache->ln1_inv_stddevs_batch[i]);
        free_matrix(cache->ln2_means_batch[i]);
        free_matrix(cache->ln2_inv_stddevs_batch[i]);
        free_matrix(cache->ffn_hidden_state[i]);
        
        // Free enhanced attention activations
        free_matrix(cache->q_projections[i]);
        free_matrix(cache->k_projections[i]);
        free_matrix(cache->v_projections[i]);
        free_matrix(cache->attention_scores[i]);
        free_matrix(cache->attention_weights[i]);
        free_matrix(cache->attention_outputs[i]);
        free_matrix(cache->layer_inputs_ln1[i]);
        free_matrix(cache->layer_inputs_ln2[i]);
    }
    free(cache->ln1_means_batch);
    free(cache->ln1_inv_stddevs_batch);
    free(cache->ln2_means_batch);
    free(cache->ln2_inv_stddevs_batch);
    free(cache->ffn_hidden_state);
    
    // Free enhanced activation arrays
    free(cache->q_projections);
    free(cache->k_projections);
    free(cache->v_projections);
    free(cache->attention_scores);
    free(cache->attention_weights);
    free(cache->attention_outputs);
    free(cache->layer_inputs_ln1);
    free(cache->layer_inputs_ln2);
    
    free_matrix(cache->final_ln_means_batch);
    free_matrix(cache->final_ln_inv_stddevs_batch);
    free(cache);
}

/* ===========================================================
 * TRANSFORMER STATE MANAGEMENT (FOR INCREMENTAL TEXT GENERATION)
 * =================================================================
 * These functions manage the transformer state - used for efficient autoregressive
 * text generation. The state includes caches for the key and value matrices
 * used in the attention mechanism.
 */

/**
 * Creates a transformer state for incremental text generation.
 * This maintains KV caches for efficient autoregressive generation.
 * 
 * @param config: Model configuration specifying dimensions and layer count
 * @return: Newly allocated TransformerState, or NULL on failure
 */
TransformerState* create_transformer_state(const TransformerConfig* config) {
    if (!config) return NULL;
    
    TransformerState* state = (TransformerState*)safe_malloc(sizeof(TransformerState));
    state->config = config;
    state->current_seq_pos = 0;
    
    // Allocate KV cache arrays for all layers
    state->key_cache = (Matrix**)safe_calloc(config->n_layers, sizeof(Matrix*));
    state->value_cache = (Matrix**)safe_calloc(config->n_layers, sizeof(Matrix*));
    
    // Create cache matrices for each layer
    // Each cache can hold max_seq_len tokens, each with d_model dimensions
    for (int i = 0; i < config->n_layers; i++) {
        state->key_cache[i] = create_matrix(config->max_seq_len, config->d_model);
        state->value_cache[i] = create_matrix(config->max_seq_len, config->d_model);
        
        // Initialize caches to zero
        fill_matrix(state->key_cache[i], 0.0f);
        fill_matrix(state->value_cache[i], 0.0f);
    }
    
    return state;
}

/**
 * Frees all memory associated with a TransformerState.
 * Always call this when done with text generation to prevent memory leaks.
 * 
 * @param state: The TransformerState to free
 * @param config: Model configuration (used to determine number of layers)
 */
void free_transformer_state(TransformerState* state, const TransformerConfig* config) {
    if (!state || !config) return;
    
    // Free all KV cache matrices
    for (int i = 0; i < config->n_layers; i++) {
        free_matrix(state->key_cache[i]);
        free_matrix(state->value_cache[i]);
    }
    
    // Free the cache arrays
    free(state->key_cache);
    free(state->value_cache);
    
    // Free the state structure itself
    free(state);
}

/**
 * Resets the transformer state for a new generation sequence.
 * This clears KV caches and resets position counters.
 * 
 * @param state: The TransformerState to reset
 */
void reset_transformer_state(TransformerState* state) {
    if (!state) return;
    
    // Reset sequence position counter
    state->current_seq_pos = 0;
    
    // Clear all KV caches by filling with zeros
    for (int i = 0; i < state->config->n_layers; i++) {
        fill_matrix(state->key_cache[i], 0.0f);
        fill_matrix(state->value_cache[i], 0.0f);
    }
}

/**
 * Performs forward pass for a single token using KV caching.
 * This is optimized for autoregressive generation where we only need
 * to process one new token at a time.
 * 
 * @param weights: Model weights for computation
 * @param state: Current generation state with KV caches
 * @param token_idx: The token ID to process
 * @return: Logits matrix (1 x vocab_size) for next token prediction
 */
Matrix* transformer_forward_pass_single_token(TransformerWeights* weights, TransformerState* state, int token_idx) {
    if (!weights || !state || token_idx < 0) {
        fprintf(stderr, "Error: Invalid parameters for single token forward pass\n");
        return NULL;
    }
    
    const TransformerConfig* config = weights->config;
    int d_model = config->d_model;
    int vocab_size = config->vocab_size;
    
    // Validate token index bounds
    if (token_idx >= vocab_size) {
        fprintf(stderr, "Warning: Token index %d exceeds vocabulary size %d, using token 0\n", 
                token_idx, vocab_size);
        token_idx = 0; // Use UNK token
    }
    
    // Check if we've exceeded maximum sequence length
    if (state->current_seq_pos >= config->max_seq_len) {
        fprintf(stderr, "Warning: Sequence position %d exceeds maximum length %d\n",
                state->current_seq_pos, config->max_seq_len);
        return NULL;
    }
    
    // Step 1: Embedding lookup (token + positional)
    Matrix* x = create_matrix(1, d_model); // Single token input
    
    // Add token embedding
    for (int d = 0; d < d_model; d++) {
        x->data[d] = weights->token_embeddings->data[token_idx * d_model + d];
    }
    
    // Add positional embedding
    int pos = state->current_seq_pos;
    for (int d = 0; d < d_model; d++) {
        x->data[d] += weights->positional_embeddings->data[pos * d_model + d];
    }
    
    // Step 2: Process through transformer layers
    Matrix* current_x = x;
    
    for (int layer = 0; layer < config->n_layers; layer++) {
        // Layer normalization 1
        Matrix* ln1_out = create_matrix(1, d_model);
        layer_norm(ln1_out, current_x, weights->layers[layer].ln1_gamma, 
                   weights->layers[layer].ln1_beta, config->layer_norm_eps);
        
        // Multi-head attention with KV caching optimization
        Matrix* attn_out = create_matrix(1, d_model);
        
        // ATTENTION WITH KV CACHING:
        // For efficient autoregressive generation, we maintain key and value caches
        // to avoid recomputing attention for previously processed tokens.
        //
        // Implementation steps:
        // 1. Compute Q, K, V for current token
        Matrix* q = create_matrix(1, d_model);
        mat_mul(q, ln1_out, weights->layers[layer].attn_weights.q_proj);
        
        Matrix* k = create_matrix(1, d_model);
        mat_mul(k, ln1_out, weights->layers[layer].attn_weights.k_proj);
        Matrix* v = create_matrix(1, d_model);
        mat_mul(v, ln1_out, weights->layers[layer].attn_weights.v_proj);
        
        // 2. Update KV cache for current layer at current position
        int cache_pos = state->current_seq_pos;
        copy_row(state->key_cache[layer], cache_pos, k, 0);
        copy_row(state->value_cache[layer], cache_pos, v, 0);
        
        // 3. Compute attention using full cached K, V and current Q
        compute_cached_attention(attn_out, q, state->key_cache[layer], 
                                state->value_cache[layer], cache_pos + 1);
        
        // Apply output projection for multi-head attention
        Matrix* projected_attn = create_matrix(1, d_model);
        mat_mul(projected_attn, attn_out, weights->layers[layer].attn_weights.o_proj);
        
        // Copy result back to attn_out
        copy_matrix(attn_out, projected_attn);
        
        // Cleanup temporary matrices
        free_matrix(q);
        free_matrix(k);
        free_matrix(v);
        free_matrix(projected_attn);
        
        // Residual connection
        mat_add(current_x, current_x, attn_out);
        free_matrix(ln1_out);
        free_matrix(attn_out);
        
        // Layer normalization 2
        Matrix* ln2_out = create_matrix(1, d_model);
        layer_norm(ln2_out, current_x, weights->layers[layer].ln2_gamma,
                   weights->layers[layer].ln2_beta, config->layer_norm_eps);
        
        // Feed-forward network
        Matrix* ffn_hidden = create_matrix(1, config->d_ff);
        mat_mul(ffn_hidden, ln2_out, weights->layers[layer].ff_weights.fc1);
        
        // Apply ReLU activation (simple element-wise operation)
        for (int i = 0; i < config->d_ff; i++) {
            if (ffn_hidden->data[i] < 0.0f) {
                ffn_hidden->data[i] = 0.0f; // ReLU: max(0, x)
            }
        }
        
        Matrix* ffn_out = create_matrix(1, d_model);
        mat_mul(ffn_out, ffn_hidden, weights->layers[layer].ff_weights.fc2);
        
        // Residual connection
        mat_add(current_x, current_x, ffn_out);
        
        free_matrix(ln2_out);
        free_matrix(ffn_hidden);
        free_matrix(ffn_out);
    }
    
    // Step 3: Final layer normalization
    Matrix* final_ln_out = create_matrix(1, d_model);
    layer_norm(final_ln_out, current_x, weights->final_ln_gamma,
               weights->final_ln_beta, config->layer_norm_eps);
    
    // Step 4: Output projection to vocabulary
    Matrix* logits = create_matrix(1, vocab_size);
    mat_mul(logits, final_ln_out, weights->output_projection);
    
    // Update sequence position for next token
    state->current_seq_pos++;
    
    // Cleanup intermediate matrices
    free_matrix(current_x);
    free_matrix(final_ln_out);
    
    return logits;
}