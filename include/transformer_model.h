#ifndef TRANSFORMER_MODEL_H
#define TRANSFORMER_MODEL_H

#include "matrix.h"
#include "transformer_config.h"

// --- Weight Structures ---
typedef struct {
    Matrix* q_proj; // d_model x d_model
    Matrix* k_proj; // d_model x d_model
    Matrix* v_proj; // d_model x d_model
    Matrix* o_proj; // d_model x d_model
    // No biases for simplicity, or add them: Matrix* q_bias, k_bias, v_bias, o_bias;
} AttentionWeights;

typedef struct {
    Matrix* fc1;    // d_model x d_ff
    Matrix* fc2;    // d_ff x d_model
    // No biases for simplicity, or add them: Matrix* fc1_bias, fc2_bias;
} FeedForwardWeights;

typedef struct {
    AttentionWeights attn_weights;
    FeedForwardWeights ff_weights;
    Matrix* ln1_gamma; // LayerNorm scale (d_model x 1)
    Matrix* ln1_beta;  // LayerNorm shift (d_model x 1)
    Matrix* ln2_gamma;
    Matrix* ln2_beta;
} TransformerLayerWeights;

typedef struct {
    const TransformerConfig* config; // Pointer to the model configuration
    Matrix* token_embeddings;       // vocab_size x d_model
    Matrix* positional_embeddings;  // max_seq_len x d_model
    TransformerLayerWeights* layers; // Array of n_layers
    Matrix* final_ln_gamma;         // For final LayerNorm before output projection
    Matrix* final_ln_beta;
    Matrix* output_projection;      // d_model x vocab_size (sometimes called LM head)
} TransformerWeights;


// --- Gradient Structure (mirrors TransformerWeights) ---
typedef TransformerWeights TransformerGradients;

TransformerGradients* create_transformer_gradients(const TransformerConfig* config);
void free_transformer_gradients(TransformerGradients* grads);
void zero_gradients(TransformerGradients* grads); // Fills all gradient matrices with 0.0f
// Calculates global L2 norm of all gradients and scales them if norm > max_norm
void clip_gradients_global_norm(TransformerGradients* grads, float max_norm);


// --- State Structures ---
// TransformerState (for single-token generation with KV cache)
typedef struct {
    const TransformerConfig* config;    // Pointer to model config
    Matrix** key_cache;               // Array[num_layers] of (max_seq_len x d_model) matrices
    Matrix** value_cache;             // Array[num_layers] of (max_seq_len x d_model) matrices
    int current_seq_pos;              // Current position in the sequence for KV cache update (0 to max_seq_len-1)
} TransformerState;

// TrainingActivations (to store intermediate values for backpropagation for a batch/sequence)
// This is highly complex and memory intensive. A simplified version:
typedef struct {
    // For each layer, store inputs and outputs of key components.
    // Example: Q, K, V projections, attention scores, norm inputs/outputs, FFN hidden state.
    // This would be an array of per-layer activation structs.
    // For simplicity, specific matrices will be passed around or recomputed where feasible.
    // A full system needs a dedicated activation manager.
    Matrix* x_embedding_output_batch; // batch_size * seq_len x d_model (or list of matrices)
    Matrix** ffn_hidden_state; // array[num_layers] of (batch_size * seq_len x d_ff)

    // Store means and inv_stddevs from LayerNorm forward pass for its backward pass
    Matrix** ln1_means_batch; // array[num_layers] of (batch_size * seq_len x 1)
    Matrix** ln1_inv_stddevs_batch;
    Matrix** ln2_means_batch;
    Matrix** ln2_inv_stddevs_batch;
    Matrix* final_ln_means_batch;
    Matrix* final_ln_inv_stddevs_batch;

    // ENHANCED: Attention-related activations for complete backward pass
    Matrix** q_projections; // array[num_layers] of (batch_size * seq_len x d_model)
    Matrix** k_projections; // array[num_layers] of (batch_size * seq_len x d_model)
    Matrix** v_projections; // array[num_layers] of (batch_size * seq_len x d_model)
    Matrix** attention_scores; // array[num_layers] of (batch_size * seq_len x seq_len)
    Matrix** attention_weights; // array[num_layers] of (batch_size * seq_len x seq_len) - after softmax
    Matrix** attention_outputs; // array[num_layers] of (batch_size * seq_len x d_model)

    // ENHANCED: Residual connection inputs for proper gradient flow
    Matrix** layer_inputs_ln1; // array[num_layers] of (batch_size * seq_len x d_model)
    Matrix** layer_inputs_ln2; // array[num_layers] of (batch_size * seq_len x d_model)
} TrainingActivationsCache;

TrainingActivationsCache* create_training_activations_cache(const TransformerConfig* config, int current_batch_size, int current_seq_len);
void free_training_activations_cache(TrainingActivationsCache* cache, const TransformerConfig* config);


// --- Model Creation/Deletion (from before) ---
TransformerWeights* create_transformer_weights(const TransformerConfig* config);
void free_transformer_weights(TransformerWeights* weights);
void init_dummy_weights(TransformerWeights* weights); // For testing without loading

// --- Weight I/O ---
int save_transformer_weights(const TransformerWeights* weights, const char* path);
int load_transformer_weights(TransformerWeights* weights, const char* path); // Ensure weights config matches file or is updated

// --- Inference (single token, from before) ---
TransformerState* create_transformer_state(const TransformerConfig* config);
void free_transformer_state(TransformerState* state, const TransformerConfig* config);
void reset_transformer_state(TransformerState* state);
Matrix* transformer_forward_pass_single_token(TransformerWeights* weights, TransformerState* state, int token_idx);


// --- Training (batch processing) ---
// input_batch: batch_size x seq_len matrix of token IDs
// target_batch: batch_size x seq_len matrix of target token IDs
// This function will perform forward pass, calculate loss, perform backward pass to populate `grads`.
// Returns the average loss for the batch.
float transformer_train_batch(
    TransformerWeights* weights,
    TransformerGradients* grads, // Gradients are accumulated here
    TrainingActivationsCache* activations_cache, // For storing/retrieving activations
    const Matrix* input_token_ids_batch, // batch_size x current_seq_len
    const Matrix* target_token_ids_batch, // batch_size x current_seq_len
    Matrix* causal_mask, // seq_len x seq_len, precomputed
    const TransformerConfig* config,
    int is_first_batch_of_epoch // To manage state if needed, e.g., for printing
);

// --- Utility / Helper function declarations that might be used by transformer_model.c ---
// Ensure these are implemented in matrix.c or another appropriate .c file and declared here if needed by transformer_model.c
// If they are purely internal to matrix.c, they don't need to be here.

// From matrix.c, if used by transformer_model.c directly and not just internally by other matrix functions
float softmax_cross_entropy_loss_and_backward(
    Matrix* grad_logits, // Output: dL/dLogits (num_tokens x vocab_size)
    const Matrix* logits, // Input: Logits from model (num_tokens x vocab_size)
    const int* targets_flat, // Input: Flattened target token IDs (num_tokens)
    int num_tokens, // Total number of tokens in the batch (batch_size * seq_len)
    int vocab_size // Vocabulary size
);

void layer_norm_backward(
    Matrix* grad_input_acc, // Output: dL/dX (accumulated)
    Matrix* grad_gain_acc,  // Output: dL/dGain (accumulated)
    Matrix* grad_bias_acc,  // Output: dL/dBias (accumulated)
    const Matrix* grad_output, // Input: dL/dY (gradient from subsequent layer)
    const Matrix* input_forward, // Input: X from forward pass
    const Matrix* gain_forward,  // Input: Gain (gamma) from forward pass
    const Matrix* means_forward, // Input: Mean(X) from forward pass
    const Matrix* inv_stddevs_forward, // Input: 1/sqrt(Var(X)+eps) from forward pass
    float eps // Epsilon value used in forward pass
);

void mat_mul_add(Matrix* C_acc, const Matrix* A, const Matrix* B);

void gelu_activation_backward(Matrix* grad_input_acc, const Matrix* input_forward, const Matrix* grad_output);


#endif // TRANSFORMER_MODEL_H