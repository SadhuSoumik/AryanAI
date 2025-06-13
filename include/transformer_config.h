/**
 * @file transformer_config.h
 * @brief Configuration structure for the transformer model
 * 
 * 
 * This file defines all the "settings" or "hyperparameters" that control
 * how the transformer model behaves. Think of these as the "knobs" you can
 * turn to change how the model works.
 * 
 * KEY CONCEPTS:
 * - vocab_size: How many different words/tokens the model knows
 * - max_seq_len: Maximum length of text the model can process at once
 * - d_model: Size of the internal representations (bigger = more complex)
 * - n_layers: How many transformer layers to stack (deeper = more powerful)
 * - n_heads: Number of attention mechanisms working in parallel
 * 
 *
 */

#ifndef TRANSFORMER_CONFIG_H
#define TRANSFORMER_CONFIG_H

/**
 * Configuration structure containing all transformer model parameters.
 * 
 * This structure is like a "recipe card" that tells the model exactly
 * how to configure itself. Every aspect of the model's architecture
 * and training process is controlled by these parameters.
 */
typedef struct {
    // === MODEL ARCHITECTURE PARAMETERS ===
    int vocab_size;         // Number of unique tokens in vocabulary (e.g., 10000)
    int max_seq_len;        // Maximum sequence length - context window (e.g., 512, 1024, 2048)
    int d_model;            // Model dimension - size of embeddings (e.g., 512, 768, 1024)
    int n_layers;           // Number of transformer layers to stack (e.g., 6, 12, 24)
    int n_heads;            // Number of attention heads per layer (e.g., 8, 12, 16)
    int d_ff;               // Feed-forward network dimension (usually 4 * d_model)
    float layer_norm_eps;   // Small value to prevent division by zero in normalization (e.g., 1e-5)

    // === TRAINING PARAMETERS ===
    float learning_rate;        // How fast the model learns (e.g., 1e-4, 3e-4)
    int batch_size;            // Number of examples processed together (e.g., 8, 16, 32)
    int num_epochs;            // How many times to go through the entire dataset
    float grad_clip_norm;      // Maximum gradient norm to prevent exploding gradients (0 = no clipping)
    
    // === LOGGING AND SAVING ===
    int save_every_n_epochs;      // How often to save model checkpoints during training
    int print_every_n_batches;    // How often to print training progress

    // === FUTURE EXTENSIONS ===
    // float dropout_rate; // Regularization technique to prevent overfitting (not implemented yet 🥹)

} TransformerConfig;

#endif // TRANSFORMER_CONFIG_H