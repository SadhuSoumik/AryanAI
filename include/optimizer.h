#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "transformer_model.h" // For TransformerWeights, TransformerGradients

// Forward declare TransformerGradients if it's defined after TransformerWeights
// typedef struct TransformerWeights TransformerGradients; // Already good

typedef struct {
    float beta1;
    float beta2;
    float eps;
    int t; // Timestep, for bias correction
    
    // Enhanced accuracy features
    float initial_lr;       // Initial learning rate for scheduling
    int warmup_steps;       // Number of warmup steps
    int total_steps;        // Total training steps for cosine decay

    TransformerWeights* m; // First moment estimates (like grads structure)
    TransformerWeights* v; // Second moment estimates ( " )
} AdamOptimizerState;

// Learning rate scheduling functions for better accuracy
float cosine_decay_lr(float initial_lr, int current_step, int total_steps);
float warmup_cosine_lr(float initial_lr, int current_step, int warmup_steps, int total_steps);
float exponential_decay_lr(float initial_lr, int current_step, float decay_rate, int decay_steps);

AdamOptimizerState* create_adam_optimizer_state(const TransformerConfig* config);

// Adam optimizer creation with learning rate scheduling for better accuracy
AdamOptimizerState* create_enhanced_adam_optimizer_state(const TransformerConfig* config,
                                                        float initial_lr,
                                                        int warmup_steps,
                                                        int total_steps);

void free_adam_optimizer_state(AdamOptimizerState* opt_state);

// Enhanced Adam optimizer with learning rate scheduling
void adam_update_with_schedule(TransformerWeights* params,
                              TransformerGradients* grads,
                              AdamOptimizerState* opt_state,
                              float grad_clip_norm_config);

void adam_update(TransformerWeights* params,
                 TransformerGradients* grads,
                 AdamOptimizerState* opt_state,
                 float learning_rate,
                 float grad_clip_norm); // Pass grad_clip_norm from config

#endif // OPTIMIZER_H