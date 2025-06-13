#include "optimizer.h"
#include "utils.h" // For safe_malloc, safe_calloc
#include <math.h>           // For sqrtf, powf
#include <stdio.h>          // For printf if debugging
#include <assert.h>         // For assert

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Helper to create a zero-initialized TransformerWeights structure for m and v
static TransformerWeights* _create_moment_vectors(const TransformerConfig* config) {
    TransformerWeights* moments = create_transformer_weights(config); // Allocates matrices
    // create_transformer_weights uses calloc, so data is already zeroed.
    return moments;
}

AdamOptimizerState* create_adam_optimizer_state(const TransformerConfig* config) {
    AdamOptimizerState* opt_state = (AdamOptimizerState*)safe_malloc(sizeof(AdamOptimizerState));
    opt_state->beta1 = 0.9f;
    opt_state->beta2 = 0.999f;
    opt_state->eps = 1e-8f;
    opt_state->t = 0;

    opt_state->m = _create_moment_vectors(config);
    opt_state->v = _create_moment_vectors(config);
    return opt_state;
}

void free_adam_optimizer_state(AdamOptimizerState* opt_state) {
    if (opt_state) {
        free_transformer_weights(opt_state->m); // Reuses the same free function
        free_transformer_weights(opt_state->v);
        free(opt_state);
    }
}

// Helper for applying Adam update to a single matrix parameter
static void _adam_update_matrix(Matrix* param_matrix, Matrix* grad_matrix,
                               Matrix* m_matrix, Matrix* v_matrix,
                               AdamOptimizerState* opt_state, float learning_rate) {
    if (!param_matrix || !grad_matrix || !m_matrix || !v_matrix) return; // Handle optional params like biases
    
    assert(param_matrix->rows == grad_matrix->rows && param_matrix->cols == grad_matrix->cols);
    assert(m_matrix->rows == grad_matrix->rows && m_matrix->cols == grad_matrix->cols);
    assert(v_matrix->rows == grad_matrix->rows && v_matrix->cols == grad_matrix->cols);

    #pragma omp parallel for
    for (int i = 0; i < param_matrix->rows * param_matrix->cols; ++i) {
        float grad_val = grad_matrix->data[i];

        // Update biased first moment estimate
        m_matrix->data[i] = opt_state->beta1 * m_matrix->data[i] + (1.0f - opt_state->beta1) * grad_val;
        // Update biased second raw moment estimate
        v_matrix->data[i] = opt_state->beta2 * v_matrix->data[i] + (1.0f - opt_state->beta2) * (grad_val * grad_val);

        // Compute bias-corrected first moment estimate
        float m_hat = m_matrix->data[i] / (1.0f - powf(opt_state->beta1, opt_state->t));
        // Compute bias-corrected second raw moment estimate
        float v_hat = v_matrix->data[i] / (1.0f - powf(opt_state->beta2, opt_state->t));

        // Update parameters
        param_matrix->data[i] -= learning_rate * m_hat / (sqrtf(v_hat) + opt_state->eps);
    }
}

void adam_update(TransformerWeights* params,
                 TransformerGradients* grads,
                 AdamOptimizerState* opt_state,
                 float learning_rate,
                 float grad_clip_norm_config) { // Renamed to avoid conflict
    opt_state->t++; // Increment timestep

    if (grad_clip_norm_config > 0.0f) {
        clip_gradients_global_norm(grads, grad_clip_norm_config);
    }

    // Update all parameters
    _adam_update_matrix(params->token_embeddings, grads->token_embeddings, opt_state->m->token_embeddings, opt_state->v->token_embeddings, opt_state, learning_rate);
    _adam_update_matrix(params->positional_embeddings, grads->positional_embeddings, opt_state->m->positional_embeddings, opt_state->v->positional_embeddings, opt_state, learning_rate);

    for (int i = 0; i < params->config->n_layers; ++i) {
        AttentionWeights* p_att = &params->layers[i].attn_weights;
        AttentionWeights* g_att = &grads->layers[i].attn_weights;
        AttentionWeights* m_att = &opt_state->m->layers[i].attn_weights;
        AttentionWeights* v_att = &opt_state->v->layers[i].attn_weights;
        _adam_update_matrix(p_att->q_proj, g_att->q_proj, m_att->q_proj, v_att->q_proj, opt_state, learning_rate);
        _adam_update_matrix(p_att->k_proj, g_att->k_proj, m_att->k_proj, v_att->k_proj, opt_state, learning_rate);
        _adam_update_matrix(p_att->v_proj, g_att->v_proj, m_att->v_proj, v_att->v_proj, opt_state, learning_rate);
        _adam_update_matrix(p_att->o_proj, g_att->o_proj, m_att->o_proj, v_att->o_proj, opt_state, learning_rate);
        // Add biases if they exist (e.g., p_att->bq, g_att->bq, m_att->bq, v_att->bq)

        FeedForwardWeights* p_ffn = &params->layers[i].ff_weights;
        FeedForwardWeights* g_ffn = &grads->layers[i].ff_weights;
        FeedForwardWeights* m_ffn = &opt_state->m->layers[i].ff_weights;
        FeedForwardWeights* v_ffn = &opt_state->v->layers[i].ff_weights;
        _adam_update_matrix(p_ffn->fc1, g_ffn->fc1, m_ffn->fc1, v_ffn->fc1, opt_state, learning_rate);
        _adam_update_matrix(p_ffn->fc2, g_ffn->fc2, m_ffn->fc2, v_ffn->fc2, opt_state, learning_rate);
        // Note: No biases in current FeedForwardWeights struct

        _adam_update_matrix(params->layers[i].ln1_gamma, grads->layers[i].ln1_gamma, opt_state->m->layers[i].ln1_gamma, opt_state->v->layers[i].ln1_gamma, opt_state, learning_rate);
        _adam_update_matrix(params->layers[i].ln1_beta, grads->layers[i].ln1_beta, opt_state->m->layers[i].ln1_beta, opt_state->v->layers[i].ln1_beta, opt_state, learning_rate);
        _adam_update_matrix(params->layers[i].ln2_gamma, grads->layers[i].ln2_gamma, opt_state->m->layers[i].ln2_gamma, opt_state->v->layers[i].ln2_gamma, opt_state, learning_rate);
        _adam_update_matrix(params->layers[i].ln2_beta, grads->layers[i].ln2_beta, opt_state->m->layers[i].ln2_beta, opt_state->v->layers[i].ln2_beta, opt_state, learning_rate);
    }

    _adam_update_matrix(params->final_ln_gamma, grads->final_ln_gamma, opt_state->m->final_ln_gamma, opt_state->v->final_ln_gamma, opt_state, learning_rate);
    _adam_update_matrix(params->final_ln_beta, grads->final_ln_beta, opt_state->m->final_ln_beta, opt_state->v->final_ln_beta, opt_state, learning_rate);
    _adam_update_matrix(params->output_projection, grads->output_projection, opt_state->m->output_projection, opt_state->v->output_projection, opt_state, learning_rate);
    // Add output_projection_b if it exists
}

// Learning rate scheduling functions
float cosine_decay_lr(float initial_lr, int current_step, int total_steps) {
    if (current_step >= total_steps) {
        return 0.0f;
    }
    
    float progress = (float)current_step / (float)total_steps;
    return initial_lr * 0.5f * (1.0f + cosf(M_PI * progress));
}

float warmup_cosine_lr(float initial_lr, int current_step, int warmup_steps, int total_steps) {
    if (current_step < warmup_steps) {
        // Linear warmup
        return initial_lr * (float)current_step / (float)warmup_steps;
    } else {
        // Cosine decay after warmup
        int decay_steps = current_step - warmup_steps;
        int total_decay_steps = total_steps - warmup_steps;
        return cosine_decay_lr(initial_lr, decay_steps, total_decay_steps);
    }
}

float exponential_decay_lr(float initial_lr, int current_step, float decay_rate, int decay_steps) {
    int num_decays = current_step / decay_steps;
    return initial_lr * powf(decay_rate, (float)num_decays);
}

// Enhanced Adam optimizer state creation with scheduling support
AdamOptimizerState* create_enhanced_adam_optimizer_state(const TransformerConfig* config, 
                                                        float initial_lr, 
                                                        int warmup_steps, 
                                                        int total_steps) {
    AdamOptimizerState* opt_state = create_adam_optimizer_state(config);
    opt_state->initial_lr = initial_lr;
    opt_state->warmup_steps = warmup_steps;
    opt_state->total_steps = total_steps;
    return opt_state;
}

// Enhanced Adam update with learning rate scheduling
void adam_update_with_schedule(TransformerWeights* params,
                              TransformerGradients* grads,
                              AdamOptimizerState* opt_state,
                              float grad_clip_norm_config) {
    opt_state->t++; // Increment timestep
    
    // Calculate learning rate with warmup and cosine decay
    float current_lr = warmup_cosine_lr(opt_state->initial_lr, 
                                       opt_state->t, 
                                       opt_state->warmup_steps, 
                                       opt_state->total_steps);
    
    if (grad_clip_norm_config > 0.0f) {
        clip_gradients_global_norm(grads, grad_clip_norm_config);
    }

    // Update all parameters with scheduled learning rate
    _adam_update_matrix(params->token_embeddings, grads->token_embeddings, 
                       opt_state->m->token_embeddings, opt_state->v->token_embeddings, 
                       opt_state, current_lr);
    _adam_update_matrix(params->positional_embeddings, grads->positional_embeddings, 
                       opt_state->m->positional_embeddings, opt_state->v->positional_embeddings, 
                       opt_state, current_lr);

    for (int i = 0; i < params->config->n_layers; ++i) {
        AttentionWeights* p_att = &params->layers[i].attn_weights;
        AttentionWeights* g_att = &grads->layers[i].attn_weights;
        AttentionWeights* m_att = &opt_state->m->layers[i].attn_weights;
        AttentionWeights* v_att = &opt_state->v->layers[i].attn_weights;
        _adam_update_matrix(p_att->q_proj, g_att->q_proj, m_att->q_proj, v_att->q_proj, opt_state, current_lr);
        _adam_update_matrix(p_att->k_proj, g_att->k_proj, m_att->k_proj, v_att->k_proj, opt_state, current_lr);
        _adam_update_matrix(p_att->v_proj, g_att->v_proj, m_att->v_proj, v_att->v_proj, opt_state, current_lr);
        _adam_update_matrix(p_att->o_proj, g_att->o_proj, m_att->o_proj, v_att->o_proj, opt_state, current_lr);

        FeedForwardWeights* p_ffn = &params->layers[i].ff_weights;
        FeedForwardWeights* g_ffn = &grads->layers[i].ff_weights;
        FeedForwardWeights* m_ffn = &opt_state->m->layers[i].ff_weights;
        FeedForwardWeights* v_ffn = &opt_state->v->layers[i].ff_weights;
        _adam_update_matrix(p_ffn->fc1, g_ffn->fc1, m_ffn->fc1, v_ffn->fc1, opt_state, current_lr);
        _adam_update_matrix(p_ffn->fc2, g_ffn->fc2, m_ffn->fc2, v_ffn->fc2, opt_state, current_lr);

        _adam_update_matrix(params->layers[i].ln1_gamma, grads->layers[i].ln1_gamma, 
                           opt_state->m->layers[i].ln1_gamma, opt_state->v->layers[i].ln1_gamma, 
                           opt_state, current_lr);
        _adam_update_matrix(params->layers[i].ln1_beta, grads->layers[i].ln1_beta, 
                           opt_state->m->layers[i].ln1_beta, opt_state->v->layers[i].ln1_beta, 
                           opt_state, current_lr);
        _adam_update_matrix(params->layers[i].ln2_gamma, grads->layers[i].ln2_gamma, 
                           opt_state->m->layers[i].ln2_gamma, opt_state->v->layers[i].ln2_gamma, 
                           opt_state, current_lr);
        _adam_update_matrix(params->layers[i].ln2_beta, grads->layers[i].ln2_beta, 
                           opt_state->m->layers[i].ln2_beta, opt_state->v->layers[i].ln2_beta, 
                           opt_state, current_lr);
    }

    _adam_update_matrix(params->final_ln_gamma, grads->final_ln_gamma, 
                       opt_state->m->final_ln_gamma, opt_state->v->final_ln_gamma, 
                       opt_state, current_lr);
    _adam_update_matrix(params->final_ln_beta, grads->final_ln_beta, 
                       opt_state->m->final_ln_beta, opt_state->v->final_ln_beta, 
                       opt_state, current_lr);
    _adam_update_matrix(params->output_projection, grads->output_projection, 
                       opt_state->m->output_projection, opt_state->v->output_projection, 
                       opt_state, current_lr);
}