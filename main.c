/**
 * @file main.c
 * @brief Main entry point for the C Transformer project
 * 
 * 🎯 WHAT THIS PROGRAM DOES
 * =========================
 * This is a complete transformer neural network implementation that can:
 * 1. TRAIN: Learn patterns from text data to predict next words
 * 2. GENERATE: Use trained models to create new text from prompts
 * 
 * 🚀 GETTING STARTED
 * ==================
    *Linux =>
 * Training:   ./AryanAi train --data text.txt --epochs 10
 * Generating: ./AryanAi generate --prompt "Hello" --max_new 50

    *Windows =>
 * Training:   ./AryanAi.exe train --data text.txt --epochs 10
 * Generating: ./AryanAi.exe generate --prompt "Hello" --max_new 50
 * 
 * 📚 CODE ORGANIZATION
 * ===================
 * - This file: Command-line interface, main training/generation loops
 * - transformer_model.c: Core AI logic (attention, layers, forward/backward pass)
 * - matrix.c: Mathematical operations (the "calculator" for the AI)
 * - tokenizer.c: Text processing (converting between text and numbers)
 * - optimizer.c: Learning algorithms (how the AI improves itself)
 */

/* 
 * CROSS-PLATFORM COMPATIBILITY SETUP
 * ===============================
 * These macros ensure our shitty code works on Windows, Linux, and macOS
 */
#ifndef _WIN32
    #define _POSIX_C_SOURCE 200809L /* For POSIX.1-2008 features */
#endif

// ==================== STANDARD LIBRARY INCLUDES ========
#include <stdio.h>      // For file operations, printf, scanf
#include <stdlib.h>     // For memory allocation, exit codes
#include <string.h>     // For string manipulation functions
#include <time.h>       // For timing and random number seeding
#include <math.h>       // For mathematical functions like expf()

// Platform-specific includes for argument parsing
#ifdef PLATFORM_WINDOWS
    // Windows doesn't have argp.h, we'll use a simple fallback (Bad Windows😤)
    #define ARGP_NOT_AVAILABLE
#else
    #include <argp.h>       // For command-line argument parsing
#endif

// ==================== PROJECT-SPECIFIC INCLUDES ============
#include "platform.h"          // Cross-platform compatibility functions
#include "transformer_config.h" // Model configuration structure
#include "transformer_model.h"  // Core transformer implementation
#include "matrix.h"             // Matrix operations for neural network math
#include "tokenizer.h"          // Text-to-token conversion
#include "optimizer.h"          // Training algorithms 
#include "utils.h"              // Utility functions (error handling, memory management)
#include "interactive_cli.h"    // Interactive command line interface

// ==================== FORWARD DECLARATIONS ======
int sample_nucleus(Matrix* logits, float top_p, int vocab_size);
char* tokenizer_decode_single(const Tokenizer* tokenizer, int token_id);

/**
 * ==================================================
 * COMMAND-LINE INTERFACE SECTION
 * =========================================
 * 
 * This section uses the 'argp' library to create a user-friendly command-line
 * interface. Users can specify:
 * - Whether to train or generate
 * - File paths for data, models, vocabularies
 * - Model hyperparameters (size, learning rate, etc.)
 * - Generation parameters (prompt, length, etc.)
 */

// Documentation string shown when ran with --help flag
static char doc[] = "AryanAi -- A minimal transformer model in C for NLP tasks.";
static char args_doc[] = "[train|generate] [OPTIONS]";

// Definition of all command-line options available for use
// Each option includes: long_name, short_key, argument_type, flags, help_text, group
#ifndef ARGP_NOT_AVAILABLE
static struct argp_option options[] = {
    {"mode", 'm', "MODE", 0, "Execution mode: 'train' or 'generate' (required)", 0},
    {"config", 'c', "FILE", 0, "Path to model configuration JSON/INI (not implemented, use defaults or individual flags)", 0},
    {"data", 'd', "FILE", 0, "Path to training data text file (for 'train' mode)", 0},
    {"vocab_load", 'v', "FILE", 0, "Path to load tokenizer vocabulary from", 0},
    {"vocab_save", 'V', "FILE", 0, "Path to save tokenizer vocabulary to (after building if 'data' is provided)", 0},
    {"weights_load", 'w', "FILE", 0, "Path to load model weights from", 0},
    {"weights_save", 'W', "FILE", 0, "Path to save model weights to (during/after training or for conversion)", 0},
    {"prompt", 'p', "TEXT", 0, "Prompt text for generation (for 'generate' mode)", 0},
    {"max_new", 'n', "NUM", 0, "Max new tokens to generate (default: 100)", 0},
    {"lr", 'l', "RATE", 0, "Learning rate (default: 1e-4)", 0},
    {"epochs", 'e', "NUM", 0, "Number of training epochs (default: 1)", 0},
    {"batch_size", 'b', "SIZE", 0, "Batch size for training (default: 8)", 0},
    {"seq_len", 's', "LEN", 0, "Max sequence length (default: 256)", 0},
    {"d_model", 'H', "DIM", 0, "Model hidden dimension (default: 128)", 0},
    {"n_layers", 'L', "NUM", 0, "Number of transformer layers (default: 3)", 0},
    {"n_heads", 'A', "NUM", 0, "Number of attention heads (default: 4)", 0},
    {"grad_accum", 'g', "STEPS", 0, "Gradient accumulation steps (default: 1)", 0},
    {0}};  // Terminator entry - all fields zero
#endif

/**
 * Structure to hold all parsed command-line arguments and model configuration.
 * This serves as the central configuration object for the entire program.
 */
typedef struct
{
    char *mode;                  // "train" or "generate"
    char *data_path;             // Path to training data file
    char *vocab_load_path;       // Path to load vocabulary from
    char *vocab_save_path;       // Path to save vocabulary to
    char *weights_load_path;     // Path to load model weights from
    char *weights_save_path;     // Path to save model weights to
    char *prompt;                // Text prompt for generation
    int max_new_tokens;          // Maximum tokens to generate
    int grad_accumulation_steps; // Number of steps to accumulate gradients
    TransformerConfig model_cfg; // Model hyperparameters
} Arguments;

/**
 * Callback function called by argp for each command-line argument.
 * This function parses individual arguments and stores them in the Arguments struct.
 */
#ifndef ARGP_NOT_AVAILABLE
static error_t parse_opt(int key, char *arg, struct argp_state *state)
{
    Arguments *arguments = (Arguments *)state->input;
    switch (key)
    {
    case 'm':
        arguments->mode = arg;
        break;
    case 'd':
        arguments->data_path = arg;
        break;
    case 'v':
        arguments->vocab_load_path = arg;
        break;
    case 'V':
        arguments->vocab_save_path = arg;
        break;
    case 'w':
        arguments->weights_load_path = arg;
        break;
    case 'W':
        arguments->weights_save_path = arg;
        break;
    case 'p':
        arguments->prompt = arg;
        break;
    case 'n':
        arguments->max_new_tokens = atoi(arg);
        break;
    case 'l':
        arguments->model_cfg.learning_rate = atof(arg);
        break;
    case 'e':
        arguments->model_cfg.num_epochs = atoi(arg);
        break;
    case 'b':
        arguments->model_cfg.batch_size = atoi(arg);
        break;
    case 's':
        arguments->model_cfg.max_seq_len = atoi(arg);
        break;
    case 'H':
        arguments->model_cfg.d_model = atoi(arg);
        break;
    case 'L':
        arguments->model_cfg.n_layers = atoi(arg);
        break;
    case 'A':
        arguments->model_cfg.n_heads = atoi(arg);
        break;
    case 'g':
        arguments->grad_accumulation_steps = atoi(arg);
        break;
    case ARGP_KEY_END:
        // Validates that required arguments are provided
        if (!arguments->mode)
        {
            argp_usage(state);
        }
        break;
    default:
        return ARGP_ERR_UNKNOWN;
    }
    return 0;
}

// Initialize argp parser with all required fields
// Format: {options, parser_function, args_doc, doc, children, help_filter, argp_domain}
static struct argp argp_parser = {options, parse_opt, args_doc, doc, 0, 0, 0};
#endif

#ifdef ARGP_NOT_AVAILABLE
/**
 * Simple Windows-compatible argument parser fallback
 * Since Windows/MinGW doesn't have argp.h, we implement basic parsing (It ain't great, but it works)
 */
static void print_help(const char* program_name) {
    printf("Usage: %s --mode [train|generate] [OPTIONS]\n\n", program_name);
    printf("AryanAi -- A minimal transformer model in C for NLP tasks.\n\n");
    printf("Options:\n");
    printf("  -m, --mode MODE          Execution mode: 'train' or 'generate' (required)\n");
    printf("  -d, --data FILE          Path to training data text file (for 'train' mode)\n");
    printf("  -v, --vocab_load FILE    Path to load tokenizer vocabulary from\n");
    printf("  -V, --vocab_save FILE    Path to save tokenizer vocabulary to\n");
    printf("  -w, --weights_load FILE  Path to load model weights from\n");
    printf("  -W, --weights_save FILE  Path to save model weights to\n");
    printf("  -p, --prompt TEXT        Prompt text for generation (for 'generate' mode)\n");
    printf("  -n, --max_new NUM        Max new tokens to generate (default: 100)\n");
    printf("  -l, --lr RATE            Learning rate (default: 1e-4)\n");
    printf("  -e, --epochs NUM         Number of training epochs (default: 1)\n");
    printf("  -b, --batch_size SIZE    Batch size for training (default: 8)\n");
    printf("  -s, --seq_len LEN        Max sequence length (default: 256)\n");
    printf("  -H, --d_model DIM        Model hidden dimension (default: 128)\n");
    printf("  -L, --n_layers NUM       Number of transformer layers (default: 3)\n");
    printf("  -A, --n_heads NUM        Number of attention heads (default: 4)\n");
    printf("  -g, --grad_accum STEPS   Gradient accumulation steps (default: 1)\n");
    printf("  -h, --help               Show this help message\n");
}

static int parse_args_simple(int argc, char* argv[], Arguments* arguments) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--mode") == 0) {
            if (++i >= argc) { printf("Error: --mode requires an argument\n"); return -1; }
            arguments->mode = argv[i];
        }
        else if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--data") == 0) {
            if (++i >= argc) { printf("Error: --data requires an argument\n"); return -1; }
            arguments->data_path = argv[i];
        }
        else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--vocab_load") == 0) {
            if (++i >= argc) { printf("Error: --vocab_load requires an argument\n"); return -1; }
            arguments->vocab_load_path = argv[i];
        }
        else if (strcmp(argv[i], "-V") == 0 || strcmp(argv[i], "--vocab_save") == 0) {
            if (++i >= argc) { printf("Error: --vocab_save requires an argument\n"); return -1; }
            arguments->vocab_save_path = argv[i];
        }
        else if (strcmp(argv[i], "-w") == 0 || strcmp(argv[i], "--weights_load") == 0) {
            if (++i >= argc) { printf("Error: --weights_load requires an argument\n"); return -1; }
            arguments->weights_load_path = argv[i];
        }
        else if (strcmp(argv[i], "-W") == 0 || strcmp(argv[i], "--weights_save") == 0) {
            if (++i >= argc) { printf("Error: --weights_save requires an argument\n"); return -1; }
            arguments->weights_save_path = argv[i];
        }
        else if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--prompt") == 0) {
            if (++i >= argc) { printf("Error: --prompt requires an argument\n"); return -1; }
            arguments->prompt = argv[i];
        }
        else if (strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--max_new") == 0) {
            if (++i >= argc) { printf("Error: --max_new requires an argument\n"); return -1; }
            arguments->max_new_tokens = atoi(argv[i]);
        }
        else if (strcmp(argv[i], "-l") == 0 || strcmp(argv[i], "--lr") == 0) {
            if (++i >= argc) { printf("Error: --lr requires an argument\n"); return -1; }
            arguments->model_cfg.learning_rate = atof(argv[i]);
        }
        else if (strcmp(argv[i], "-e") == 0 || strcmp(argv[i], "--epochs") == 0) {
            if (++i >= argc) { printf("Error: --epochs requires an argument\n"); return -1; }
            arguments->model_cfg.num_epochs = atoi(argv[i]);
        }
        else if (strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--batch_size") == 0) {
            if (++i >= argc) { printf("Error: --batch_size requires an argument\n"); return -1; }
            arguments->model_cfg.batch_size = atoi(argv[i]);
        }
        else if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--seq_len") == 0) {
            if (++i >= argc) { printf("Error: --seq_len requires an argument\n"); return -1; }
            arguments->model_cfg.max_seq_len = atoi(argv[i]);
        }
        else if (strcmp(argv[i], "-H") == 0 || strcmp(argv[i], "--d_model") == 0) {
            if (++i >= argc) { printf("Error: --d_model requires an argument\n"); return -1; }
            arguments->model_cfg.d_model = atoi(argv[i]);
        }
        else if (strcmp(argv[i], "-L") == 0 || strcmp(argv[i], "--n_layers") == 0) {
            if (++i >= argc) { printf("Error: --n_layers requires an argument\n"); return -1; }
            arguments->model_cfg.n_layers = atoi(argv[i]);
        }
        else if (strcmp(argv[i], "-A") == 0 || strcmp(argv[i], "--n_heads") == 0) {
            if (++i >= argc) { printf("Error: --n_heads requires an argument\n"); return -1; }
            arguments->model_cfg.n_heads = atoi(argv[i]);
        }
        else if (strcmp(argv[i], "-g") == 0 || strcmp(argv[i], "--grad_accum") == 0) {
            if (++i >= argc) { printf("Error: --grad_accum requires an argument\n"); return -1; }
            arguments->grad_accumulation_steps = atoi(argv[i]);
        }
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_help(argv[0]);
            return 1; // Indicate help was shown
        }
        else {
            printf("Error: Unknown argument '%s'\n", argv[i]);
            print_help(argv[0]);
            return -1;
        }
    }
    
    // Validate required arguments
    if (!arguments->mode) {
        printf("Error: --mode is required\n");
        print_help(argv[0]);
        return -1;
    }
    
    return 0; // Success 😎
}
#endif

/**
 * =============================================
 * UTILITY FUNCTION IMPLEMENTATIONS
 * ================================================
 * Helper functions that are used by the main program but not in other modules.
 */

/**
 * Structure to hold token probability pairs for sorting in nucleus sampling.
 */
typedef struct {
    int token_id;
    float probability;
} TokenProb;

/**
 * Comparison function for sorting TokenProb structures by probability (descending).
 */
static int compare_token_probs(const void *a, const void *b) {
    const TokenProb *ta = (const TokenProb *)a;
    const TokenProb *tb = (const TokenProb *)b;
    
    // Sort in descending order of probability
    if (ta->probability > tb->probability) return -1;
    if (ta->probability < tb->probability) return 1;
    return 0;
}

/**
 * Nucleus (top-p) sampling implementation.
 * This selects the next token by sampling from the smallest set of tokens
 * whose cumulative probability exceeds the threshold 'top_p'.
 * 
 * Algorithm:
 * 1. Convert logits to probabilities using softmax
 * 2. Sort tokens by probability in descending order
 * 3. Find the nucleus (smallest set of tokens with cumulative probability >= top_p)
 * 4. Sample from the nucleus using the renormalized probabilities
 */
int sample_nucleus(Matrix* logits, float top_p, int vocab_size) {
    if (!logits || vocab_size <= 0) return 0;
    
    // Clamp top_p to valid range
    if (top_p <= 0.0f) top_p = 0.01f;
    if (top_p > 1.0f) top_p = 1.0f;
    
    // Step 1: Convert logits to probabilities using softmax
    float max_logit = logits->data[0];
    for (int i = 1; i < vocab_size; ++i) {
        if (logits->data[i] > max_logit) max_logit = logits->data[i];
    }
    
    float sum_exp = 0.0f;
    for (int i = 0; i < vocab_size; ++i) {
        logits->data[i] = expf(logits->data[i] - max_logit);
        sum_exp += logits->data[i];
    }
    
    // Normalize to probabilities
    for (int i = 0; i < vocab_size; ++i) {
        logits->data[i] /= sum_exp;
    }
    
    // Step 2: Create array of token-probability pairs
    TokenProb *token_probs = (TokenProb *)safe_malloc(vocab_size * sizeof(TokenProb));
    for (int i = 0; i < vocab_size; ++i) {
        token_probs[i].token_id = i;
        token_probs[i].probability = logits->data[i];
    }
    
    // Step 3: Sort tokens by probability in descending order
    qsort(token_probs, vocab_size, sizeof(TokenProb), compare_token_probs);
    
    // Step 4: Find the nucleus (cumulative probability >= top_p)
    float cumulative_prob = 0.0f;
    int nucleus_size = 0;
    
    for (int i = 0; i < vocab_size; ++i) {
        cumulative_prob += token_probs[i].probability;
        nucleus_size++;
        
        // Include at least one token, then check if we've reached the threshold
        if (cumulative_prob >= top_p && nucleus_size > 0) {
            break;
        }
    }
    
    // Ensure we have at least one token in the nucleus
    if (nucleus_size == 0) nucleus_size = 1;
    
    // Step 5: Renormalize probabilities within the nucleus
    float nucleus_prob_sum = 0.0f;
    for (int i = 0; i < nucleus_size; ++i) {
        nucleus_prob_sum += token_probs[i].probability;
    }
    
    // Step 6: Sample from the nucleus using renormalized probabilities
    float random_val = ((float)rand() / RAND_MAX) * nucleus_prob_sum;
    float running_sum = 0.0f;
    int selected_token = token_probs[0].token_id; // Default fallback
    
    for (int i = 0; i < nucleus_size; ++i) {
        running_sum += token_probs[i].probability;
        if (running_sum >= random_val) {
            selected_token = token_probs[i].token_id;
            break;
        }
    }
    
    // Freeing allocated memory 😵
    free(token_probs);
    
    return selected_token;
}

/**
 * Decode a single token ID to its string representation.
 * This is a wrapper around the tokenizer's word lookup function.
 * 
 * @param tokenizer: The tokenizer instance
 * @param token_id: The token ID to decode
 * @return: Allocated string containing the token text (caller must free)
 */
char* tokenizer_decode_single(const Tokenizer* tokenizer, int token_id) {
    if (!tokenizer) return NULL;
    
    const char* word = tokenizer_get_word(tokenizer, token_id);
    if (!word) return NULL;
    
    // Return a copy that the caller can free
    return strdup(word);
}

/**
 * ========================================================
 * TRAINING DATA MANAGEMENT SECTION
 * =================================================
 * This section handles loading data, tokenizing it, and organizing it
 * into batches suitable for training.
 */

/**
 * Structure to hold batched training data.
 * In language modeling, we predict the next token given previous tokens.
 * So input[i] = [t1, t2, ..., tk] and target[i] = [t2, t3, ..., t(k+1)]
 */
typedef struct
{
    int **batched_input_ids;  // [num_batches][batch_size * seq_len] - input token sequences
    int **batched_target_ids; // [num_batches][batch_size * seq_len] - target token sequences (shifted by 1)
    int num_batches;          // Total number of batches created
    int num_tokens_total;     // Total number of tokens across all batches
} TrainingData;

/**
 * Utility function to shuffle an array of integers for batch shuffling.
 * Uses Fisher-Yates shuffle algorithm for randomness.
 */
void shuffle_int_array(int *array, int size)
{
    for (int i = size - 1; i > 0; i--)
    {
        int j = rand() % (i + 1);
        int temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}

/**
 * Loads text data from file, tokenizes it, and creates training batches.
 *
 * @param file_path: Path to the text file containing training data
 * @param tokenizer: Tokenizer to convert text to token IDs
 * @param batch_size: Number of sequences per batch
 * @param seq_len: Length of each sequence
 * @return: Pointer to TrainingData structure, or NULL on failure
 */
TrainingData *load_and_batch_training_data(const char *file_path, Tokenizer *tokenizer,
                                           int batch_size, int seq_len)
{
    printf("Loading training data from: %s\n", file_path);

    // Step 1: Read the text file into memory
    FILE *fp = fopen(file_path, "r");
    if (!fp)
    {
        perror("load_and_batch_training_data fopen");
        return NULL;
    }

    // Get file size to allocate appropriate buffer
    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    // Allocate buffer and read entire file
    char *full_text = (char *)safe_malloc(file_size + 1);
    size_t bytes_read = fread(full_text, 1, file_size, fp);
    if (bytes_read != (size_t)file_size) {
        fprintf(stderr, "Warning: Expected to read %ld bytes, but read " SIZE_T_FMT " bytes\n", file_size, bytes_read);
    }
    full_text[file_size] = '\0'; // Null terminate
    fclose(fp);

    // Step 2: Tokenize the entire text into a sequence of token IDs
    // Estimate maximum possible tokens (rough overestimate 😶)
    int *all_token_ids = (int *)safe_malloc(file_size * sizeof(int));
    int total_tokens_from_file = tokenizer_encode(tokenizer, full_text, all_token_ids, file_size, 0, 0);
    free(full_text); // No longer needed (Like Me🙂)

    // Validate we have enough tokens for at least one sequence
    if (total_tokens_from_file < seq_len + 1)
    {
        fprintf(stderr, "Not enough tokens in data file (%d) for one sequence of length %d.\n",
                total_tokens_from_file, seq_len);
        free(all_token_ids);
        return NULL;
    }

    // Step 3: Calculate how many sequences and batches we can create
    // For language modeling: each sequence needs seq_len input tokens + 1 target token
    int num_sequences = (total_tokens_from_file - 1) / seq_len;
    int num_batches = num_sequences / batch_size;

    if (num_batches == 0)
    {
        fprintf(stderr, "Not enough sequences for one batch. Num sequences: %d, batch_size: %d\n",
                num_sequences, batch_size);
        free(all_token_ids);
        return NULL;
    }

    // Step 4: Allocate TrainingData structure
    TrainingData *data = (TrainingData *)safe_malloc(sizeof(TrainingData));
    data->num_batches = num_batches;
    data->batched_input_ids = (int **)safe_malloc(num_batches * sizeof(int *));
    data->batched_target_ids = (int **)safe_malloc(num_batches * sizeof(int *));
    data->num_tokens_total = num_batches * batch_size * seq_len;

    // Step 5: Create input/target pairs and organize into batches
    // For each sequence: input = [t_i, t_{i+1}, ..., t_{i+seq_len-1}]
    //                   target = [t_{i+1}, t_{i+2}, ..., t_{i+seq_len}] 😵
    int current_token_idx = 0;
    for (int b = 0; b < num_batches; ++b)
    {
        // Allocate memory for this batch
        data->batched_input_ids[b] = (int *)safe_malloc(batch_size * seq_len * sizeof(int));
        data->batched_target_ids[b] = (int *)safe_malloc(batch_size * seq_len * sizeof(int));

        // Fill this batch with sequences
        for (int i = 0; i < batch_size; ++i)
        {
            // Safety check to prevent buffer overflow 🤓
            if (current_token_idx + seq_len >= total_tokens_from_file)
            {
                fprintf(stderr, "Ran out of tokens unexpectedly during batching.\n");
                // Clean up partially allocated data 😵
                for (int cleanup_b = 0; cleanup_b <= b; ++cleanup_b)
                {
                    if (cleanup_b < b || i > 0)
                    {
                        free(data->batched_input_ids[cleanup_b]);
                        free(data->batched_target_ids[cleanup_b]);
                    }
                }
                free(data->batched_input_ids);
                free(data->batched_target_ids);
                free(data);
                free(all_token_ids);
                return NULL;
            }

            // Copy sequence tokens to batch arrays
            for (int t = 0; t < seq_len; ++t)
            {
                data->batched_input_ids[b][i * seq_len + t] = all_token_ids[current_token_idx + t];
                data->batched_target_ids[b][i * seq_len + t] = all_token_ids[current_token_idx + t + 1];
            }
            current_token_idx += seq_len; // Move to next sequence 
        }
    }

    free(all_token_ids);
    printf("Successfully loaded training data: %d batches, %d tokens total.\n",
           data->num_batches, data->num_tokens_total);
    return data;
}

/**
 * Shuffle the order of batches for better training dynamics.
 * This helps prevent the model from memorizing the order of data.
 */
void shuffle_training_batches(TrainingData *data)
{
    if (!data || data->num_batches <= 1)
        return;

    // Create array of batch indices 🤔
    int *batch_indices = (int *)safe_malloc(data->num_batches * sizeof(int));
    for (int i = 0; i < data->num_batches; ++i)
    {
        batch_indices[i] = i;
    }

    // Shuffle the indices using Fisher-Yates algorithm
    shuffle_int_array(batch_indices, data->num_batches);

    // Create temporary arrays to hold shuffled batches
    int **temp_input_ids = (int **)safe_malloc(data->num_batches * sizeof(int *));
    int **temp_target_ids = (int **)safe_malloc(data->num_batches * sizeof(int *));

    // Copy batches in shuffled order
    for (int i = 0; i < data->num_batches; ++i)
    {
        temp_input_ids[i] = data->batched_input_ids[batch_indices[i]];
        temp_target_ids[i] = data->batched_target_ids[batch_indices[i]];
    }

    // Replace original arrays with shuffled versions
    free(data->batched_input_ids);
    free(data->batched_target_ids);
    data->batched_input_ids = temp_input_ids;
    data->batched_target_ids = temp_target_ids;

    free(batch_indices);
}

/**
 * Freeing all memory allocated for training data.
 * Must be called when done with TrainingData to prevent memory leaks.
 */
void free_training_data(TrainingData *data)
{
    if (!data)
        return;

    // Free each batch
    for (int b = 0; b < data->num_batches; ++b)
    {
        free(data->batched_input_ids[b]);
        free(data->batched_target_ids[b]);
    }

    // Free batch arrays
    free(data->batched_input_ids);
    free(data->batched_target_ids);

    // Free main structure
    free(data);
}

/**
 * ========================================================
 * TEXT GENERATION SECTION
 * ====================================
 * This section implements text generation using the trained transformer model.
 */

/**
 * Generates text using the transformer model with nucleus (top-p) sampling.
 * More sophisticated than simple greedy or top-k sampling.
 *
 * @param weights: Trained model weights
 * @param tokenizer: Tokenizer for encoding/decoding text
 * @param prompt: Initial text to start generation from
 * @param max_new_tokens: Maximum number of new tokens to generate
 * @param top_p: Nucleus sampling parameter (0.0 to 1.0)
 */
void generate_text_words(TransformerWeights *weights, Tokenizer *tokenizer,
                         const char *prompt, int max_new_tokens, float top_p)
{
    printf("Starting text generation...\n");
    printf("Prompt: \"%s\"\n", prompt);
    printf("Max new tokens: %d, Top-p: %.2f\n", max_new_tokens, top_p);

    // Step 1: Tokenize the input prompt
    int max_prompt_tokens = strlen(prompt) + 10; // Conservative estimate 
    int *prompt_tokens = (int *)safe_malloc(max_prompt_tokens * sizeof(int));
    int prompt_length = tokenizer_encode(tokenizer, prompt, prompt_tokens, max_prompt_tokens, 0, 0);

    if (prompt_length == 0)
    {
        fprintf(stderr, "Warning: Empty prompt after tokenization.\n");
        free(prompt_tokens);
        return;
    }

    printf("Prompt tokenized to %d tokens.\n", prompt_length);

    // Step 2: Initialize transformer state for generation
    TransformerState *state = create_transformer_state(weights->config);
    if (!state) {
        fprintf(stderr, "Error: Failed to create transformer state.\n");
        free(prompt_tokens);
        return;
    }

    // Step 3: Prepare context buffer (prompt + generated tokens, for simplicity)
    int max_context_len = prompt_length + max_new_tokens;
    int *context_tokens = (int *)safe_malloc(max_context_len * sizeof(int));

    // Copy prompt tokens to context
    memcpy(context_tokens, prompt_tokens, prompt_length * sizeof(int));
    int current_length = prompt_length;

    // Process prompt tokens through the model to warm up the KV cache 🦾
    printf("Processing prompt through model...\n");
    for (int i = 0; i < prompt_length; i++) {
        Matrix *logits = transformer_forward_pass_single_token(weights, state, context_tokens[i]);
        if (logits) {
            free_matrix(logits); // We don't need prompt logits, just warming cache 😪
        }
    }

    // Step 4: Generate tokens one by one
    printf("\nGenerated text:\n%s", prompt); // Print the prompt first 
    
    // Keep track of generated tokens for proper spacing
    int *generated_tokens = (int *)safe_malloc(max_new_tokens * sizeof(int));
    int generated_count = 0;

    for (int i = 0; i < max_new_tokens; ++i)
    {
        // Generate next token using the last token in context
        int last_token = context_tokens[current_length - 1];
        
        // Forward pass through the transformer for single token
        Matrix *logits = transformer_forward_pass_single_token(weights, state, last_token);

        if (!logits)
        {
            fprintf(stderr, "Error: Forward pass failed during generation.\n");
            break;
        }

        // Step 4: Sample next token using nucleus sampling
        int next_token = sample_nucleus(logits, top_p, tokenizer->vocab_size);

        // Add sampled token to context
        context_tokens[current_length] = next_token;
        current_length++;
        
        // Stop if we hit end-of-sequence token (if defined)
        if (next_token == tokenizer->eos_id)
        {
            printf("\n[End of sequence reached]\n");
            break;
        }
        
        // Skip special tokens for output, but still process them internally
        if (next_token != tokenizer->bos_id && next_token != tokenizer->pad_id && next_token != tokenizer->eos_id) {
            // Add to generated tokens for proper decoding
            generated_tokens[generated_count++] = next_token;
            
            // Just print the new token with proper spacing
            char *token_text = tokenizer_decode_single(tokenizer, next_token);
            if (token_text) {
                // Add space before token if not the first generated token
                if (generated_count > 1) {
                    printf(" ");
                }
                printf("%s", token_text);
                fflush(stdout);
                free(token_text);
            }
        }

        // Freeing memory for this iteration
        free_matrix(logits);

        // Prevent infinite context growth (sliding window approach)
        if (current_length >= weights->config->max_seq_len)
        {
            // Reset transformer state and restart generation
            reset_transformer_state(state); // Could be improved
            printf("\n[Context limit reached, resetting state]\n");
            
            // Keep only the last portion of context
            int keep_tokens = weights->config->max_seq_len / 2;
            memmove(context_tokens, context_tokens + (current_length - keep_tokens),
                    keep_tokens * sizeof(int));
            current_length = keep_tokens;
            
            // Re-process the kept context to rebuild cache
            for (int j = 0; j < current_length - 1; j++) {
                Matrix *rebuild_logits = transformer_forward_pass_single_token(weights, state, context_tokens[j]);
                if (rebuild_logits) {
                    free_matrix(rebuild_logits);
                }
            }
        }
    }

    printf("\n\nGeneration completed.\n");

    // Cleanup 🐒
    free_transformer_state(state, weights->config);
    free(prompt_tokens);
    free(context_tokens);
    free(generated_tokens);
}

/**
 * Validates basic transformer configuration constraints.
 */
static int validate_config(TransformerConfig* config) {
    if (!config) return 0;
    
    // Validate basic constraints
    if (config->d_model % config->n_heads != 0) {
        fprintf(stderr, "Error: d_model (%d) must be divisible by n_heads (%d)\n", 
               config->d_model, config->n_heads);
        config->n_heads = config->d_model / 16;
        if (config->n_heads < 1) config->n_heads = 1;
    }
    
    if (config->vocab_size <= 0) {
        fprintf(stderr, "Error: vocab_size must be positive, got %d\n", config->vocab_size);
        config->vocab_size = 1000;
    }
    
    if (config->max_seq_len <= 0) {
        fprintf(stderr, "Error: max_seq_len must be positive, got %d\n", config->max_seq_len);
        config->max_seq_len = 128;
    }
    
    return 1;
}






/**
 * ============================================================================
 * MAIN FUNCTION
 * ============================================================================
 * Entry point of the program. Handles argument parsing, model initialization,
 * and dispatches to training or generation modes.
 */
int main(int argc, char *argv[])
{
    printf("C Transformer - Starting up...\n");

    // Step 1: Initialize argument structure with default values
    Arguments arguments;

    // Set default model hyperparameters
    arguments.model_cfg.learning_rate = 3e-4f;                  // Learning rate
    arguments.model_cfg.num_epochs = 1;                         // Single epoch for quick testing
    arguments.model_cfg.batch_size = 8;                         // Default batch size
    arguments.model_cfg.vocab_size = 10000;                     // Vocabulary size
    arguments.model_cfg.max_seq_len = 256;                      // Context window length
    arguments.model_cfg.d_model = 128;                          // Model dimension
    arguments.model_cfg.n_layers = 8;                           // Number of layers
    arguments.model_cfg.n_heads = 8;                            // Number of attention heads (d_model/n_heads=16)
    arguments.model_cfg.d_ff = arguments.model_cfg.d_model * 3; // Feed-forward dimension
    arguments.model_cfg.layer_norm_eps = 1e-5f;                 // Layer normalization epsilon
    arguments.model_cfg.grad_clip_norm = 1.0f;                  // Gradient clipping threshold
    arguments.model_cfg.save_every_n_epochs = 1;                // Save weights after each epoch
    arguments.model_cfg.print_every_n_batches = 10;             // Print progress every 10 batches

    // Set default paths and values (enhanced for consistent file management 😪)
    arguments.mode = NULL;
    arguments.data_path = NULL;
    arguments.vocab_load_path = "aryan_vocab.vocab"; // Default vocabulary file
    arguments.vocab_save_path = "aryan_vocab.vocab"; // Always save to same location
    arguments.weights_load_path = NULL;
    arguments.weights_save_path = "aryan_model.bin"; // Default save path
    arguments.prompt = "Hello";                      // Default generation prompt
    arguments.max_new_tokens = 100;                  // Default generation length
    arguments.grad_accumulation_steps = 1;          // No gradient accumulation by default

    // Step 2: Parse command-line arguments or launch interactive CLI
    if (argc == 1) {
        // No arguments provided - launch interactive CLI
        printf("No arguments provided. Launching interactive CLI...\n");
        run_interactive_cli();
        return 0;
    }

#ifndef ARGP_NOT_AVAILABLE
    argp_parse(&argp_parser, argc, argv, 0, 0, &arguments);
#else
    int parse_result = parse_args_simple(argc, argv, &arguments);
    if (parse_result != 0) {
        return (parse_result > 0) ? 0 : 1; // Help shown (0) or error (1)
    }
#endif
    TransformerConfig *cfg = &arguments.model_cfg; // Convenience pointer

    // Initialize random seed for reproducible randomness
    srand(time(NULL));

    // Step 3: Initialize tokenizer (load existing or create new)
    printf("Initializing tokenizer...\n");
    Tokenizer *tokenizer = NULL;

    // Try to load existing vocabulary first
    if (arguments.vocab_load_path && (tokenizer = tokenizer_load(arguments.vocab_load_path)) != NULL)
    {
        printf("Loaded tokenizer from %s (vocab_size: %d)\n",
               arguments.vocab_load_path, tokenizer->vocab_size);
    }
    else
    {
        // Create new tokenizer if loading failed or no path provided
        tokenizer = create_tokenizer(DEFAULT_VOCAB_CAPACITY);

        if (arguments.data_path)
        {
            // Build vocabulary from training data if available
            printf("Building tokenizer vocabulary from corpus: %s\n", arguments.data_path);
            tokenizer_build_from_corpus(tokenizer, arguments.data_path, 1 /*min_freq*/);

            // Save the built vocabulary if path provided
            if (arguments.vocab_save_path)
            {
                tokenizer_save(tokenizer, arguments.vocab_save_path);
                printf("Saved tokenizer vocabulary to %s\n", arguments.vocab_save_path);
            }
        }
        else if (arguments.mode && strcmp(arguments.mode, "train") == 0)
        {
            // Training mode requires either data to build vocab or pre-built vocab
            fprintf(stderr, "Error: Training mode requires --data or a pre-built --vocab_load.\n");
            free_tokenizer(tokenizer);
            return 1;
        }
        // For generation without training data, use default tokens (handled in create_tokenizer)
    }

    // Update configuration with actual vocabulary size
    cfg->vocab_size = tokenizer->vocab_size;

    // Step 4: Validate configuration
    if (!validate_config(cfg)) {
        fprintf(stderr, "Error: Configuration validation failed.\n");
        free_tokenizer(tokenizer);
        return 1;
    }

    // Step 5: Initialize transformer model weights
    printf("Initializing transformer model...\n");
    printf("Model configuration: d_model=%d, n_layers=%d, n_heads=%d, vocab_size=%d\n",
           cfg->d_model, cfg->n_layers, cfg->n_heads, cfg->vocab_size);

    TransformerWeights *weights = create_transformer_weights(cfg);
    if (weights == NULL)
    {
        fprintf(stderr, "Error: Failed to create/allocate model weights.\n");
        free_tokenizer(tokenizer);
        return 1;
    }

    // Try to load pre-trained weights if path provided
    if (arguments.weights_load_path && load_transformer_weights(weights, arguments.weights_load_path))
    {
        printf("Loaded model weights from %s\n", arguments.weights_load_path);
    }
    else
    {
        printf("Initializing model with random weights.\n");
        init_dummy_weights(weights); // Initialize with random values 🎇
    }

    // Step 6: Execute training or generation based on mode
    if (strcmp(arguments.mode, "train") == 0)
    {
        /**
         * TRAINING MODE
         * ==============
         * 1. Load and prepare training data
         * 2. Initialize optimizer and gradients
         * 3. Run training loop with forward/backward passes
         * 4. Save model weights periodically
         */

        if (!arguments.data_path)
        {
            fprintf(stderr, "Error: Training mode requires --data path.\n");
            free_transformer_weights(weights);
            free_tokenizer(tokenizer);
            return 1;
        }

        printf("\n=== STARTING TRAINING ===\n");
        printf("Configuration:\n");
        printf("  Learning Rate: %.2e (with cosine decay + warmup)\n", cfg->learning_rate);
        printf("  Epochs: %d\n", cfg->num_epochs);
        printf("  Batch Size: %d\n", cfg->batch_size);

        // Initialize training components
        TransformerGradients *grads = create_transformer_gradients(cfg);
        
        // Load training data first to calculate total steps
        TrainingData *train_data = load_and_batch_training_data(arguments.data_path, tokenizer,
                                                                cfg->batch_size, cfg->max_seq_len);
        if (!train_data)
        {
            fprintf(stderr, "Failed to load training data.\n");
            free_transformer_gradients(grads);
            free_transformer_weights(weights);
            free_tokenizer(tokenizer);
            return 1;
        }
        
        // Calculate total training steps for learning rate scheduling
        int total_steps = cfg->num_epochs * train_data->num_batches;
        int warmup_steps = total_steps / 10; // 10% warmup 🤺

        printf("  Sequence Length: %d\n", cfg->max_seq_len);
        printf("  Model Dimension: %d\n", cfg->d_model);
        printf("  Layers: %d\n", cfg->n_layers);
        printf("  Attention Heads: %d\n", cfg->n_heads);
        printf("  Gradient Accumulation Steps: %d\n", arguments.grad_accumulation_steps);
        printf("  Total Training Steps: %d\n", total_steps);
        printf("  Warmup Steps: %d\n", warmup_steps);
        printf("  Enhanced Adam Optimizer: ENABLED\n");
        printf("  Gradient Clipping: %.1f\n", cfg->grad_clip_norm);
        
        // Create optimizer with learning rate scheduling
        AdamOptimizerState *optimizer_state = create_enhanced_adam_optimizer_state(cfg, 
                                                                                   cfg->learning_rate,
                                                                                   warmup_steps,
                                                                                   total_steps);
        if (!optimizer_state)
        {
            fprintf(stderr, "Failed to create optimizer state.\n");
            free_transformer_gradients(grads);
            free_training_data(train_data);
            free_transformer_weights(weights);
            free_tokenizer(tokenizer);
            return 1;
        }

        // Create activation cache and causal mask (created once for efficiency 😎)
        TrainingActivationsCache *act_cache = create_training_activations_cache(cfg, cfg->batch_size, cfg->max_seq_len);
        Matrix *causal_mask = create_causal_mask(cfg->max_seq_len);

        // Main training loop over epochs
        for (int epoch = 0; epoch < cfg->num_epochs; ++epoch)
        {
            printf("\n--- Epoch %d/%d ---\n", epoch + 1, cfg->num_epochs);

            float total_epoch_loss = 0.0f;
            long tokens_processed_epoch = 0;
            time_t epoch_start_time = time(NULL);

            // Shuffle batches for better training dynamics
            shuffle_training_batches(train_data);

            // Initialize gradient accumulation
            zero_gradients(grads);
            int accumulated_steps = 0;

            // Loop over all batches in this epoch
            for (int b = 0; b < train_data->num_batches; ++b)
            {
                // Prepare batch data (convert int arrays to float matrices)
                int num_tokens_this_batch = cfg->batch_size * cfg->max_seq_len;
                Matrix *input_batch_mat = create_matrix(num_tokens_this_batch, 1);
                Matrix *target_batch_mat = create_matrix(num_tokens_this_batch, 1);

                // Convert integer token IDs to float matrices
                for (int k = 0; k < num_tokens_this_batch; ++k)
                {
                    input_batch_mat->data[k] = (float)train_data->batched_input_ids[b][k];
                    target_batch_mat->data[k] = (float)train_data->batched_target_ids[b][k];
                }

                // Forward and backward pass for this batch
                float batch_loss = transformer_train_batch(weights, grads, act_cache,
                                                           input_batch_mat, target_batch_mat,
                                                           causal_mask, cfg, (b == 0));

                accumulated_steps++;

                // Update weights when we've accumulated enough gradients 
                if (accumulated_steps >= arguments.grad_accumulation_steps)
                {
                    // Apply optimizer update with learning rate scheduling
                    adam_update_with_schedule(weights, grads, optimizer_state, cfg->grad_clip_norm);

                    // Reset gradients for next accumulation cycle
                    zero_gradients(grads);
                    accumulated_steps = 0;
                }

                // Update statistics
                total_epoch_loss += batch_loss * num_tokens_this_batch; // batch_loss is per-token
                tokens_processed_epoch += num_tokens_this_batch;

                // Print progress periodically
                if ((b + 1) % cfg->print_every_n_batches == 0 || b == train_data->num_batches - 1)
                {
                    float tokens_per_sec = (float)tokens_processed_epoch / (difftime(time(NULL), epoch_start_time) + 1e-6);
                    float current_lr = warmup_cosine_lr(optimizer_state->initial_lr, 
                                                       optimizer_state->t, 
                                                       optimizer_state->warmup_steps, 
                                                       optimizer_state->total_steps);
                    printf("  Batch %d/%d | Loss: %.4f | LR: %.2e | Tokens/sec: %.0f\n",
                           b + 1, train_data->num_batches, batch_loss, current_lr, tokens_per_sec);
                }

                // Free up batch matrices
                free_matrix(input_batch_mat);
                free_matrix(target_batch_mat);
            } // End batch loop

            // Handle any remaining accumulated gradients
            if (accumulated_steps > 0)
            {
                adam_update_with_schedule(weights, grads, optimizer_state, cfg->grad_clip_norm);
            }

            // Calculate and print epoch statistics
            float avg_epoch_loss = (tokens_processed_epoch > 0) ? total_epoch_loss / tokens_processed_epoch : 0.0f;
            float epoch_time = difftime(time(NULL), epoch_start_time);
            printf("Epoch %d completed | Average Loss: %.4f | Time: %.2f sec | Tokens/sec: %.0f\n",
                   epoch + 1, avg_epoch_loss, epoch_time, (float)tokens_processed_epoch / epoch_time);

            // Save model weights periodically
            if ((epoch + 1) % cfg->save_every_n_epochs == 0 && arguments.weights_save_path)
            {
                printf("Saving model weights to %s...\n", arguments.weights_save_path);
                save_transformer_weights(weights, arguments.weights_save_path);
            }
        } // End epoch loop ❌

        printf("\n=== TRAINING COMPLETED ===\n");

        // Free training resources
        free_transformer_gradients(grads);
        free_adam_optimizer_state(optimizer_state);
        free_training_data(train_data);
        free_training_activations_cache(act_cache, cfg);
        free_matrix(causal_mask);
    }
    else if (strcmp(arguments.mode, "generate") == 0)
    {
        /**
         * GENERATION MODE
         * ===============
         * Use the trained model to generate text from a given prompt.
         */

        printf("\n=== STARTING GENERATION ===\n");

        // Perform text generation 💥
        generate_text_words(weights, tokenizer, arguments.prompt, arguments.max_new_tokens, 0.9f /*top_p*/);
    }
    else
    {
        // Invalid mode specified
        fprintf(stderr, "Invalid mode: %s. Use 'train' or 'generate'.\n", arguments.mode);
#ifndef ARGP_NOT_AVAILABLE
        argp_help(&argp_parser, stderr, ARGP_HELP_STD_HELP, argv[0]);
#else
        print_help(argv[0]);
#endif
        free_transformer_weights(weights);
        free_tokenizer(tokenizer);
        return 1;
    }

    // Step 7: Cleanup and exit 💥💥💥
    printf("\nCleaning up resources...\n");
    free_transformer_weights(weights);
    free_tokenizer(tokenizer);

    printf("Program execution completed successfully.\n");
    return 0;
}


// Completing this project took my soul.