#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stddef.h>
#include "platform.h"  // For cross-platform types

#define DEFAULT_VOCAB_CAPACITY 4096 // For words
#define MAX_WORD_LEN 128          // Max length of a word/token string
#define HASH_TABLE_SIZE 65536
#define HASH_LOAD_FACTOR 0.75

typedef struct Tokenizer {
    char** vocab_strs;      // Array of token strings
    int vocab_size;
    int vocab_capacity;

    // Special token IDs (fixed for simplicity after creation)
    int unk_id;
    int bos_id; // Beginning of sequence
    int eos_id; // End of sequence
    int pad_id; // Padding token

    // New fields for enhanced implementation
    void* word_to_id;  // Opaque pointer to HashTable
    platform_rwlock_t vocab_lock;
    platform_atomic_int_t ref_count;
} Tokenizer;

/**
 * Create a new tokenizer with specified initial capacity
 */
Tokenizer* create_tokenizer(int initial_capacity);

/**
 * Free a tokenizer instance
 */
void free_tokenizer(Tokenizer* tokenizer);

/**
 * Increment reference count (for thread-safe sharing)
 */
void tokenizer_addref(Tokenizer* tokenizer);

/**
 * Add a token string to vocab, returns its ID. Handles resizing.
 */
int tokenizer_add_word(Tokenizer* tokenizer, const char* word);

/**
 * Get token ID for given word string
 */
int tokenizer_get_id(const Tokenizer* tokenizer, const char* word);

/**
 * Get word string for given token ID
 */
const char* tokenizer_get_word(const Tokenizer* tokenizer, int id);

/**
 * Build vocabulary from a text file. min_freq to filter.
 * Normalizes words to lowercase and handles basic punctuation.
 */
int tokenizer_build_from_corpus(Tokenizer* tokenizer, const char* corpus_path, int min_freq);

/**
 * Save tokenizer to file
 */
int tokenizer_save(const Tokenizer* tokenizer, const char* path);

/**
 * Load tokenizer from file
 */
Tokenizer* tokenizer_load(const char* path);

/**
 * Tokenize a line of text into an array of token IDs
 * `token_ids_out` should be pre-allocated. `max_output_tokens` is its capacity.
 * Returns the number of tokens produced.
 */
int tokenizer_encode(const Tokenizer* tokenizer, const char* text,
                     int* token_ids_out, int max_output_tokens,
                     int add_bos, int add_eos);

/**
 * Decode a sequence of token IDs back into a string.
 * Caller must free the returned string.
 */
char* tokenizer_decode(const Tokenizer* tokenizer, const int* token_ids, int num_tokens);

#endif // TOKENIZER_H