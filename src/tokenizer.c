#include "tokenizer.h"
#include "utils.h"
#include "platform.h"  // For cross-platform threading and file operations
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>

// Hash table implementation for fast word->ID lookup

#define HASH_TABLE_SIZE 65536
#define HASH_LOAD_FACTOR 0.75

typedef struct HashNode {
    char* word;
    int id;
    struct HashNode* next;
} HashNode;

typedef struct {
    HashNode** buckets;
    size_t size;
    size_t capacity;
    platform_rwlock_t lock;
} HashTable;

// Word frequency structure for corpus building

typedef struct WordFreq {
    char* word;
    int count;
    struct WordFreq* next;
} WordFreq;

typedef struct {
    WordFreq** buckets;
    size_t capacity;
    platform_mutex_t lock;
} FreqTable;

// Hash function (djb2 algorithm?🤔)
static unsigned int hash_string(const char* str) {
    unsigned int hash = 5381;
    int c;
    while ((c = *str++)) {
        hash = ((hash << 5) + hash) + c;
    }
    return hash;
}

// Hash table operations

static HashTable* create_hash_table(size_t initial_capacity) {
    HashTable* ht = (HashTable*)safe_malloc(sizeof(HashTable));
    ht->capacity = initial_capacity > 0 ? initial_capacity : HASH_TABLE_SIZE;
    ht->buckets = (HashNode**)calloc(ht->capacity, sizeof(HashNode*));
    if (!ht->buckets) {
        free(ht);
        return NULL;
    }
    ht->size = 0;
    platform_rwlock_init(&ht->lock);
    return ht;
}

static void free_hash_table(HashTable* ht) {
    if (!ht) return;
    
    for (size_t i = 0; i < ht->capacity; i++) {
        HashNode* node = ht->buckets[i];
        while (node) {
            HashNode* next = node->next;
            free(node->word);
            free(node);
            node = next;
        }
    }
    platform_rwlock_destroy(&ht->lock);
    free(ht->buckets);
    free(ht);
}

static int hash_table_insert(HashTable* ht, const char* word, int id) {
    if (!ht || !word) return 0;
    
    platform_rwlock_wrlock(&ht->lock);
    
    unsigned int index = hash_string(word) % ht->capacity;
    HashNode* node = ht->buckets[index];
    
    // Checks if word already exists

    while (node) {
        if (strcmp(node->word, word) == 0) {
            platform_rwlock_unlock_wr(&ht->lock);
            return node->id; // Return existing ID
        }
        node = node->next;
    }
    
    // Creates new node
    HashNode* new_node = (HashNode*)safe_malloc(sizeof(HashNode));
    new_node->word = strdup(word);
    if (!new_node->word) {
        free(new_node);
        platform_rwlock_unlock_wr(&ht->lock);
        return -1;
    }
    new_node->id = id;
    new_node->next = ht->buckets[index];
    ht->buckets[index] = new_node;
    ht->size++;
    
    platform_rwlock_unlock_wr(&ht->lock);
    return id;
}

static int hash_table_lookup(HashTable* ht, const char* word) {
        if (!ht || !word) return -1;
    
    platform_rwlock_rdlock(&ht->lock);
    
    unsigned int index = hash_string(word) % ht->capacity;
    HashNode* node = ht->buckets[index];
    
    while (node) {
        if (strcmp(node->word, word) == 0) {
            int id = node->id;
            platform_rwlock_unlock_rd(&ht->lock);
            return id;
        }
        node = node->next;
    }
    
    platform_rwlock_unlock_rd(&ht->lock);
    return -1;
}

// Frequency table for corpus building

static FreqTable* create_freq_table(size_t capacity) {
    FreqTable* ft = (FreqTable*)safe_malloc(sizeof(FreqTable));
    ft->capacity = capacity > 0 ? capacity : HASH_TABLE_SIZE;
    ft->buckets = (WordFreq**)calloc(ft->capacity, sizeof(WordFreq*));
    if (!ft->buckets) {
        free(ft);
        return NULL;
    }
    platform_mutex_init(&ft->lock);
    return ft;
}

static void free_freq_table(FreqTable* ft) {
    if (!ft) return;
    
    for (size_t i = 0; i < ft->capacity; i++) {
        WordFreq* freq = ft->buckets[i];
        while (freq) {
            WordFreq* next = freq->next;
            free(freq->word);
            free(freq);
            freq = next;
        }
    }
    platform_mutex_destroy(&ft->lock);
    free(ft->buckets);
    free(ft);
}

static void freq_table_increment(FreqTable* ft, const char* word) {
    if (!ft || !word) return;
    
    platform_mutex_lock(&ft->lock);
    
    unsigned int index = hash_string(word) % ft->capacity;
    WordFreq* freq = ft->buckets[index];
    
    // Look for existing word
    while (freq) {
        if (strcmp(freq->word, word) == 0) {
            freq->count++;
            platform_mutex_unlock(&ft->lock);
            return;
        }
        freq = freq->next;
    }
    
    // Creates new frequency entry

    WordFreq* new_freq = (WordFreq*)safe_malloc(sizeof(WordFreq));
    new_freq->word = strdup(word);
    if (!new_freq->word) {
        free(new_freq);
        platform_mutex_unlock(&ft->lock);
        return;
    }
    new_freq->count = 1;
    new_freq->next = ft->buckets[index];
    ft->buckets[index] = new_freq;
    
    platform_mutex_unlock(&ft->lock);
}

// Text processing utilities

static int is_valid_word_char(unsigned char c) {
    return isalnum(c) || c == '\'' || c == '-';
}

static void normalize_word(char* word) {
    char* src = word;
    char* dst = word;
    
    while (*src) {
        if (is_valid_word_char(*src)) {
            *dst++ = tolower(*src);
        }
        src++;
    }
    *dst = '\0';
}

static int extract_words_from_line(const char* line, FreqTable* freq_table) {
    if (!line || !freq_table) return 0;
    
    char word_buffer[MAX_WORD_LEN];
    const char* ptr = line;
    int word_count = 0;
    
    while (*ptr) {
        // Skips non-word characters

        while (*ptr && !is_valid_word_char(*ptr)) ptr++;
        if (!*ptr) break;
        
        // Extracts word

        int len = 0;
        while (*ptr && is_valid_word_char(*ptr) && len < MAX_WORD_LEN - 1) {
            word_buffer[len++] = *ptr++;
        }
        word_buffer[len] = '\0';
        
        if (len > 0) {
            normalize_word(word_buffer);
            if (strlen(word_buffer) > 0) {
                freq_table_increment(freq_table, word_buffer);
                word_count++;
            }
        }
    }
    
    return word_count;
}

// Main tokenizer functions

Tokenizer* create_tokenizer(int initial_capacity) {
    Tokenizer* t = (Tokenizer*)safe_malloc(sizeof(Tokenizer));
    if (!t) return NULL;
    
    t->vocab_capacity = (initial_capacity > 0) ? initial_capacity : DEFAULT_VOCAB_CAPACITY;
    t->vocab_strs = (char**)safe_malloc(t->vocab_capacity * sizeof(char*));
    if (!t->vocab_strs) {
        free(t);
        return NULL;
    }
    
    t->word_to_id = create_hash_table(t->vocab_capacity * 2);
    if (!t->word_to_id) {
        free(t->vocab_strs);
        free(t);
        return NULL;
    }
    
    if (0) {
        free_hash_table(t->word_to_id);
        free(t->vocab_strs);
        free(t);
        return NULL;
    }
    
    t->vocab_size = 0;
    platform_atomic_store(&t->ref_count, 1); platform_rwlock_init(&t->vocab_lock); platform_rwlock_init(&t->vocab_lock);
    
    // Add special tokens

    t->unk_id = tokenizer_add_word(t, "<UNK>");
    t->bos_id = tokenizer_add_word(t, "<BOS>");
    t->eos_id = tokenizer_add_word(t, "<EOS>");
    t->pad_id = tokenizer_add_word(t, "<PAD>");
    
    if (t->unk_id < 0 || t->bos_id < 0 || t->eos_id < 0 || t->pad_id < 0) {
        free_tokenizer(t);
        return NULL;
    }
    
    return t;
}

void tokenizer_addref(Tokenizer* tokenizer) {
    if (tokenizer) {
        platform_atomic_increment(&tokenizer->ref_count);
    }
}

void free_tokenizer(Tokenizer* tokenizer) {
    if (!tokenizer) return;
    
    int ref_count = platform_atomic_decrement_fetch(&tokenizer->ref_count);
    if (ref_count > 1) return; // Still has references
    
    platform_rwlock_wrlock(&tokenizer->vocab_lock);
    
    for (int i = 0; i < tokenizer->vocab_size; i++) {
        free(tokenizer->vocab_strs[i]);
    }
    free(tokenizer->vocab_strs);
    free_hash_table(tokenizer->word_to_id);
    
    platform_rwlock_unlock_wr(&tokenizer->vocab_lock);
    platform_rwlock_destroy(&tokenizer->vocab_lock);
    free(tokenizer);
}

int tokenizer_add_word(Tokenizer* tokenizer, const char* word) {
    if (!tokenizer || !word || strlen(word) == 0) return -1;
    
    platform_rwlock_wrlock(&tokenizer->vocab_lock);
    
    // Check if word already exists in hash table

    int existing_id = hash_table_lookup(tokenizer->word_to_id, word);
    if (existing_id >= 0) {
        platform_rwlock_unlock_wr(&tokenizer->vocab_lock);
        return existing_id;
    }
    
    // Resizeable vocab. if needed!
    if (tokenizer->vocab_size >= tokenizer->vocab_capacity) {
        int new_capacity = tokenizer->vocab_capacity * 2;
        char** new_vocab = (char**)realloc(tokenizer->vocab_strs, 
                                          new_capacity * sizeof(char*));
        if (!new_vocab) {
            platform_rwlock_unlock_wr(&tokenizer->vocab_lock);
            return -1;
        }
        tokenizer->vocab_strs = new_vocab;
        tokenizer->vocab_capacity = new_capacity;
    }
    
    // Adds word to vocab

    tokenizer->vocab_strs[tokenizer->vocab_size] = strdup(word);
    if (!tokenizer->vocab_strs[tokenizer->vocab_size]) {
        platform_rwlock_unlock_wr(&tokenizer->vocab_lock);
        return -1;
    }
    
    int new_id = tokenizer->vocab_size++;
    
    // Adds to hash table

    if (hash_table_insert(tokenizer->word_to_id, word, new_id) != new_id) {
        // Hash table insertion failed, rollback
        free(tokenizer->vocab_strs[new_id]);
        tokenizer->vocab_size--;
        platform_rwlock_unlock_wr(&tokenizer->vocab_lock);
        return -1;
    }
    
    platform_rwlock_unlock_wr(&tokenizer->vocab_lock);
    return new_id;
}

int tokenizer_get_id(const Tokenizer* tokenizer, const char* word) {
    if (!tokenizer || !word) return -1;
    
    platform_rwlock_rdlock(&tokenizer->vocab_lock);
    int id = hash_table_lookup(tokenizer->word_to_id, word);
    platform_rwlock_unlock_rd(&tokenizer->vocab_lock);
    
    return (id >= 0) ? id : tokenizer->unk_id;
}

const char* tokenizer_get_word(const Tokenizer* tokenizer, int id) {
    if (!tokenizer) return NULL;
    
    platform_rwlock_rdlock(&tokenizer->vocab_lock);
    const char* word = NULL;
    
    if (id >= 0 && id < tokenizer->vocab_size) {
        word = tokenizer->vocab_strs[id];
    } else {
        word = tokenizer->vocab_strs[tokenizer->unk_id];
    }
    
    platform_rwlock_unlock_rd(&tokenizer->vocab_lock);
    return word;
}

int tokenizer_build_from_corpus(Tokenizer* tokenizer, const char* corpus_path, int min_freq) {
    if (!tokenizer || !corpus_path || min_freq < 1) {
        fprintf(stderr, "Invalid parameters for tokenizer_build_from_corpus\n");
        return 0;
    }
    
    FILE* fp = fopen(corpus_path, "r");
    if (!fp) {
        fprintf(stderr, "Failed to open corpus file '%s': %s\n", corpus_path, strerror(errno));
        return 0;
    }
    
    FreqTable* freq_table = create_freq_table(HASH_TABLE_SIZE);
    if (!freq_table) {
        fprintf(stderr, "Failed to create frequency table\n");
        fclose(fp);
        return 0;
    }
    
    printf("Building vocabulary from corpus '%s'...\n", corpus_path);
    
    char* line = NULL;
    size_t line_cap = 0;
    ssize_t line_len;
    long lines_processed = 0;
    long total_words = 0;
    
    while ((line_len = platform_getline(&line, &line_cap, fp)) != -1) {
        lines_processed++;
        total_words += extract_words_from_line(line, freq_table);
        
        if (lines_processed % 10000 == 0) {
            printf("Processed %ld lines, %ld words...\n", lines_processed, total_words);
        }
    }
    
    free(line);
    fclose(fp);
    
    printf("Corpus processing complete. Lines: %ld, Words: %ld\n", lines_processed, total_words);
    printf("Adding words with frequency >= %d to vocabulary...\n", min_freq);
    
    int words_added = 0;
    for (size_t i = 0; i < freq_table->capacity; i++) {
        WordFreq* freq = freq_table->buckets[i];
        while (freq) {
            if (freq->count >= min_freq && strlen(freq->word) > 0) {
                if (tokenizer_add_word(tokenizer, freq->word) >= 0) {
                    words_added++;
                }
            }
            freq = freq->next;
        }
    }
    
    free_freq_table(freq_table);
    
    printf("Vocabulary building complete. Added %d words (min_freq=%d)\n", words_added, min_freq);
    printf("Total vocabulary size: %d\n", tokenizer->vocab_size);
    
    return 1;
}

int tokenizer_save(const Tokenizer* tokenizer, const char* path) {
    if (!tokenizer || !path) return 0;
    
    FILE* fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "Failed to open file '%s' for writing: %s\n", path, strerror(errno));
        return 0;
    }
    
    platform_rwlock_rdlock(&tokenizer->vocab_lock);
    
    // Write header with validation info

    fprintf(fp, "vocab_size: %d\n", tokenizer->vocab_size);
    fprintf(fp, "unk_id: %d\n", tokenizer->unk_id);
    fprintf(fp, "bos_id: %d\n", tokenizer->bos_id);
    fprintf(fp, "eos_id: %d\n", tokenizer->eos_id);
    fprintf(fp, "pad_id: %d\n", tokenizer->pad_id);
    fprintf(fp, "checksum: %u\n", hash_string("TOKENIZER_V1"));
    
    // Write vocab

    for (int i = 0; i < tokenizer->vocab_size; i++) {
        fprintf(fp, "%s\n", tokenizer->vocab_strs[i]);
    }
    
    platform_rwlock_unlock_rd(&tokenizer->vocab_lock);
    
    if (fclose(fp) != 0) {
        fprintf(stderr, "Error closing file '%s': %s\n", path, strerror(errno));
        return 0;
    }
    
    printf("Tokenizer saved to '%s' (vocab_size: %d)\n", path, tokenizer->vocab_size);
    return 1;
}

Tokenizer* tokenizer_load(const char* path) {
    if (!path) return NULL;
    
    FILE* fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "Failed to open tokenizer file '%s': %s\n", path, strerror(errno));
        return NULL;
    }
    
    int vocab_size = 0, unk_id = 0, bos_id = 0, eos_id = 0, pad_id = 0;
    unsigned int checksum = 0;
    
    // Read and validate header

    if (fscanf(fp, "vocab_size: %d\n", &vocab_size) != 1 || vocab_size <= 0) {
        fprintf(stderr, "Invalid vocab_size in tokenizer file\n");
        goto load_error;
    }
    
    if (fscanf(fp, "unk_id: %d\n", &unk_id) != 1 ||
        fscanf(fp, "bos_id: %d\n", &bos_id) != 1 ||
        fscanf(fp, "eos_id: %d\n", &eos_id) != 1 ||
        fscanf(fp, "pad_id: %d\n", &pad_id) != 1) {
        fprintf(stderr, "Invalid special token IDs in tokenizer file\n");
        goto load_error;
    }
    
    if (fscanf(fp, "checksum: %u\n", &checksum) != 1 || 
        checksum != hash_string("TOKENIZER_V1")) {
        fprintf(stderr, "Invalid checksum in tokenizer file\n");
        goto load_error;
    }
    
    // Validate special token IDs
    if (unk_id < 0 || bos_id < 0 || eos_id < 0 || pad_id < 0 ||
        unk_id >= vocab_size || bos_id >= vocab_size || 
        eos_id >= vocab_size || pad_id >= vocab_size) {
        fprintf(stderr, "Special token IDs out of range\n");
        goto load_error;
    }
    
    // Creates tokenizer

    Tokenizer* t = (Tokenizer*)safe_malloc(sizeof(Tokenizer));
    if (!t) goto load_error;
    
    t->vocab_capacity = vocab_size;
    t->vocab_strs = (char**)safe_malloc(t->vocab_capacity * sizeof(char*));
    if (!t->vocab_strs) {
        free(t);
        goto load_error;
    }
    
    t->word_to_id = create_hash_table(vocab_size * 2);
    if (!t->word_to_id) {
        free(t->vocab_strs);
        free(t);
        goto load_error;
    }
    
    if (0) {
        free_hash_table(t->word_to_id);
        free(t->vocab_strs);
        free(t);
        goto load_error;
    }
    
    t->vocab_size = 0;
    platform_atomic_store(&t->ref_count, 1); platform_rwlock_init(&t->vocab_lock); platform_rwlock_init(&t->vocab_lock);
    t->unk_id = unk_id;
    t->bos_id = bos_id;
    t->eos_id = eos_id;
    t->pad_id = pad_id;
    
    // Load vocabulary
    char line_buf[MAX_WORD_LEN + 2];
    for (int i = 0; i < vocab_size; i++) {
        if (!fgets(line_buf, sizeof(line_buf), fp)) {
            fprintf(stderr, "Unexpected end of file while reading vocabulary\n");
            free_tokenizer(t);
            goto load_error;
        }
        
        // Removes newline
        line_buf[strcspn(line_buf, "\n")] = '\0';
        
        if (strlen(line_buf) == 0) {
            fprintf(stderr, "Empty word at position %d\n", i);
            free_tokenizer(t);
            goto load_error;
        }
        
        t->vocab_strs[i] = strdup(line_buf);
        if (!t->vocab_strs[i]) {
            free_tokenizer(t);
            goto load_error;
        }
        
        // Adds to hash table
        if (hash_table_insert(t->word_to_id, line_buf, i) != i) {
            fprintf(stderr, "Failed to add word '%s' to hash table\n", line_buf);
            free_tokenizer(t);
            goto load_error;
        }
        
        t->vocab_size++;
    }
    
    fclose(fp);
    
    // Final validation

    if (t->vocab_size != vocab_size) {
        fprintf(stderr, "Loaded vocab size (%d) doesn't match expected (%d)\n", 
                t->vocab_size, vocab_size);
        free_tokenizer(t);
        return NULL;
    }
    
    printf("Tokenizer loaded from '%s'. Vocab size: %d\n", path, t->vocab_size);
    return t;

load_error:
    if (fp) fclose(fp);
    return NULL;
}

int tokenizer_encode(const Tokenizer* tokenizer, const char* text,
                     int* token_ids_out, int max_output_tokens,
                     int add_bos, int add_eos) {
    if (!tokenizer || !text || !token_ids_out || max_output_tokens <= 0) {
        return 0;
    }
    
    int count = 0;
    
    // Add BOS token

    if (add_bos && count < max_output_tokens) {
        token_ids_out[count++] = tokenizer->bos_id;
    }
    
    char word_buffer[MAX_WORD_LEN];
    const char* ptr = text;
    
    while (*ptr && count < max_output_tokens - (add_eos ? 1 : 0)) {
        // Skip non-word characters
        while (*ptr && !is_valid_word_char(*ptr)) ptr++;
        if (!*ptr) break;
        
        // Extract word
        int len = 0;
        while (*ptr && is_valid_word_char(*ptr) && len < MAX_WORD_LEN - 1) {
            word_buffer[len++] = *ptr++;
        }
        word_buffer[len] = '\0';
        
        if (len > 0) {
            normalize_word(word_buffer);
            if (strlen(word_buffer) > 0) {
                int token_id = tokenizer_get_id(tokenizer, word_buffer);
                token_ids_out[count++] = token_id;
            }
        }
    }
    
    // Add EOS token

    if (add_eos && count < max_output_tokens) {
        token_ids_out[count++] = tokenizer->eos_id;
    }
    
    return count;
}

char* tokenizer_decode(const Tokenizer* tokenizer, const int* token_ids, int num_tokens) {
    if (!tokenizer || !token_ids || num_tokens <= 0) {
        return strdup(""); // Return empty string
    }
    
    size_t buffer_capacity = 1024;
    char* buffer = (char*)safe_malloc(buffer_capacity);
    if (!buffer) return NULL;
    
    buffer[0] = '\0';
    size_t current_len = 0;
    int first_word = 1;
    
    for (int i = 0; i < num_tokens; i++) {
        // Skip special tokens appropriately
        if (token_ids[i] == tokenizer->bos_id) continue;
        if (token_ids[i] == tokenizer->eos_id) break;
        if (token_ids[i] == tokenizer->pad_id) continue;
        
        const char* word = tokenizer_get_word(tokenizer, token_ids[i]);
        if (!word) continue;
        
        size_t word_len = strlen(word);
        size_t space_len = first_word ? 0 : 1;
        
        // Resize buffer. if needed!
        while (current_len + space_len + word_len + 1 > buffer_capacity) {
            buffer_capacity *= 2;
            char* new_buffer = (char*)realloc(buffer, buffer_capacity);
            if (!new_buffer) {
                free(buffer);
                return NULL;
            }
            buffer = new_buffer;
        }
        
        // Adds space if not first word
        if (!first_word) {
            buffer[current_len++] = ' ';
        }
        
        // Adds word

        memcpy(buffer + current_len, word, word_len);
        current_len += word_len;
        buffer[current_len] = '\0';
        
        first_word = 0;
    }
    
    return buffer;
}

// Utility functions
int tokenizer_get_vocab_size(const Tokenizer* tokenizer) {
    if (!tokenizer) return 0;
    
    platform_rwlock_rdlock(&tokenizer->vocab_lock);
    int size = tokenizer->vocab_size;
    platform_rwlock_unlock_rd(&tokenizer->vocab_lock);
    
    return size;
}

void tokenizer_print_stats(const Tokenizer* tokenizer) {
    if (!tokenizer) return;
    
    platform_rwlock_rdlock(&tokenizer->vocab_lock);
    
    printf("Tokenizer Statistics:\n");
    printf("  Vocabulary size: %d\n", tokenizer->vocab_size);
    printf("  Vocabulary capacity: %d\n", tokenizer->vocab_capacity);
    printf("  Special tokens:\n");
    printf("    UNK: %d ('%s')\n", tokenizer->unk_id, 
           tokenizer_get_word(tokenizer, tokenizer->unk_id));
    printf("    BOS: %d ('%s')\n", tokenizer->bos_id, 
           tokenizer_get_word(tokenizer, tokenizer->bos_id));
    printf("    EOS: %d ('%s')\n", tokenizer->eos_id, 
           tokenizer_get_word(tokenizer, tokenizer->eos_id));
    printf("    PAD: %d ('%s')\n", tokenizer->pad_id, 
           tokenizer_get_word(tokenizer, tokenizer->pad_id));
    printf("  Hash table size: " SIZE_T_FMT "\n", ((HashTable*)tokenizer->word_to_id)->size);
    printf("  Reference count: %ld\n", (long)platform_atomic_load(&tokenizer->ref_count));
    
    platform_rwlock_unlock_rd(&tokenizer->vocab_lock);
}