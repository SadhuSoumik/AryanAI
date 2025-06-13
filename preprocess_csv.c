#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdbool.h>
#include <errno.h>
#include <sys/stat.h>

#define MAX_LINE_LENGTH 8192
#define MAX_FIELDS 32
#define MAX_DESCRIPTION_LENGTH 2000
#define MIN_TEXT_LENGTH 5
#define MAX_SCHEMA_FIELDS 16
#define BUFFER_SIZE 65536

typedef enum {
    TYPE_UNDEFINED,
    TYPE_SENTIMENT,
    TYPE_LEETCODE,
    TYPE_CUSTOM,
    TYPE_CLASSIFICATION,
    TYPE_QA
} DatasetType;

typedef enum {
    ENCODING_UTF8,
    ENCODING_LATIN1,
    ENCODING_AUTO
} EncodingType;

typedef struct {
    char name[64];
    int index;
    bool required;
    bool is_label;
    int min_length;
    int max_length;
} FieldSchema;

typedef struct {
    DatasetType type;
    EncodingType encoding;
    char delimiter;
    bool has_header;
    bool strict_mode;
    bool remove_duplicates;
    bool balance_classes;
    bool validate_data;
    int max_lines;
    int skip_lines;
    double train_split;
    FieldSchema fields[MAX_SCHEMA_FIELDS];
    int field_count;
    char output_format[32]; // json, txt, csv
} ProcessingConfig;

typedef struct {
    char **data;
    int count;
    int capacity;
} StringArray;

// Statistics tracking
typedef struct {
    int total_lines;
    int processed_lines;
    int skipped_lines;
    int error_lines;
    int duplicate_lines;
    double avg_text_length;
    int class_distribution[32];
    int unique_classes;
} ProcessingStats;

// Enhanced string utilities
static void safe_strcpy(char *dest, const char *src, size_t dest_size) {
    if (!dest || !src || dest_size == 0) return;
    strncpy(dest, src, dest_size - 1);
    dest[dest_size - 1] = '\0';
}

static int strcasecmp_portable(const char *s1, const char *s2) {
    if (!s1 || !s2) return s1 ? 1 : (s2 ? -1 : 0);
    while (*s1 && *s2) {
        int c1 = tolower((unsigned char)*s1);
        int c2 = tolower((unsigned char)*s2);
        if (c1 != c2) return c1 - c2;
        s1++;
        s2++;
    }
    return tolower((unsigned char)*s1) - tolower((unsigned char)*s2);
}

// Enhanced text cleaning with more HTML entities and better Unicode handling
static void clean_text(char *text, size_t max_len, bool strict) {
    if (!text) return;
    size_t len = strlen(text);
    if (len == 0 || strcasecmp_portable(text, "nan") == 0 || 
        strcasecmp_portable(text, "null") == 0 || strcasecmp_portable(text, "n/a") == 0) {
        text[0] = '\0';
        return;
    }

    // Trim leading/trailing whitespace
    char *start = text;
    while (*start && isspace((unsigned char)*start)) start++;
    char *end = text + len - 1;
    while (end > start && isspace((unsigned char)*end)) end--;
    *(end + 1) = '\0';
    if (start != text) memmove(text, start, strlen(start) + 1);

    // Enhanced HTML entity decoding
    struct { const char *entity; const char *replacement; } entities[] = {
        {"&lt;", "<"}, {"&gt;", ">"}, {"&amp;", "&"}, {"&quot;", "\""}, 
        {"&apos;", "'"}, {"&nbsp;", " "}, {"&#39;", "'"}, {"&#34;", "\""},
        {"&hellip;", "..."}, {"&mdash;", "--"}, {"&ndash;", "-"},
        {"&lsquo;", "'"}, {"&rsquo;", "'"}, {"&ldquo;", "\""}, {"&rdquo;", "\""}
    };
    
    for (int i = 0; i < sizeof(entities) / sizeof(entities[0]); i++) {
        char *pos;
        while ((pos = strstr(text, entities[i].entity)) != NULL) {
            size_t entity_len = strlen(entities[i].entity);
            size_t repl_len = strlen(entities[i].replacement);
            memmove(pos + repl_len, pos + entity_len, strlen(pos + entity_len) + 1);
            memcpy(pos, entities[i].replacement, repl_len);
        }
    }

    // Strip HTML/XML tags with better handling
    char *tag_start;
    while ((tag_start = strchr(text, '<')) != NULL) {
        char *tag_end = strchr(tag_start, '>');
        if (!tag_end) {
            *tag_start = '\0';
            break;
        }
        memmove(tag_start, tag_end + 1, strlen(tag_end + 1) + 1);
    }

    // Remove control characters and normalize whitespace
    char *rd = text, *wr = text;
    bool last_space = false;
    while (*rd) {
        unsigned char c = (unsigned char)*rd;
        if (c < 32 && c != '\t' && c != '\n' && c != '\r') {
            // Skip control characters
        } else if (isspace(c)) {
            if (!last_space && wr != text) {
                *wr++ = ' ';
                last_space = true;
            }
        } else {
            *wr++ = *rd;
            last_space = false;
        }
        rd++;
    }
    *wr = '\0';

    // Remove excessive punctuation in strict mode
    if (strict) {
        char *src = text, *dst = text;
        int punct_count = 0;
        while (*src) {
            if (ispunct((unsigned char)*src)) {
                if (++punct_count <= 3) *dst++ = *src;
            } else {
                punct_count = 0;
                *dst++ = *src;
            }
            src++;
        }
        *dst = '\0';
    }

    // Final length checks
    len = strlen(text);
    if (len < MIN_TEXT_LENGTH) {
        text[0] = '\0';
        return;
    }
    if (len >= max_len) {
        text[max_len - 1] = '\0';
    }
}

// Advanced CSV parser with configurable delimiter and quote handling
static int parse_csv_line(char *line, char fields[][MAX_LINE_LENGTH], int max_fields, char delimiter) {
    int field_count = 0;
    bool in_quotes = false;
    char *ptr = line;
    char buffer[MAX_LINE_LENGTH];
    int buf_pos = 0;
    char quote_char = '"';

    // Auto-detect quote character if not standard
    if (strchr(line, '\'') && !strchr(line, '"')) {
        quote_char = '\'';
    }

    while (*ptr && field_count < max_fields) {
        if (*ptr == quote_char) {
            if (in_quotes && *(ptr + 1) == quote_char) {
                // Escaped quote
                if (buf_pos < MAX_LINE_LENGTH - 1) buffer[buf_pos++] = quote_char;
                ptr += 2;
                continue;
            }
            in_quotes = !in_quotes;
            ptr++;
        } else if (*ptr == delimiter && !in_quotes) {
            buffer[buf_pos] = '\0';
            safe_strcpy(fields[field_count++], buffer, MAX_LINE_LENGTH);
            buf_pos = 0;
            ptr++;
        } else {
            if (buf_pos < MAX_LINE_LENGTH - 1) buffer[buf_pos++] = *ptr;
            ptr++;
        }
    }
    
    // Handle last field
    if (field_count < max_fields) {
        buffer[buf_pos] = '\0';
        safe_strcpy(fields[field_count++], buffer, MAX_LINE_LENGTH);
    }
    
    return field_count;
}

// Auto-detect CSV delimiter
static char detect_delimiter(FILE *file) {
    char sample[4096];
    long pos = ftell(file);
    
    if (!fgets(sample, sizeof(sample), file)) {
        fseek(file, pos, SEEK_SET);
        return ',';
    }
    fseek(file, pos, SEEK_SET);

    int comma_count = 0, semicolon_count = 0, tab_count = 0, pipe_count = 0;
    char *ptr = sample;
    bool in_quotes = false;
    
    while (*ptr) {
        if (*ptr == '"') in_quotes = !in_quotes;
        else if (!in_quotes) {
            switch (*ptr) {
                case ',': comma_count++; break;
                case ';': semicolon_count++; break;
                case '\t': tab_count++; break;
                case '|': pipe_count++; break;
            }
        }
        ptr++;
    }

    if (tab_count > 0) return '\t';
    if (semicolon_count > comma_count && semicolon_count > pipe_count) return ';';
    if (pipe_count > comma_count) return '|';
    return ',';
}

// Enhanced file encoding detection
static EncodingType detect_encoding(FILE *file) {
    unsigned char buffer[3];
    long pos = ftell(file);
    
    if (fread(buffer, 1, 3, file) == 3) {
        if (buffer[0] == 0xEF && buffer[1] == 0xBB && buffer[2] == 0xBF) {
            fseek(file, pos + 3, SEEK_SET); // Skip UTF-8 BOM
            return ENCODING_UTF8;
        }
    }
    
    fseek(file, pos, SEEK_SET);
    return ENCODING_AUTO;
}

// Data validation functions
static bool validate_field(const char *field, const FieldSchema *schema) {
    if (!field || !schema) return false;
    
    int len = strlen(field);
    if (schema->required && len == 0) return false;
    if (len < schema->min_length || (schema->max_length > 0 && len > schema->max_length)) {
        return false;
    }
    
    return true;
}

// Duplicate detection using simple hash
static unsigned int hash_string(const char *str) {
    unsigned int hash = 5381;
    while (*str) {
        hash = ((hash << 5) + hash) + (unsigned char)*str++;
    }
    return hash;
}

static bool is_duplicate(const char *text, unsigned int *seen_hashes, int *hash_count, int max_hashes) {
    unsigned int hash = hash_string(text);
    for (int i = 0; i < *hash_count; i++) {
        if (seen_hashes[i] == hash) return true;
    }
    if (*hash_count < max_hashes) {
        seen_hashes[(*hash_count)++] = hash;
    }
    return false;
}

// Enhanced output formatting
static void write_output_json(FILE *out, const char *fields[], const FieldSchema *schema, int field_count) {
    fprintf(out, "{");
    for (int i = 0; i < field_count; i++) {
        if (i > 0) fprintf(out, ",");
        fprintf(out, "\"%s\":\"%s\"", schema[i].name, fields[i]);
    }
    fprintf(out, "}\n");
}

static void write_output_txt(FILE *out, const char *fields[], const FieldSchema *schema, 
                           int field_count, DatasetType type) {
    switch (type) {
        case TYPE_SENTIMENT:
            fprintf(out, "Text: %s\nSentiment: %s\n---\n", fields[0], fields[1]);
            break;
        case TYPE_LEETCODE:
            fprintf(out, "Problem: %s\nDifficulty: %s\nDescription: %s\n---\n", 
                   fields[0], fields[1], fields[2]);
            break;
        case TYPE_QA:
            fprintf(out, "Question: %s\nAnswer: %s\n---\n", fields[0], fields[1]);
            break;
        case TYPE_CLASSIFICATION:
            fprintf(out, "Text: %s\nCategory: %s\n---\n", fields[0], fields[1]);
            break;
        default:
            for (int i = 0; i < field_count; i++) {
                fprintf(out, "%s: %s\n", schema[i].name, fields[i]);
            }
            fprintf(out, "---\n");
    }
}

// Main processing function with enhanced capabilities
static void process_file_enhanced(const char *input_file, const char *output_file, 
                                ProcessingConfig *config, ProcessingStats *stats) {
    FILE *csv = fopen(input_file, "rb");
    if (!csv) {
        fprintf(stderr, "Error opening input file '%s': %s\n", input_file, strerror(errno));
        return;
    }

    FILE *out = fopen(output_file, "w");
    if (!out) {
        fprintf(stderr, "Error opening output file '%s': %s\n", output_file, strerror(errno));
        fclose(csv);
        return;
    }

    // Auto-detect encoding and delimiter if needed
    if (config->encoding == ENCODING_AUTO) {
        config->encoding = detect_encoding(csv);
    }
    if (config->delimiter == '\0') {
        config->delimiter = detect_delimiter(csv);
    }

    char line[MAX_LINE_LENGTH];
    char fields[MAX_FIELDS][MAX_LINE_LENGTH];
    const char *field_ptrs[MAX_FIELDS];
    unsigned int *seen_hashes = NULL;
    int hash_count = 0;
    
    if (config->remove_duplicates) {
        int alloc_size = config->max_lines > 0 ? config->max_lines : 100000; // Default allocation for unlimited mode
        seen_hashes = calloc(alloc_size, sizeof(unsigned int));
    }

    // Skip initial lines if requested
    for (int i = 0; i < config->skip_lines; i++) {
        if (!fgets(line, sizeof(line), csv)) break;
    }

    // Handle header
    if (config->has_header && fgets(line, sizeof(line), csv)) {
        stats->total_lines++;
        printf("Header: %s", line);
    }

    // Process data lines
    while (fgets(line, sizeof(line), csv) && 
           (config->max_lines == 0 || stats->processed_lines < config->max_lines)) {
        stats->total_lines++;
        
        // Remove newline
        line[strcspn(line, "\r\n")] = '\0';
        if (strlen(line) == 0) continue;

        int field_count = parse_csv_line(line, fields, MAX_FIELDS, config->delimiter);
        
        // Validate minimum field count
        bool valid = true;
        if (field_count < config->field_count) {
            stats->error_lines++;
            if (config->strict_mode) continue;
            valid = false;
        }

        // Clean and validate fields
        for (int i = 0; i < field_count && i < config->field_count; i++) {
            clean_text(fields[i], MAX_LINE_LENGTH, config->strict_mode);
            field_ptrs[i] = fields[i];
            
            if (config->validate_data && !validate_field(fields[i], &config->fields[i])) {
                valid = false;
                break;
            }
        }

        if (!valid) {
            stats->error_lines++;
            if (config->strict_mode) continue;
        }

        // Check for duplicates
        if (config->remove_duplicates && field_count > 0) {
            int alloc_size = config->max_lines > 0 ? config->max_lines : 100000;
            if (is_duplicate(fields[0], seen_hashes, &hash_count, alloc_size)) {
                stats->duplicate_lines++;
                continue;
            }
        }

        // Write output in specified format
        if (strcmp(config->output_format, "json") == 0) {
            write_output_json(out, field_ptrs, config->fields, field_count);
        } else {
            write_output_txt(out, field_ptrs, config->fields, field_count, config->type);
        }

        stats->processed_lines++;
        stats->avg_text_length += strlen(fields[0]);
        
        if (stats->processed_lines % 1000 == 0) {
            printf("Processed %d/%d lines (%.1f%%)...\n", 
                   stats->processed_lines, stats->total_lines,
                   100.0 * stats->processed_lines / stats->total_lines);
        }
    }

    if (seen_hashes) free(seen_hashes);
    fclose(csv);
    fclose(out);
    
    stats->avg_text_length /= (stats->processed_lines > 0 ? stats->processed_lines : 1);
}

// Configuration presets
static void setup_sentiment_config(ProcessingConfig *config) {
    config->type = TYPE_SENTIMENT;
    config->field_count = 2;
    safe_strcpy(config->fields[0].name, "text", sizeof(config->fields[0].name));
    config->fields[0].index = 0;
    config->fields[0].required = true;
    config->fields[0].min_length = MIN_TEXT_LENGTH;
    safe_strcpy(config->fields[1].name, "sentiment", sizeof(config->fields[1].name));
    config->fields[1].index = 1;
    config->fields[1].required = true;
    config->fields[1].is_label = true;
}

static void setup_leetcode_config(ProcessingConfig *config) {
    config->type = TYPE_LEETCODE;
    config->field_count = 3;
    safe_strcpy(config->fields[0].name, "title", sizeof(config->fields[0].name));
    config->fields[0].index = 0;
    config->fields[0].required = true;
    safe_strcpy(config->fields[1].name, "difficulty", sizeof(config->fields[1].name));
    config->fields[1].index = 1;
    config->fields[1].required = true;
    safe_strcpy(config->fields[2].name, "description", sizeof(config->fields[2].name));
    config->fields[2].index = 2;
    config->fields[2].required = true;
    config->fields[2].min_length = 50;
}

static DatasetType detect_type_enhanced(const char *input_file, const char *override) {
    if (override) {
        if (strcmp(override, "sentiment") == 0) return TYPE_SENTIMENT;
        if (strcmp(override, "leetcode") == 0) return TYPE_LEETCODE;
        if (strcmp(override, "qa") == 0) return TYPE_QA;
        if (strcmp(override, "classification") == 0) return TYPE_CLASSIFICATION;
        if (strcmp(override, "custom") == 0) return TYPE_CUSTOM;
    }
    
    // Enhanced auto-detection
    const char *filename = strrchr(input_file, '/');
    filename = filename ? filename + 1 : input_file;
    
    if (strstr(filename, "sentiment") || strstr(filename, "review")) return TYPE_SENTIMENT;
    if (strstr(filename, "leetcode") || strstr(filename, "problem")) return TYPE_LEETCODE;
    if (strstr(filename, "qa") || strstr(filename, "question")) return TYPE_QA;
    if (strstr(filename, "class") || strstr(filename, "category")) return TYPE_CLASSIFICATION;
    
    return TYPE_UNDEFINED;
}

static void print_usage(const char *prog_name) {
    printf("Enhanced CSV Processor v2.0\n");
    printf("Usage: %s <input_file> --output <output_file> [options]\n\n", prog_name);
    printf("Options:\n");
    printf("  --output <file>          Output file path (required)\n");
    printf("  --type <type>            Dataset type: sentiment, leetcode, qa, classification, custom\n");
    printf("  --max-lines <n>          Maximum lines to process (default: 0 = no limit)\n");
    printf("  --skip-lines <n>         Skip first n lines (default: 0)\n");
    printf("  --delimiter <char>       CSV delimiter (auto-detect if not specified)\n");
    printf("  --format <fmt>           Output format: txt, json, csv (default: txt)\n");
    printf("  --encoding <enc>         Input encoding: utf8, latin1, auto (default: auto)\n");
    printf("  --no-header              CSV has no header row\n");
    printf("  --strict                 Enable strict validation mode\n");
    printf("  --remove-duplicates      Remove duplicate entries\n");
    printf("  --validate               Enable data validation\n");
    printf("  --train-split <ratio>    Split ratio for training data (0.0-1.0)\n");
    printf("  --help                   Show this help message\n");
}

static void print_stats(const ProcessingStats *stats) {
    printf("\n=== Processing Statistics ===\n");
    printf("Total lines read: %d\n", stats->total_lines);
    printf("Lines processed: %d\n", stats->processed_lines);
    printf("Lines skipped: %d\n", stats->skipped_lines);
    printf("Error lines: %d\n", stats->error_lines);
    printf("Duplicate lines: %d\n", stats->duplicate_lines);
    printf("Average text length: %.1f characters\n", stats->avg_text_length);
    printf("Success rate: %.1f%%\n", 
           stats->total_lines > 0 ? 100.0 * stats->processed_lines / stats->total_lines : 0);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    // Initialize configuration with defaults
    ProcessingConfig config = {
        .type = TYPE_UNDEFINED,
        .encoding = ENCODING_AUTO,
        .delimiter = '\0',
        .has_header = true,
        .strict_mode = false,
        .remove_duplicates = false,
        .validate_data = false,
        .max_lines = 0, // 0 means no limit - process entire file
        .skip_lines = 0,
        .train_split = 0.8,
        .field_count = 0
    };
    safe_strcpy(config.output_format, "txt", sizeof(config.output_format));

    const char *input_file = argv[1];
    const char *output_file = NULL;
    const char *type_arg = NULL;

    // Parse command line arguments
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return EXIT_SUCCESS;
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_file = argv[++i];
        } else if (strcmp(argv[i], "--type") == 0 && i + 1 < argc) {
            type_arg = argv[++i];
        } else if (strcmp(argv[i], "--max-lines") == 0 && i + 1 < argc) {
            config.max_lines = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--skip-lines") == 0 && i + 1 < argc) {
            config.skip_lines = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--delimiter") == 0 && i + 1 < argc) {
            config.delimiter = argv[++i][0];
        } else if (strcmp(argv[i], "--format") == 0 && i + 1 < argc) {
            safe_strcpy(config.output_format, argv[++i], sizeof(config.output_format));
        } else if (strcmp(argv[i], "--no-header") == 0) {
            config.has_header = false;
        } else if (strcmp(argv[i], "--strict") == 0) {
            config.strict_mode = true;
        } else if (strcmp(argv[i], "--remove-duplicates") == 0) {
            config.remove_duplicates = true;
        } else if (strcmp(argv[i], "--validate") == 0) {
            config.validate_data = true;
        } else if (strcmp(argv[i], "--train-split") == 0 && i + 1 < argc) {
            config.train_split = atof(argv[++i]);
        }
    }

    // Validation
    if (!output_file) {
        fprintf(stderr, "Error: --output parameter is required\n");
        return EXIT_FAILURE;
    }

    if (config.max_lines < 0) {
        fprintf(stderr, "Error: max-lines must be non-negative (0 means no limit)\n");
        return EXIT_FAILURE;
    }

    // Test input file accessibility
    struct stat st;
    if (stat(input_file, &st) != 0) {
        fprintf(stderr, "Error: Input file '%s' not accessible: %s\n", input_file, strerror(errno));
        return EXIT_FAILURE;
    }

    // Detect and setup dataset type
    config.type = detect_type_enhanced(input_file, type_arg);
    if (config.type == TYPE_UNDEFINED) {
        fprintf(stderr, "Could not auto-detect dataset type. Please specify --type\n");
        return EXIT_FAILURE;
    }

    // Setup field schemas based on type
    switch (config.type) {
        case TYPE_SENTIMENT:
            setup_sentiment_config(&config);
            break;
        case TYPE_LEETCODE:
            setup_leetcode_config(&config);
            break;
        default:
            config.field_count = 2; // Default for custom types
            break;
    }

    // Initialize statistics
    ProcessingStats stats = {0};

    // Process the file
    printf("Enhanced CSV Processor v2.0\n");
    printf("Input: %s (%.1fKB)\n", input_file, st.st_size / 1024.0);
    printf("Output: %s\n", output_file);
    printf("Type: %s\n", type_arg ? type_arg : "auto-detected");
    printf("Format: %s\n", config.output_format);
    printf("Max lines: %s\n", config.max_lines == 0 ? "unlimited" : "limited");
    if (config.max_lines > 0) {
        printf("Limit: %d lines\n", config.max_lines);
    }
    printf("Processing...\n");

    process_file_enhanced(input_file, output_file, &config, &stats);

    // Print final statistics
    print_stats(&stats);
    
    if (stats.processed_lines > 0) {
        printf("\nProcessing completed successfully!\n");
        printf("Output file: %s\n", output_file);
        printf("You can now train with: ./AryanAi.exe train --data %s\n", output_file);
    } else {
        printf("\nNo data was processed. Please check your input file and settings.\n");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}