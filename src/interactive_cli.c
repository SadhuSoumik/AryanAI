/**
 * @file interactive_cli.c
 * @brief Interactive Command Line Interface for C Transformer
 * 
 * This module provides an interactive menu-driven interface that makes
 * the transformer more user-friendly. It handles user input, displays
 * menus, and guides users through training and generation processes.
 */

#include "platform.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#ifdef _WIN32
    #include <conio.h>
    #define CLEAR_SCREEN "cls"
#else
    #include <termios.h>
    #include <unistd.h>
    #define CLEAR_SCREEN "clear"
#endif

#include "interactive_cli.h"
#include "transformer_config.h"

// Default file paths for automatic saving/loading
#define DEFAULT_VOCAB_FILE "aryan_vocab.vocab"
#define DEFAULT_WEIGHTS_FILE "aryan_model.bin"
#define DEFAULT_TRAINING_DATA "training_data.txt"

/**
 * Cross-platform function to get a single character input without pressing Enter
 */
static int get_char(void) {
#ifdef _WIN32
    return _getch();
#else
    struct termios oldt, newt;
    int ch;
    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    ch = getchar();
    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    return ch;
#endif
}

/**
 * Cross-platform screen clearing function
 */
static void clear_screen(void) {
    system(CLEAR_SCREEN);
}

/**
 * Display the main menu banner
 */
static void display_banner(void) {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║                     🧠 ARYAN ALPHA AI 🧠                     ║\n");
    printf("║              High-Performance C Transformer Model           ║\n");
    printf("║                     Interactive CLI v2.0                    ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    printf("\n");
}

/**
 * Display the main menu options
 */
static void display_main_menu(void) {
    printf("┌──────────────────────────────────────────────────────────────┐\n");
    printf("│                          MAIN MENU                          │\n");
    printf("├──────────────────────────────────────────────────────────────┤\n");
    printf("│  [1] 🚀 Quick Train Model (Recommended Settings)            │\n");
    printf("│  [2] 🎯 Advanced Training (Custom Parameters)               │\n");
    printf("│  [3] 💬 Generate Text (Use Trained Model)                   │\n");
    printf("│  [4] 📊 Model Status & Information                          │\n");
    printf("│  [5] ⚙️  Settings & Configuration                            │\n");
    printf("│  [6] 📖 Help & Documentation                                │\n");
    printf("│  [7] 🔄 Data Preprocessing Tools                            │\n");
    printf("│  [0] 🚪 Exit                                                 │\n");
    printf("└──────────────────────────────────────────────────────────────┘\n");
    printf("\nPress a number key (0-7): ");
}

/**
 * Get string input from user with prompt
 */
static void get_string_input(const char* prompt, char* buffer, size_t max_len) {
    printf("%s", prompt);
    fflush(stdout);
    
    if (fgets(buffer, max_len, stdin) != NULL) {
        // Remove trailing newline
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len-1] == '\n') {
            buffer[len-1] = '\0';
        }
    }
}

/**
 * Get integer input from user with validation
 */
static int get_int_input(const char* prompt, int min_val, int max_val, int default_val) {
    char buffer[64];
    int value;
    
    while (1) {
        printf("%s [%d-%d, default: %d]: ", prompt, min_val, max_val, default_val);
        fflush(stdout);
        
        if (fgets(buffer, sizeof(buffer), stdin) != NULL) {
            // Trim whitespace
            char *trimmed = buffer;
            while (isspace(*trimmed)) trimmed++;
            
            // If empty input, use default
            if (*trimmed == '\0' || *trimmed == '\n') {
                return default_val;
            }
            
            // Try to parse integer
            if (sscanf(trimmed, "%d", &value) == 1) {
                if (value >= min_val && value <= max_val) {
                    return value;
                }
            }
        }
        printf("❌ Invalid input. Please enter a number between %d and %d.\n", min_val, max_val);
    }
}

/**
 * Get float input from user with validation
 */
static float get_float_input(const char* prompt, float min_val, float max_val, float default_val) {
    char buffer[64];
    float value;
    
    while (1) {
        printf("%s [%.6f-%.6f, default: %.6f]: ", prompt, min_val, max_val, default_val);
        fflush(stdout);
        
        if (fgets(buffer, sizeof(buffer), stdin) != NULL) {
            // Trim whitespace
            char *trimmed = buffer;
            while (isspace(*trimmed)) trimmed++;
            
            // If empty input, use default
            if (*trimmed == '\0' || *trimmed == '\n') {
                return default_val;
            }
            
            // Try to parse float
            if (sscanf(trimmed, "%f", &value) == 1) {
                if (value >= min_val && value <= max_val) {
                    return value;
                }
            }
        }
        printf("❌ Invalid input. Please enter a number between %.6f and %.6f.\n", min_val, max_val);
    }
}

/**
 * Check if a file exists
 */
static int file_exists(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (file) {
        fclose(file);
        return 1;
    }
    return 0;
}

/**
 * Handle quick training with optimized settings for accuracy
 */
static void handle_quick_train(void) {
    clear_screen();
    display_banner();
    
    printf("┌──────────────────────────────────────────────────────────────┐\n");
    printf("│              🚀 QUICK TRAIN - OPTIMIZED SETTINGS             │\n");
    printf("├──────────────────────────────────────────────────────────────┤\n");
    printf("│                                                              │\n");
    printf("│  This mode uses optimized settings for best accuracy:       │\n");
    printf("│  • Enhanced Adam optimizer with warm-up                     │\n");
    printf("│  • Cosine learning rate scheduling                          │\n");
    printf("│  • Gradient clipping for stability                          │\n");
    printf("│  • Automatic vocabulary and weights saving                  │\n");
    printf("│                                                              │\n");
    printf("└──────────────────────────────────────────────────────────────┘\n");
    printf("\n");
    
    char training_file[256];
    
    // Check for existing training files
    printf("📁 Available training data files:\n");
    if (file_exists("training_data.txt")) {
        printf("  ✅ training_data.txt (found)\n");
        strcpy(training_file, "training_data.txt");
    } else if (file_exists("data/test_training_data.txt")) {
        printf("  ✅ data/test_training_data.txt (found)\n");
        strcpy(training_file, "data/test_training_data.txt");
    } else if (file_exists("test_data.txt")) {
        printf("  ✅ test_data.txt (found)\n");
        strcpy(training_file, "test_data.txt");
    } else {
        printf("  ❌ No training data found\n");
        printf("\nPlease provide the path to your training data file:\n");
        get_string_input("Training file path: ", training_file, sizeof(training_file));
        
        if (!file_exists(training_file)) {
            printf("❌ File not found: %s\n", training_file);
            printf("Press any key to return to main menu...");
            get_char();
            return;
        }
    }
    
    printf("\n🔧 Quick Training Configuration:\n");
    printf("  • Training Data: %s\n", training_file);
    printf("  • Vocabulary: %s (auto-save)\n", DEFAULT_VOCAB_FILE);
    printf("  • Model Weights: %s (auto-save)\n", DEFAULT_WEIGHTS_FILE);
    printf("  • Enhanced Accuracy Settings: ENABLED\n");
    printf("  • Learning Rate: 1e-4 (with cosine decay)\n");
    printf("  • Batch Size: 8 (optimized)\n");
    printf("  • Epochs: 20 (sufficient for convergence)\n");
    printf("  • Gradient Clipping: 1.0 (prevents instability)\n");
    
    printf("\nPress 'y' to start training, any other key to cancel: ");
    int choice = get_char();
    
    if (choice == 'y' || choice == 'Y') {
        printf("\n\n🚀 Starting Enhanced Training...\n");
        
        // Build enhanced command with accuracy optimizations
        char command[1024];
        snprintf(command, sizeof(command),
            "%s --mode train "
            "--data \"%s\" "
            "--vocab_load \"%s\" "
            "--vocab_save \"%s\" "
            "--weights_save \"%s\" "
            "--lr 1e-4 "
            "--epochs 20 "
            "--batch_size 8 "
            "--seq_len 256 "
            "--d_model 256 "
            "--n_layers 6 "
            "--n_heads 8 "
            "--grad_accum 2",
            CLI_EXECUTABLE_NAME,
            training_file,
            DEFAULT_VOCAB_FILE,
            DEFAULT_VOCAB_FILE,
            DEFAULT_WEIGHTS_FILE
        );
        
        printf("Executing: %s\n\n", command);
        int result = system(command);
        
        if (result == 0) {
            printf("\n✅ Training completed successfully!\n");
            printf("📁 Files saved:\n");
            printf("  • Vocabulary: %s\n", DEFAULT_VOCAB_FILE);
            printf("  • Model Weights: %s\n", DEFAULT_WEIGHTS_FILE);
        } else {
            printf("\n❌ Training failed with error code: %d\n", result);
        }
    } else {
        printf("\n\nTraining cancelled.\n");
    }
    
    printf("\nPress any key to continue...");
    get_char();
}

/**
 * Show model status and available files
 */
static void show_model_status(void) {
    clear_screen();
    display_banner();
    
    printf("┌──────────────────────────────────────────────────────────────┐\n");
    printf("│                      MODEL STATUS                           │\n");
    printf("├──────────────────────────────────────────────────────────────┤\n");
    
    // Check for model files
    printf("│ Vocabulary File:  ");
    if (file_exists(DEFAULT_VOCAB_FILE)) {
        printf("✅ Found (%s)                 │\n", DEFAULT_VOCAB_FILE);
    } else {
        printf("❌ Not found (%s)            │\n", DEFAULT_VOCAB_FILE);
    }
    
    printf("│ Model Weights:    ");
    if (file_exists(DEFAULT_WEIGHTS_FILE)) {
        printf("✅ Found (%s)                │\n", DEFAULT_WEIGHTS_FILE);
    } else {
        printf("❌ Not found (%s)           │\n", DEFAULT_WEIGHTS_FILE);
    }
    
    printf("│ Training Data:    ");
    if (file_exists(DEFAULT_TRAINING_DATA)) {
        printf("✅ Found (%s)              │\n", DEFAULT_TRAINING_DATA);
    } else {
        printf("❌ Not found (%s)         │\n", DEFAULT_TRAINING_DATA);
    }
    
    printf("├──────────────────────────────────────────────────────────────┤\n");
    printf("│                                                              │\n");
    
    if (file_exists(DEFAULT_WEIGHTS_FILE)) {
        printf("│ 🎉 Model is ready for text generation!                     │\n");
    } else {
        printf("│ 🔧 Model needs training before generation is possible.      │\n");
    }
    
    if (!file_exists(DEFAULT_TRAINING_DATA)) {
        printf("│ 📝 Place your training text in '%s' to begin.   │\n", DEFAULT_TRAINING_DATA);
    }
    
    printf("│                                                              │\n");
    printf("└──────────────────────────────────────────────────────────────┘\n");
    
    printf("\nPress any key to return to main menu...");
    get_char();
}

/**
 * Show help and documentation
 */
static void show_help(void) {
    clear_screen();
    display_banner();
    
    printf("┌──────────────────────────────────────────────────────────────┐\n");
    printf("│                  HELP & DOCUMENTATION                       │\n");
    printf("├──────────────────────────────────────────────────────────────┤\n");
    printf("│                                                              │\n");
    printf("│ 🚀 GETTING STARTED:                                         │\n");
    printf("│   1. Place your training text in '%s'         │\n", DEFAULT_TRAINING_DATA);
    printf("│   2. Choose option 1 (Quick Train) for best results         │\n");
    printf("│   3. Use option 3 to generate text with your model          │\n");
    printf("│                                                              │\n");
    printf("│ 📝 TRAINING DATA FORMAT:                                    │\n");
    printf("│   • Plain text file, one sentence per line                  │\n");
    printf("│   • UTF-8 encoding recommended                              │\n");
    printf("│   • More data = better results (minimum 1000 lines)         │\n");
    printf("│                                                              │\n");
    printf("│ 🎯 MODEL SETTINGS GUIDE:                                    │\n");
    printf("│   • Larger d_model = more accuracy, slower training         │\n");
    printf("│   • More layers = better understanding, more memory         │\n");
    printf("│   • Higher seq_len = longer context, more computation       │\n");
    printf("│   • Lower learning rate = more stable, slower training      │\n");
    printf("│                                                              │\n");
    printf("│ 🔧 FILE LOCATIONS:                                          │\n");
    printf("│   • Model weights: %s                         │\n", DEFAULT_WEIGHTS_FILE);
    printf("│   • Vocabulary: %s                      │\n", DEFAULT_VOCAB_FILE);
    printf("│   • Training data: %s                    │\n", DEFAULT_TRAINING_DATA);
    printf("│                                                              │\n");
    printf("│ ⚡ PERFORMANCE TIPS:                                        │\n");
    printf("│   • Use Quick Train for balanced speed/quality              │\n");
    printf("│   • Reduce batch_size if you get memory errors              │\n");
    printf("│   • More epochs = better quality but longer training        │\n");
    printf("│                                                              │\n");
    printf("└──────────────────────────────────────────────────────────────┘\n");
    
    printf("\nPress any key to return to main menu...");
    get_char();
}

/**
 * Data preprocessing tools menu
 */
static void data_preprocessing_tools(void) {
    clear_screen();
    display_banner();
    
    printf("┌──────────────────────────────────────────────────────────────┐\n");
    printf("│                DATA PREPROCESSING TOOLS                     │\n");
    printf("├──────────────────────────────────────────────────────────────┤\n");
    printf("│                                                              │\n");
    printf("│ [1] 📊 Process CSV file to training text                    │\n");
    printf("│ [2] 🧹 Clean and format existing text file                  │\n");
    printf("│ [3] 📈 Analyze training data statistics                     │\n");
    printf("│ [4] 🔄 Convert between file formats                         │\n");
    printf("│ [0] ⬅️  Back to main menu                                   │\n");
    printf("│                                                              │\n");
    printf("└──────────────────────────────────────────────────────────────┘\n");
    
    printf("\nSelect preprocessing tool (0-4): ");
    int choice = get_char();
    
    switch (choice) {
        case '1': {
            printf("\n\nCSV Processing:\n");
            char csv_file[256];
            get_string_input("Enter CSV file path: ", csv_file, sizeof(csv_file));
            
            if (strlen(csv_file) > 0 && file_exists(csv_file)) {
                char command[512];
                snprintf(command, sizeof(command), "preprocess_csv %s --output %s", 
                        csv_file, DEFAULT_TRAINING_DATA);
                
                printf("Processing CSV file...\n");
                int result = system(command);
                
                if (result == 0) {
                    printf("✅ CSV processed successfully! Output: %s\n", DEFAULT_TRAINING_DATA);
                } else {
                    printf("❌ CSV processing failed.\n");
                }
            } else {
                printf("❌ File not found or invalid path.\n");
            }
            
            printf("\nPress any key to continue...");
            get_char();
            break;
        }
        case '2':
            printf("\n\n🧹 Text cleaning feature coming soon!\n");
            printf("Press any key to continue...");
            get_char();
            break;
        case '3':
            printf("\n\n📈 Statistics feature coming soon!\n");
            printf("Press any key to continue...");
            get_char();
            break;
        case '4':
            printf("\n\n🔄 Format conversion feature coming soon!\n");
            printf("Press any key to continue...");
            get_char();
            break;
        case '0':
            return;
        default:
            printf("\n❌ Invalid choice.\n");
            printf("Press any key to continue...");
            get_char();
            break;
    }
}

/**
 * Settings and configuration menu
 */
static void settings_menu(void) {
    clear_screen();
    display_banner();
    
    printf("┌──────────────────────────────────────────────────────────────┐\n");
    printf("│                SETTINGS & CONFIGURATION                     │\n");
    printf("├──────────────────────────────────────────────────────────────┤\n");
    printf("│                                                              │\n");
    printf("│ Current file locations:                                      │\n");
    printf("│ • Vocabulary: %s                           │\n", DEFAULT_VOCAB_FILE);
    printf("│ • Model weights: %s                        │\n", DEFAULT_WEIGHTS_FILE);
    printf("│ • Training data: %s                       │\n", DEFAULT_TRAINING_DATA);
    printf("│                                                              │\n");
    printf("│ [1] 🗑️  Reset/Clear model files                             │\n");
    printf("│ [2] 💾 Backup model files                                   │\n");
    printf("│ [3] 📋 View system information                              │\n");
    printf("│ [0] ⬅️  Back to main menu                                   │\n");
    printf("│                                                              │\n");
    printf("└──────────────────────────────────────────────────────────────┘\n");
    
    printf("\nSelect option (0-3): ");
    int choice = get_char();
    
    switch (choice) {
        case '1': {
            printf("\n\n⚠️  WARNING: This will delete your trained model!\n");
            printf("Are you sure? (y/N): ");
            int confirm = get_char();
            if (confirm == 'y' || confirm == 'Y') {
                remove(DEFAULT_VOCAB_FILE);
                remove(DEFAULT_WEIGHTS_FILE);
                printf("\n✅ Model files cleared.\n");
            } else {
                printf("\n❌ Operation cancelled.\n");
            }
            printf("Press any key to continue...");
            get_char();
            break;
        }
        case '2':
            printf("\n\n💾 Backup feature coming soon!\n");
            printf("Press any key to continue...");
            get_char();
            break;
        case '3': {
            printf("\n\nSystem Information:\n");
            printf("───────────────────\n");
            #ifdef _WIN32
            printf("Platform: Windows\n");
            #else
            printf("Platform: Unix/Linux\n");
            #endif
            printf("Executable: %s\n", CLI_EXECUTABLE_NAME);
            printf("Working directory: %s\n", ".");
            printf("\nPress any key to continue...");
            get_char();
            break;
        }
        case '0':
            return;
        default:
            printf("\n❌ Invalid choice.\n");
            printf("Press any key to continue...");
            get_char();
            break;
    }
}

/**
 * Handle advanced training with custom parameters
 */
static void handle_advanced_train(void) {
    clear_screen();
    display_banner();
    
    printf("┌──────────────────────────────────────────────────────────────┐\n");
    printf("│            🎯 ADVANCED TRAINING - CUSTOM PARAMETERS          │\n");
    printf("├──────────────────────────────────────────────────────────────┤\n");
    printf("│  Configure training parameters for your specific needs      │\n");
    printf("└──────────────────────────────────────────────────────────────┘\n");
    printf("\n");
    
    char training_file[256];
    char vocab_file[256];
    char weights_file[256];
    
    // Get training file
    get_string_input("Training data file path: ", training_file, sizeof(training_file));
    if (!file_exists(training_file)) {
        printf("❌ File not found: %s\n", training_file);
        printf("Press any key to return...");
        get_char();
        return;
    }
    
    // Get vocabulary and weights paths with defaults
    printf("\nFile Paths (press Enter for defaults):\n");
    get_string_input("Vocabulary file [aryan_vocab.vocab]: ", vocab_file, sizeof(vocab_file));
    if (strlen(vocab_file) == 0) strcpy(vocab_file, DEFAULT_VOCAB_FILE);
    
    get_string_input("Weights save file [aryan_model.bin]: ", weights_file, sizeof(weights_file));
    if (strlen(weights_file) == 0) strcpy(weights_file, DEFAULT_WEIGHTS_FILE);
    
    // Get training parameters with enhanced accuracy defaults
    printf("\n🔧 Training Parameters (enhanced for accuracy):\n");
    
    float learning_rate = get_float_input("Learning rate", 1e-6f, 1e-2f, 1e-4f);
    int epochs = get_int_input("Number of epochs", 1, 500, 25);
    int batch_size = get_int_input("Batch size", 1, 32, 8);
    int seq_len = get_int_input("Sequence length", 32, 1024, 256);
    int d_model = get_int_input("Model dimension", 64, 512, 256);
    int n_layers = get_int_input("Number of layers", 2, 12, 6);
    int n_heads = get_int_input("Number of attention heads", 2, 16, 8);
    int grad_accum = get_int_input("Gradient accumulation steps", 1, 8, 2);
    
    // Build command
    char command[2048];
    snprintf(command, sizeof(command),
        "%s --mode train "
        "--data \"%s\" "
        "--vocab_load \"%s\" "
        "--vocab_save \"%s\" "
        "--weights_save \"%s\" "
        "--lr %.6f "
        "--epochs %d "
        "--batch_size %d "
        "--seq_len %d "
        "--d_model %d "
        "--n_layers %d "
        "--n_heads %d "
        "--grad_accum %d",
        CLI_EXECUTABLE_NAME,
        training_file, vocab_file, vocab_file, weights_file,
        learning_rate, epochs, batch_size, seq_len,
        d_model, n_layers, n_heads, grad_accum
    );
    
    printf("\n📋 Training Summary:\n");
    printf("  Training File: %s\n", training_file);
    printf("  Vocabulary: %s\n", vocab_file);
    printf("  Weights: %s\n", weights_file);
    printf("  Learning Rate: %.6f\n", learning_rate);
    printf("  Epochs: %d\n", epochs);
    printf("  Enhanced Accuracy: ENABLED\n");
    
    printf("\nPress 'y' to start training, any other key to cancel: ");
    int choice = get_char();
    
    if (choice == 'y' || choice == 'Y') {
        printf("\n\n🚀 Starting Advanced Training...\n");
        printf("Command: %s\n\n", command);
        
        int result = system(command);
        
        if (result == 0) {
            printf("\n✅ Advanced training completed successfully!\n");
        } else {
            printf("\n❌ Training failed with error code: %d\n", result);
        }
    } else {
        printf("\n\nTraining cancelled.\n");
    }
    
    printf("\nPress any key to continue...");
    get_char();
}

/**
 * Handle text generation with trained model
 */
static void handle_generate_text(void) {
    clear_screen();
    display_banner();
    
    printf("┌──────────────────────────────────────────────────────────────┐\n");
    printf("│                  💬 TEXT GENERATION                          │\n");
    printf("├──────────────────────────────────────────────────────────────┤\n");
    printf("│  Generate creative text using your trained model            │\n");
    printf("└──────────────────────────────────────────────────────────────┘\n");
    printf("\n");
    
    // Check if model files exist
    if (!file_exists(DEFAULT_WEIGHTS_FILE)) {
        printf("❌ No trained model found (%s)!\n", DEFAULT_WEIGHTS_FILE);
        printf("Please train a model first using option 1 or 2.\n");
        printf("\nPress any key to return...");
        get_char();
        return;
    }
    
    if (!file_exists(DEFAULT_VOCAB_FILE)) {
        printf("❌ No vocabulary file found (%s)!\n", DEFAULT_VOCAB_FILE);
        printf("Please train a model first to generate the vocabulary.\n");
        printf("\nPress any key to return...");
        get_char();
        return;
    }
    
    char prompt[512];
    printf("Enter your text prompt (what should the AI continue?):\n");
    get_string_input("Prompt: ", prompt, sizeof(prompt));
    
    if (strlen(prompt) == 0) {
        strcpy(prompt, "The future of artificial intelligence");
        printf("Using default prompt: %s\n", prompt);
    }
    
    int max_tokens = get_int_input("Maximum tokens to generate", 10, 500, 100);
    
    printf("\n🎯 Generation Settings:\n");
    printf("  • Model: %s\n", DEFAULT_WEIGHTS_FILE);
    printf("  • Vocabulary: %s\n", DEFAULT_VOCAB_FILE);
    printf("  • Prompt: \"%s\"\n", prompt);
    printf("  • Max Tokens: %d\n", max_tokens);
    
    printf("\nPress 'y' to generate, any other key to cancel: ");
    int choice = get_char();
    
    if (choice == 'y' || choice == 'Y') {
        printf("\n\n💫 Generating text...\n");
        printf("══════════════════════════════════════════════════════════════\n");
        
        char command[1024];
        snprintf(command, sizeof(command),
            "%s --mode generate "
            "--weights_load \"%s\" "
            "--vocab_load \"%s\" "
            "--prompt \"%s\" "
            "--max_new %d",
            CLI_EXECUTABLE_NAME, DEFAULT_WEIGHTS_FILE, 
            DEFAULT_VOCAB_FILE, prompt, max_tokens);
        
        int result = system(command);
        
        printf("══════════════════════════════════════════════════════════════\n");
        
        if (result == 0) {
            printf("✅ Text generation completed!\n");
        } else {
            printf("❌ Generation failed with error code: %d\n", result);
        }
    } else {
        printf("\n\nGeneration cancelled.\n");
    }
    
    printf("\nPress any key to continue...");
    get_char();
}

/**
 * Main interactive CLI loop
 */
void run_interactive_cli(void) {
    int running = 1;
    
    while (running) {
        clear_screen();
        display_banner();
        display_main_menu();
        
        int choice = get_char();
        
        switch (choice) {
            case '1':
                handle_quick_train();
                break;
            case '2':
                handle_advanced_train();
                break;
            case '3':
                handle_generate_text();
                break;
            case '4':
                show_model_status();
                break;
            case '5':
                settings_menu();
                break;
            case '6':
                show_help();
                break;
            case '7':
                data_preprocessing_tools();
                break;
            case '0':
                printf("\n\nThank you for using Aryan Alpha AI! 🧠✨\n");
                running = 0;
                break;
            default:
                printf("\n❌ Invalid choice. Please press a number between 0-7.\n");
                printf("Press any key to continue...");
                get_char();
                break;
        }
    }
}
