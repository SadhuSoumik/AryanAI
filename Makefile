# ====================================================================
# CLEAN C TRANSFORMER - SIMPLIFIED MAKEFILE  
# ====================================================================
# A simple, cross-platform Makefile without unnecessary complexity

CC = gcc

# Basic compiler flags
CFLAGS = -O3 -Wall -Wextra -std=c11 -Iinclude
LDFLAGS = -lm

# Add OpenMP if available (optional)
OPENMP_AVAILABLE := $(shell echo "" | $(CC) -fopenmp -E - >/dev/null 2>&1 && echo yes || echo no)
ifeq ($(OPENMP_AVAILABLE),yes)
    CFLAGS += -fopenmp
    LDFLAGS += -fopenmp
endif

# Platform detection and release structure
UNAME_S := $(shell uname -s 2>/dev/null || echo "Windows")
ifeq ($(UNAME_S),Linux)
    PLATFORM = linux-x64
    EXECUTABLE_SUFFIX = 
endif
ifeq ($(UNAME_S),Darwin)
    PLATFORM = macos-x64
    EXECUTABLE_SUFFIX = 
endif
ifeq ($(UNAME_S),Windows)
    PLATFORM = windows-x64
    EXECUTABLE_SUFFIX = .exe
    LDFLAGS += -static-libgcc
endif

# Release directories (single location for all builds)
RELEASE_DIR = model/releases/$(PLATFORM)
BIN_DIR = $(RELEASE_DIR)/bin
DATA_DIR = $(RELEASE_DIR)/data
MODELS_DIR = $(RELEASE_DIR)/models
DOCS_DIR = $(RELEASE_DIR)/docs
OBJ_DIR = model/builds/$(PLATFORM)-obj

# Source files
SOURCES = main.c src/matrix.c src/optimizer.c src/platform.c src/tokenizer.c src/transformer_model.c src/utils.c src/interactive_cli.c
# Place object files in deployment obj directory
OBJECTS = $(patsubst %.c,$(OBJ_DIR)/%.o,$(SOURCES))

# Target executables in deployment directory
TARGET = $(BIN_DIR)/Aaryan$(EXECUTABLE_SUFFIX)
PREPROCESS_TARGET = $(BIN_DIR)/preprocess_csv$(EXECUTABLE_SUFFIX)

# Default target
all: deploy_structure $(TARGET) $(PREPROCESS_TARGET) copy_files

# Create deployment directory structure
deploy_structure:
	@mkdir -p $(BIN_DIR) $(DATA_DIR) $(MODELS_DIR) $(DOCS_DIR) $(OBJ_DIR) $(OBJ_DIR)/src
	@echo "Created deployment structure for $(PLATFORM)"

# Build main executable
$(TARGET): $(OBJECTS) | deploy_structure
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS)
	@echo "Built $(TARGET)"

# Build preprocessor utility
$(PREPROCESS_TARGET): preprocess_csv.c | deploy_structure
	$(CC) $(CFLAGS) preprocess_csv.c -o $@ $(LDFLAGS)
	@echo "Built $(PREPROCESS_TARGET)"

# Build object files - place them in deployment obj directory
$(OBJ_DIR)/%.o: %.c | deploy_structure
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

# Copy essential files to deployment directory
copy_files: deploy_structure
	@cp -r data/* $(DATA_DIR)/ 2>/dev/null || true
	@cp README.md $(DEPLOY_DIR)/ 2>/dev/null || true
	@cp *.bin $(MODELS_DIR)/ 2>/dev/null || echo "No model files to copy yet"
	@cp *.vocab $(MODELS_DIR)/ 2>/dev/null || echo "No vocab files to copy yet"
	@echo "Copied essential files to $(DEPLOY_DIR)"

# Clean build artifacts but keep deployment
clean:
	rm -rf $(OBJ_DIR)
	rm -f *.o src/*.o  # Remove any stray object files from old builds
	@echo "Cleaned build artifacts (kept deployment directory)"

# Clean everything including deployment
clean-all: clean
	rm -rf deploy/
	@echo "Cleaned everything including deployment directories"

# Install to system (Unix systems only)
install: $(TARGET)
ifeq ($(PLATFORM),linux-x64)
	sudo cp $(TARGET) /usr/local/bin/Aaryan
	sudo cp $(PREPROCESS_TARGET) /usr/local/bin/preprocess_csv
endif

.PHONY: all deploy_structure copy_files clean clean-all install
