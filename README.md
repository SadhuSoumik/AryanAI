# AryanAi - High-Performance C Transformer Model

A lightweight, cross-platform transformer model implementation written in pure C, designed for efficient training and text generation without external dependencies.

## ✨ Features

- **Pure C Implementation**: No external machine learning libraries required
- **Cross-Platform**: Supports Linux and Windows (x64 architectures)
- **Memory Efficient**: Optimized for low-memory environments
- **Fast Training**: Multi-threaded training with configurable parameters
- **Portable**: Self-contained executables with no runtime dependencies
- **Flexible Input**: Support for CSV and text file formats
- **Complete Implementation**: All transformer components fully implemented
- **Streamlined Build System**: Consolidated build scripts and organized output structure

## 📁 Project Structure

```
AryanAi/
├── 📂 src/                          # Core source code
│   ├── interactive_cli.c            # Interactive command-line interface
│   ├── matrix.c                     # Matrix operations and linear algebra
│   ├── optimizer.c                  # Training optimizers (Adam, SGD)
│   ├── platform.c                   # Cross-platform compatibility layer
│   ├── tokenizer.c                  # Text tokenization and vocabulary management
│   ├── transformer_model.c          # Transformer model architecture
│   └── utils.c                      # Utility functions and helpers
│
├── 📂 include/                      # Header files
│   ├── interactive_cli.h            # CLI interface definitions
│   ├── matrix.h                     # Matrix operations API
│   ├── optimizer.h                  # Optimizer interfaces
│   ├── platform.h                   # Cross-platform compatibility
│   ├── tokenizer.h                  # Tokenizer interface
│   ├── transformer_config.h         # Model configuration structures
│   ├── transformer_model.h          # Model architecture definitions
│   └── utils.h                      # Utility function declarations
│
├── 📂 data/                         # Training data and datasets
│   ├── test_training_data.txt       # Sample training text
│   └── test_data.txt                # Additional test data
│
├── 📂 model/                        # Model builds and releases
│   ├── builds/                      # Temporary build artifacts (auto-cleaned)
│   └── releases/                    # Platform-specific release packages
│       ├── linux-x64/               # Linux x64 distribution
│       └── windows-x64/             # Windows x64 distribution
│
├── 📂 scripts/                      # Build and utility scripts
│   ├── benchmark_test.sh            # Performance benchmarking suite
│   ├── build-linux.sh              # Linux native build script
│   ├── build-macos.sh              # macOS build script (planned)
│   ├── build-windows-cross.sh      # Windows cross-compilation script
│   ├── build-windows.bat           # Windows native build script
│   ├── create-windows-release.sh   # Windows release packaging
│   └── test_model.sh               # Model testing utilities
│
├── 📂 docs/                         # Documentation
│   ├── BEGINNERS_GUIDE.md          # Getting started guide
│   ├── BUILD_ORGANIZATION.md       # Build system documentation
│   └── CROSS_PLATFORM_GUIDE.md     # Cross-platform development guide
│
├── 📂 cmake/                        # CMake configuration
│   ├── BuildInstructions.txt.in    # Build instruction template
│   └── windows-toolchain.cmake     # Windows cross-compilation toolchain
│
├── 📂 benchmark_results/            # Performance test results and logs
│
├── main.c                           # Main application entry point
├── preprocess_csv.c                 # CSV preprocessing utility
├── CMakeLists.txt                   # CMake build configuration
├── Makefile                         # GNU Make build file (alternative)
└── README.md                        # This documentation
```

## 🛠️ Building from Source

### Prerequisites

**Linux:**

- GCC compiler (4.9 or later)
- Make or CMake
- POSIX threads support

**Windows Cross-Compilation (on Linux):**

- MinGW-w64 cross-compiler
- Make or CMake

**Windows Native Build:**

- Visual Studio or MinGW
- Windows SDK

### Installation

#### Linux Native Build

```bash
# Clone the repository
git clone https://github.com/your-username/AryanAi.git
cd AryanAi

# Recommended: Using build script (outputs to model/releases/linux-x64/)
chmod +x scripts/build-linux.sh
./scripts/build-linux.sh

# Alternative: Using CMake directly
mkdir build && cd build
cmake .. && make
cd ..

# Alternative: Using Make directly
make clean && make
```

#### Windows Cross-Compilation (from Linux)

```bash
# Install MinGW-w64 (Ubuntu/Debian)
sudo apt update
sudo apt install mingw-w64

# Build Windows executables (outputs to model/releases/windows-x64/)
chmod +x scripts/build-windows-cross.sh
./scripts/build-windows-cross.sh
```

#### Windows Native Build

```cmd
REM Using the provided batch script
scripts\build-windows.bat

REM Or using CMake
mkdir build && cd build
cmake .. -G "Visual Studio 16 2019" -A x64
cmake --build . --config Release
```

### Build Output Locations

After successful compilation:

- **Linux executables**: `model/releases/linux-x64/bin/`
- **Windows executables**: `model/releases/windows-x64/bin/`
- **Temporary build files**: `model/builds/` (automatically cleaned)

## 🎯 Quick Start

### Using Pre-built Releases

#### Linux

```bash
cd model/releases/linux-x64/
./train.sh      # Train the model with default parameters
./generate.sh   # Generate text with trained model
```

#### Windows

```cmd
cd model\releases\windows-x64\
train.bat       # Train the model with default parameters
generate.bat    # Generate text with trained model
```

### Interactive CLI Mode

Start the interactive command-line interface:

```bash
# Linux
cd model/releases/linux-x64/
./bin/Aaryan

# Windows
cd model\releases\windows-x64\
bin\AryanAi.exe
```

## 📊 Training & Generation

### Data Preparation

**CSV Format Example:**

```csv
Human,AI
"Hello, how are you?","I'm doing well, thank you for asking!"
"What's the weather like?","I don't have access to current weather data."
```

**Text Format Example:**

```
The quick brown fox jumps over the lazy dog.
This is a sample training sentence.
Add your training data here.
```

### Training Parameters

| Parameter      | Description         | Default | Recommended Range |
| -------------- | ------------------- | ------- | ----------------- |
| `--d_model`    | Model dimensions    | 256     | 64-512            |
| `--n_heads`    | Attention heads     | 8       | 2-16              |
| `--n_layers`   | Transformer layers  | 6       | 1-12              |
| `--epochs`     | Training iterations | 10      | 1-500             |
| `--lr`         | Learning rate       | 1e-4    | 1e-6 to 1e-2      |
| `--batch_size` | Batch size          | 8       | 1-32              |
| `--seq_len`    | Sequence length     | 128     | 32-1024           |

### Basic Usage Examples

#### Training a Model

```bash
# Linux
cd model/releases/linux-x64/
./bin/Aaryan --mode train \
  --data data/test_training_data.txt \
  --epochs 20 \
  --d_model 256 \
  --n_layers 6 \
  --weights_save models/my_model.bin \
  --vocab_save models/my_vocab.vocab

# Windows
cd model\releases\windows-x64\
bin\AryanAi.exe --mode train ^
  --data data\test_training_data.txt ^
  --epochs 20 ^
  --d_model 256 ^
  --n_layers 6 ^
  --weights_save models\my_model.bin ^
  --vocab_save models\my_vocab.vocab
```

#### Text Generation

```bash
# Linux
./bin/Aaryan --mode generate \
  --weights_load models/my_model.bin \
  --vocab_load models/my_vocab.vocab \
  --prompt "The quick brown fox" \
  --max_new 50

# Windows
bin\AryanAi.exe --mode generate ^
  --weights_load models\my_model.bin ^
  --vocab_load models\my_vocab.vocab ^
  --prompt "The quick brown fox" ^
  --max_new 50
```

#### CSV Preprocessing

```bash
# Linux
./bin/preprocess_csv input.csv output.txt

# Windows
bin\preprocess_csv.exe input.csv output.txt
```

## ✅ Implementation Status

**Current Version**: Feature-complete transformer implementation

### Core Components (All Implemented)

- ✅ **Transformer Architecture**: Full forward and backward pass
- ✅ **Multi-Head Attention**: Complete attention mechanism with gradients
- ✅ **Feed-Forward Networks**: FFN layers with backpropagation
- ✅ **Layer Normalization**: Forward and backward pass
- ✅ **Embedding Layers**: Token and positional embeddings
- ✅ **Adam Optimizer**: Gradient-based optimization with momentum
- ✅ **Text Generation**: Autoregressive generation with proper spacing
- ✅ **Data Processing**: Unlimited CSV file processing
- ✅ **Memory Management**: Efficient matrix operations and cleanup

### Recent Improvements (June 2025)

- ✅ **Enhanced Text Generation**: Fixed spacing issues in output
- ✅ **Unlimited CSV Processing**: Removed 10,000 line limit
- ✅ **Cleaner Console Output**: Removed optimization suggestions
- ✅ **Improved Build System**: Consolidated output structure
- ✅ **Cross-Platform Compatibility**: Enhanced Windows support

## 📈 Performance Benchmarks

### Typical Performance (Linux x64)

| Configuration                  | Training Time\* | Memory Usage | Model Size |
| ------------------------------ | --------------- | ------------ | ---------- |
| Small (vocab: 500, dim: 64)    | ~5 minutes      | ~100MB       | ~2MB       |
| Medium (vocab: 1000, dim: 128) | ~15 minutes     | ~200MB       | ~8MB       |
| Large (vocab: 2000, dim: 256)  | ~45 minutes     | ~500MB       | ~32MB      |

\*Approximate times on modern multi-core CPU with sample dataset

### Running Benchmarks

```bash
# Run performance benchmarks
./scripts/benchmark_test.sh

# View existing results
ls benchmark_results/
```

## 🧠 Model Parameters, Size & Capabilities

### Model Architecture Details

The AryanAi transformer implements a complete attention-based architecture with configurable parameters for different use cases and computational constraints.

#### Core Architecture Components

| Component                | Description                     | Configurable Parameters          |
| ------------------------ | ------------------------------- | -------------------------------- |
| **Embedding Layer**      | Token and positional embeddings | `vocab_size`, `d_model`          |
| **Multi-Head Attention** | Self-attention mechanism        | `n_heads`, `d_model`             |
| **Feed-Forward Network** | Position-wise dense layers      | `d_model`, `d_ff` (4×d_model)    |
| **Layer Normalization**  | Stabilizes training             | Applied before attention and FFN |
| **Transformer Blocks**   | Complete encoder layers         | `n_layers`                       |

### Model Configurations & Capabilities

#### Lightweight Configuration

```bash
# Minimal resource usage - Good for learning and testing
--d_model 64 --n_layers 2 --n_heads 4 --vocab_size 500 --seq_len 64
```

- **Model Size**: ~0.5MB
- **Memory Usage**: ~50MB during training
- **Training Time**: 2-5 minutes on modern CPU
- **Capabilities**: Basic text completion, simple patterns
- **Use Cases**: Educational purposes, rapid prototyping, embedded systems

#### Small Configuration (Default)

```bash
# Balanced performance - Recommended for most users
--d_model 128 --n_layers 4 --n_heads 8 --vocab_size 1000 --seq_len 128
```

- **Model Size**: ~2-4MB
- **Memory Usage**: ~100-150MB during training
- **Training Time**: 5-15 minutes on modern CPU
- **Capabilities**: Good text generation, handles basic conversations
- **Use Cases**: Personal projects, small applications, learning transformers

#### Medium Configuration

```bash
# Higher quality - Good balance of quality and resources
--d_model 256 --n_layers 6 --n_heads 8 --vocab_size 2000 --seq_len 256
```

- **Model Size**: ~8-16MB
- **Memory Usage**: ~200-400MB during training
- **Training Time**: 15-45 minutes on modern CPU
- **Capabilities**: Coherent text generation, better context understanding
- **Use Cases**: Content generation, chatbots, research projects

#### Large Configuration

```bash
# Maximum quality - Requires significant resources
--d_model 512 --n_layers 8 --n_heads 16 --vocab_size 5000 --seq_len 512
```

- **Model Size**: ~32-64MB
- **Memory Usage**: ~500MB-1GB during training
- **Training Time**: 1-3 hours on modern CPU
- **Capabilities**: High-quality text generation, complex pattern recognition
- **Use Cases**: Production applications, research, complex language tasks

### Parameter Impact Analysis

#### Model Dimension (`d_model`)

- **64**: Basic embeddings, simple patterns
- **128**: Standard quality for most applications
- **256**: Good quality text generation
- **512**: High-quality, nuanced understanding
- **Impact**: Quadratic effect on model size and memory usage

#### Number of Layers (`n_layers`)

- **2**: Minimal depth, basic transformations
- **4**: Standard depth for small models
- **6**: Good depth for quality generation
- **8+**: Deep understanding, complex patterns
- **Impact**: Linear effect on model size, significant impact on training time

#### Attention Heads (`n_heads`)

- **4**: Basic attention patterns
- **8**: Standard multi-head attention
- **16**: Rich attention mechanisms
- **Impact**: Must divide evenly into `d_model`

#### Vocabulary Size (`vocab_size`)

- **500**: Basic vocabulary, limited expression
- **1000**: Good coverage for simple tasks
- **2000**: Comprehensive vocabulary for most text
- **5000+**: Extensive vocabulary, handles complex text
- **Impact**: Linear effect on embedding layer size

#### Sequence Length (`seq_len`)

- **64**: Short context, basic completion
- **128**: Standard context window
- **256**: Good context for conversations
- **512+**: Long context, complex dependencies
- **Impact**: Quadratic effect on attention computation

### Memory Requirements by Configuration

| Configuration | Model Size | Training RAM | Generation RAM | GPU Equivalent      |
| ------------- | ---------- | ------------ | -------------- | ------------------- |
| Lightweight   | 0.5MB      | 50MB         | 20MB           | ~1M parameters      |
| Small         | 2-4MB      | 100-150MB    | 50MB           | ~2-5M parameters    |
| Medium        | 8-16MB     | 200-400MB    | 100MB          | ~10-20M parameters  |
| Large         | 32-64MB    | 500MB-1GB    | 200MB          | ~50-100M parameters |

### Capability Matrix

| Feature                   | Lightweight | Small      | Medium       | Large        |
| ------------------------- | ----------- | ---------- | ------------ | ------------ |
| **Text Completion**       | ✅ Basic    | ✅ Good    | ✅ Excellent | ✅ Superior  |
| **Conversation**          | ❌ Limited  | ✅ Basic   | ✅ Good      | ✅ Excellent |
| **Code Generation**       | ❌ No       | ❌ Limited | ✅ Basic     | ✅ Good      |
| **Creative Writing**      | ❌ No       | ❌ Limited | ✅ Good      | ✅ Excellent |
| **Context Understanding** | ❌ Minimal  | ✅ Basic   | ✅ Good      | ✅ Strong    |
| **Multi-turn Dialogue**   | ❌ No       | ❌ Limited | ✅ Good      | ✅ Excellent |

### Training Data Requirements

#### Minimum Dataset Sizes

- **Lightweight**: 1,000+ sentences (~50KB text)
- **Small**: 5,000+ sentences (~200KB text)
- **Medium**: 20,000+ sentences (~1MB text)
- **Large**: 100,000+ sentences (~5MB+ text)

#### Data Quality Guidelines

- **Diversity**: Include varied sentence structures and topics
- **Length**: Mix of short and long sentences for robust learning
- **Quality**: Clean, grammatically correct text for better results
- **Format**: Consistent formatting (UTF-8 plain text or CSV)

### Performance vs Resource Trade-offs

#### Training Time Scaling

```
Configuration   | CPU Cores | Training Time (1000 sentences)
Lightweight     | 2-4       | 2-5 minutes
Small          | 2-4       | 5-15 minutes
Medium         | 4-8       | 15-45 minutes
Large          | 8+        | 1-3 hours
```

#### Quality Metrics (Subjective)

- **Coherence**: How well the model maintains topic consistency
- **Fluency**: Natural language flow and grammar
- **Creativity**: Ability to generate novel, interesting content
- **Context**: Understanding and maintaining conversation context

### Optimization Recommendations

#### For Limited Resources (<2GB RAM)

```bash
# Ultra-lightweight configuration
--d_model 32 --n_layers 2 --n_heads 2 --vocab_size 300 --seq_len 32 --batch_size 2
```

#### For Standard Desktops (4-8GB RAM)

```bash
# Recommended balanced configuration
--d_model 128 --n_layers 4 --n_heads 8 --vocab_size 1500 --seq_len 128 --batch_size 8
```

#### For High-Performance Systems (16GB+ RAM)

```bash
# Maximum quality configuration
--d_model 384 --n_layers 8 --n_heads 12 --vocab_size 3000 --seq_len 384 --batch_size 16
```

### Model File Structure

When training completes, the following files are generated:

```
models/
├── model_name.bin           # Model weights (~90% of total size)
│   ├── Embedding weights   # vocab_size × d_model
│   ├── Attention weights   # n_layers × (4 × d_model²)
│   ├── FFN weights         # n_layers × (8 × d_model²)
│   └── Layer norm params   # n_layers × (4 × d_model)
├── vocab_name.vocab         # Vocabulary mappings (~10% of total size)
└── config.json             # Model configuration (optional)
```

---

**Note**: This is a learning-oriented implementation focused on understanding transformer architecture rather than production-scale performance.
