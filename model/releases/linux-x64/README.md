# AryanAlpha - High-Performance C Transformer Model

A lightweight, cross-platform transformer model implementation written in pure C, designed for efficient training and text generation without external dependencies.

## 🚀 Features

- **Pure C Implementation**: No external machine learning libraries required
- **Cross-Platform**: Supports Linux and Windows (x64 architectures)
- **Memory Efficient**: Optimized for low-memory environments
- **Fast Training**: Multi-threaded training with configurable parameters
- **Portable**: Self-contained executables with no runtime dependencies
- **Flexible Input**: Support for CSV and text file formats

## 📁 Project Structure

```
AryanAlpha/
├── 📂 src/                     # Core source code
│   ├── matrix.c               # Matrix operations and linear algebra
│   ├── tokenizer.c            # Text tokenization and vocabulary management
│   ├── transformer.c          # Transformer model architecture
│   ├── training.c             # Training algorithms and backpropagation
│   └── text_generation.c     # Text generation and inference
├── 📂 include/                # Header files
│   ├── matrix.h              # Matrix operations API
│   ├── tokenizer.h           # Tokenizer interface
│   ├── transformer.h         # Model architecture definitions
│   ├── training.h            # Training functions
│   ├── text_generation.h     # Generation utilities
│   └── platform.h            # Cross-platform compatibility
├── 📂 data/                   # Training data and datasets
│   ├── simple_conversations.csv  # Sample conversation dataset
│   └── training_data.txt         # Processed training text
├── 📂 models/                 # Trained model weights
│   ├── vocab.bin             # Vocabulary embeddings
│   ├── weights.bin           # Model parameters
│   └── model_metadata.bin    # Model configuration
├── 📂 scripts/               # Build and utility scripts
│   ├── build-linux.sh       # Linux native build
│   ├── build-windows-cross.sh   # Windows cross-compilation
│   ├── create-linux-release.sh  # Linux release packaging
│   └── create-windows-release.sh # Windows release packaging
├── 📂 release/               # Pre-built release packages
│   ├── linux-x64/           # Linux x64 distribution
│   │   ├── bin/             # Executables
│   │   ├── data/            # Sample data
│   │   ├── models/          # Pre-trained models
│   │   ├── train.sh         # Training script
│   │   ├── generate.sh      # Generation script
│   │   └── README.txt       # Platform-specific instructions
│   └── windows-x64/         # Windows x64 distribution
│       ├── bin/             # Windows executables (.exe)
│       ├── data/            # Sample data
│       ├── models/          # Pre-trained models
│       ├── train.bat        # Training batch script
│       ├── generate.bat     # Generation batch script
│       └── README_WINDOWS.txt # Windows-specific instructions
├── 📂 build/                 # Build artifacts (Linux)
├── 📂 build-windows-cross/   # Cross-compilation artifacts
├── main.c                   # Main application entry point
├── preprocess_csv.c         # CSV preprocessing utility
├── CMakeLists.txt           # CMake build configuration
├── Makefile                 # GNU Make build file
└── README.md               # This documentation
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

### Installation

#### Linux Native Build

```bash
# Clone the repository
git clone <repository-url>
cd AryanAlpha

# Option 1: Using Make
make clean && make

# Option 2: Using CMake
mkdir build && cd build
cmake .. && make

# Option 3: Using build script
chmod +x scripts/build-linux.sh
./scripts/build-linux.sh
```

#### Windows Cross-Compilation

```bash
# Install MinGW-w64 (Ubuntu/Debian)
sudo apt update
sudo apt install mingw-w64

# Build Windows executables
chmod +x scripts/build-windows-cross.sh
./scripts/build-windows-cross.sh
```

## 🎯 Quick Start

### Using Pre-built Releases

#### Linux
```bash
cd release/linux-x64/
./train.sh      # Train the model
./generate.sh   # Generate text
```

#### Windows
```batch
cd release\windows-x64\
train.bat       # Train the model
generate.bat    # Generate text
```

### Manual Usage

#### Training a Model
```bash
# Preprocess your data (if using CSV)
./preprocess_csv data/simple_conversations.csv data/training_data.txt

# Train the model
./c_transformer --train \
  --input data/training_data.txt \
  --vocab-size 1000 \
  --embed-dim 128 \
  --num-heads 4 \
  --num-layers 2 \
  --epochs 100 \
  --learning-rate 0.001
```

#### Text Generation
```bash
# Generate text using trained model
./c_transformer --generate \
  --prompt "The quick brown" \
  --max-tokens 50 \
  --temperature 0.8
```

## 📊 Training Instructions

### Data Preparation

1. **CSV Format**: Use the provided `simple_conversations.csv` as a template:
   ```csv
   Human,AI
   "Hello, how are you?","I'm doing well, thank you for asking!"
   "What's the weather like?","I don't have access to current weather data."
   ```

2. **Text Format**: Plain text files with sentences separated by newlines:
   ```
   The quick brown fox jumps over the lazy dog.
   This is a sample training sentence.
   Add your training data here.
   ```

### Training Parameters

| Parameter | Description | Default | Recommended Range |
|-----------|-------------|---------|-------------------|
| `--vocab-size` | Size of vocabulary | 1000 | 500-5000 |
| `--embed-dim` | Embedding dimensions | 128 | 64-512 |
| `--num-heads` | Attention heads | 4 | 2-8 |
| `--num-layers` | Transformer layers | 2 | 1-6 |
| `--epochs` | Training iterations | 100 | 50-500 |
| `--learning-rate` | Learning rate | 0.001 | 0.0001-0.01 |

### Cross-Platform Training

#### Linux Training
```bash
# Full training pipeline
cd release/linux-x64/
./train.sh

# Custom training
./bin/c_transformer --train \
  --input data/training_data.txt \
  --vocab-size 2000 \
  --embed-dim 256 \
  --epochs 200
```

#### Windows Training
```batch
REM Full training pipeline
cd release\windows-x64\
train.bat

REM Custom training
bin\c_transformer.exe --train ^
  --input data\training_data.txt ^
  --vocab-size 2000 ^
  --embed-dim 256 ^
  --epochs 200
```

## 🔧 Advanced Usage

### Custom Datasets

1. **Prepare your data** in CSV or text format
2. **Preprocess if needed**:
   ```bash
   ./preprocess_csv your_data.csv processed_data.txt
   ```
3. **Train with custom parameters**:
   ```bash
   ./c_transformer --train --input processed_data.txt [options]
   ```

### Performance Tuning

- **Memory Usage**: Reduce `--embed-dim` and `--vocab-size` for lower memory
- **Training Speed**: Decrease `--num-layers` and `--num-heads` for faster training
- **Model Quality**: Increase `--epochs` and `--embed-dim` for better results

### Model Files

After training, the following files are created in the `models/` directory:
- `vocab.bin`: Vocabulary embeddings and word mappings
- `weights.bin`: Model parameters (weights and biases)
- `model_metadata.bin`: Model configuration and hyperparameters

## 🔍 Troubleshooting

### Common Issues

1. **Compilation Errors**:
   - Ensure GCC/MinGW is properly installed
   - Check that all source files are present
   - Verify platform.h compatibility

2. **Training Failures**:
   - Check input data format
   - Verify sufficient disk space for model files
   - Ensure vocabulary size isn't too large for dataset

3. **Generation Issues**:
   - Verify model files exist in `models/` directory
   - Check that model was trained successfully
   - Ensure vocabulary is compatible

### Memory Requirements

- **Minimum**: 512MB RAM
- **Recommended**: 2GB+ RAM
- **Large Models**: 4GB+ RAM (for vocab-size > 5000)

### Platform-Specific Notes

**Linux**:
- Requires POSIX thread support
- Uses argp for command-line parsing
- Native compilation recommended

**Windows**:
- Cross-compiled using MinGW-w64
- Custom argument parser (no argp dependency)
- May require Visual C++ Redistributable

## 📈 Performance Benchmarks

| Configuration | Training Time* | Memory Usage | Model Size |
|---------------|----------------|--------------|------------|
| Small (vocab: 500, dim: 64) | ~5 minutes | ~100MB | ~2MB |
| Medium (vocab: 1000, dim: 128) | ~15 minutes | ~200MB | ~8MB |
| Large (vocab: 2000, dim: 256) | ~45 minutes | ~500MB | ~32MB |

*Approximate times on modern multi-core CPU with sample dataset

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test across platforms
4. Commit your changes: `git commit -am 'Add feature'`
5. Push to the branch: `git push origin feature-name`
6. Submit a pull request

### Code Standards

- Follow C99 standard
- Maintain cross-platform compatibility
- Include appropriate error handling
- Document new functions and structures
- Test on both Linux and Windows (cross-compiled)

## 📄 License

This project is open-source and available under the MIT License.

## 🙏 Acknowledgments

- Built with pure C for maximum portability
- Inspired by modern transformer architectures
- Designed for educational and research purposes

## 📞 Support

For questions, issues, or contributions:
- Create an issue in the repository
- Check the troubleshooting section above
- Review the platform-specific README files in release directories

---

**Note**: This is a learning-oriented implementation focused on understanding transformer architecture rather than production-scale performance.