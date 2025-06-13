#!/bin/bash
# Windows cross-compilation build script using MinGW-w64 on Linux
# This script builds Windows executables from Linux

set -e  # Exit on any error

echo "=== C Transformer Windows Cross-Compilation Build Script ==="

# Check if MinGW-w64 is installed
if ! command -v x86_64-w64-mingw32-gcc >/dev/null 2>&1; then
    echo "Error: MinGW-w64 cross-compiler not found!"
    echo "Install with: sudo apt install mingw-w64"
    exit 1
fi

echo "✓ MinGW-w64 cross-compiler found"
echo "Compiler version:"
x86_64-w64-mingw32-gcc --version | head -1

# Create windows build directory in proper location
BUILD_DIR="model/releases/windows-x64"
mkdir -p "$BUILD_DIR/bin"

echo ""
echo "Building Windows executable with MinGW-w64..."

# Cross-compile the main transformer program
echo "Building AryanAi.exe..."
x86_64-w64-mingw32-gcc -std=c99 -O2 -Wall -Wextra \
    -DPLATFORM_WINDOWS \
    -Iinclude \
    -o "$BUILD_DIR/bin/AryanAi.exe" \
    main.c \
    src/interactive_cli.c \
    src/tokenizer.c \
    src/transformer_model.c \
    src/matrix.c \
    src/optimizer.c \
    src/platform.c \
    src/utils.c \
    -lm

# Cross-compile the CSV preprocessor
echo "Building preprocess_csv.exe..."
x86_64-w64-mingw32-gcc -std=c99 -O2 -Wall -Wextra \
    -DPLATFORM_WINDOWS \
    -Iinclude \
    -o "$BUILD_DIR/bin/preprocess_csv.exe" \
    preprocess_csv.c \
    -lm

echo ""
echo "✓ Windows cross-compilation completed successfully!"
echo ""
echo "Built executables:"
echo "  - $BUILD_DIR/bin/AryanAi.exe"
echo "  - $BUILD_DIR/bin/preprocess_csv.exe"
echo ""

# Test if the executables were created
if [[ -f "$BUILD_DIR/bin/AryanAi.exe" && -f "$BUILD_DIR/bin/preprocess_csv.exe" ]]; then
    echo "File sizes:"
    ls -lh "$BUILD_DIR/bin/"*.exe
    echo ""
    echo "✓ All Windows executables built successfully!"
else
    echo "❌ Error: Some executables were not created"
    exit 1
fi
