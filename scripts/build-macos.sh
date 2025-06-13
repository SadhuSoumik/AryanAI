#!/bin/bash
# macOS build script

set -e  # Exit on any error

echo "=== C Transformer macOS Build Script ==="

# Check for Xcode Command Line Tools
if ! command -v gcc >/dev/null 2>&1; then
    echo "Error: Xcode Command Line Tools not found"
    echo "Please install with: xcode-select --install"
    exit 1
fi

echo "✓ Xcode Command Line Tools found"

# Check for Homebrew (optional but recommended)
if command -v brew >/dev/null 2>&1; then
    echo "✓ Homebrew found"
    
    # Install OpenMP if available
    if ! brew list libomp >/dev/null 2>&1; then
        echo "Installing OpenMP for better performance..."
        brew install libomp
    else
        echo "✓ OpenMP already installed"
    fi
else
    echo "Warning: Homebrew not found. Consider installing for easier dependency management:"
    echo '  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
fi

echo ""
echo "Building with CMake..."

# Create build directory
mkdir -p build-macos
cd build-macos

# Configure
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build . --config Release

echo ""
echo "✓ Build completed successfully!"
echo "Executable: $(pwd)/c_transformer"
echo ""
echo "To test the build:"
echo "  ./c_transformer --help"
