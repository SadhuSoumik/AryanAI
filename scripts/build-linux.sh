#!/bin/bash
# Linux build script

set -e  # Exit on any error

echo "=== C Transformer Linux Build Script ==="

# Detect distribution
if command -v apt-get >/dev/null 2>&1; then
    DISTRO="debian"
    PKG_MGR="apt-get"
elif command -v yum >/dev/null 2>&1; then
    DISTRO="rhel"
    PKG_MGR="yum"
elif command -v dnf >/dev/null 2>&1; then
    DISTRO="fedora"
    PKG_MGR="dnf"
elif command -v pacman >/dev/null 2>&1; then
    DISTRO="arch"
    PKG_MGR="pacman"
else
    DISTRO="unknown"
fi

echo "Detected distribution: $DISTRO"

# Function to install packages
install_deps() {
    case $DISTRO in
        debian)
            echo "Installing dependencies with apt-get..."
            sudo apt-get update
            sudo apt-get install -y build-essential cmake libpthread-stubs0-dev
            ;;
        rhel)
            echo "Installing dependencies with yum..."
            sudo yum groupinstall -y "Development Tools"
            sudo yum install -y cmake
            ;;
        fedora)
            echo "Installing dependencies with dnf..."
            sudo dnf groupinstall -y "Development Tools"
            sudo dnf install -y cmake
            ;;
        arch)
            echo "Installing dependencies with pacman..."
            sudo pacman -S --needed --noconfirm base-devel cmake
            ;;
        *)
            echo "Unknown distribution. Please install the following manually:"
            echo "  - GCC compiler"
            echo "  - CMake"
            echo "  - pthread development libraries"
            ;;
    esac
}

# Check for required tools
if ! command -v gcc >/dev/null 2>&1; then
    echo "GCC not found. Installing dependencies..."
    install_deps
else
    echo "✓ GCC found"
fi

if ! command -v cmake >/dev/null 2>&1; then
    echo "CMake not found. Installing dependencies..."
    install_deps
else
    echo "✓ CMake found"
fi

echo ""
echo "Building with CMake..."

# Create build directory
mkdir -p build-linux
cd build-linux

# Configure
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build . --config Release

echo ""
echo "✓ Build completed successfully!"
echo "Executable: $(pwd)/Aaryan"
echo ""
echo "To test the build:"
echo "  ./Aaryan --help"
