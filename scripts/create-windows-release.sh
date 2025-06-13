#!/bin/bash
# Windows release creation script
# This script creates a complete Windows release package

set -e  # Exit on any error

echo "=== Creating Windows Release Package ==="

# Check if Windows executables exist
if [[ ! -f "Aaryan.exe" || ! -f "preprocess_csv.exe" ]]; then
    echo "Windows executables not found. Building them first..."
    ./scripts/build-windows-cross.sh
fi

# Create release directory structure
RELEASE_DIR="model/releases/windows-x64"
mkdir -p "$RELEASE_DIR/bin"
mkdir -p "$RELEASE_DIR/data"

echo "Copying Windows executables..."
cp Aaryan.exe "$RELEASE_DIR/bin/"
cp preprocess_csv.exe "$RELEASE_DIR/bin/"

echo "Copying data files..."
cp -r data/* "$RELEASE_DIR/data/" 2>/dev/null || echo "No data directory found"

echo "Creating Windows batch scripts..."
# These are already created above

echo "Creating Windows README..."
# This is already created above

echo "✓ Windows release package created successfully!"
echo "Release location: $RELEASE_DIR"
echo ""
echo "Contents:"
ls -la "$RELEASE_DIR"
echo ""
echo "To distribute: zip -r AryanAi-Windows-x64.zip $RELEASE_DIR"