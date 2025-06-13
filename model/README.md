# AryanAlpha Model Directory

This directory contains organized builds, releases, and model assets for the AryanAlpha transformer project.

## Directory Structure

```
model/
├── releases/           # Platform-specific release packages
│   ├── linux-x64/     # Linux x64 distribution
│   ├── windows-x64/   # Windows x64 distribution
│   └── macos-x64/     # macOS x64 distribution (planned)
├── builds/            # Build artifacts and object files
│   ├── linux-x64-obj/    # Linux build objects
│   └── windows-x64-cross/ # Windows cross-compilation artifacts
├── data/              # Shared training and test data
├── weights/           # Trained model weights and checkpoints
└── README.md          # This file
```

## Releases

### Linux x64
- **Location**: `releases/linux-x64/`
- **Executables**: `bin/Aaryan`, `bin/preprocess_csv`
- **Status**: ✅ Ready

### Windows x64
- **Location**: `releases/windows-x64/`
- **Executables**: `bin/Aaryan.exe`, `bin/preprocess_csv.exe`
- **Status**: ✅ Ready (cross-compiled from Linux)

### macOS x64
- **Location**: `releases/macos-x64/`
- **Status**: 📋 Planned

## Usage

### Linux
```bash
cd releases/linux-x64
./bin/Aaryan --help
```

### Windows
```cmd
cd releases\windows-x64
bin\Aaryan.exe --help
```

## Data Files

Training and test data are available in each platform's `data/` directory and the shared `data/` directory.

## Build Artifacts

Intermediate build files and object files are stored in the `builds/` directory for debugging and incremental compilation.
