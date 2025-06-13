# Build Organization Guide

## Overview
The AryanAlpha C Transformer project now uses a well-organized build system that separates compiled artifacts into platform-specific deployment directories.

## Directory Structure

### Platform-Specific Deployment
```
deploy/
├── linux-x64/
│   ├── bin/          # Compiled executables
│   ├── obj/          # Compiled object files
│   │   ├── main.o    # Main program object
│   │   └── src/      # Source module objects
│   ├── data/         # Runtime data files
│   ├── models/       # Trained model files
│   └── docs/         # Platform-specific documentation
└── windows-x64/
    ├── bin/          # Windows executables (.exe)
    ├── obj/          # Windows object files
    │   ├── main.o
    │   └── src/
    ├── data/
    ├── models/
    └── docs/
```

## Build Systems

### Makefile
- Object files: `deploy/$(PLATFORM)/obj/`
- Executables: `deploy/$(PLATFORM)/bin/`
- Automatically creates deployment structure
- Cross-platform compatible

### CMake
- Configured for same deployment structure
- Uses `CMAKE_RUNTIME_OUTPUT_DIRECTORY` for executables
- Uses `CMAKE_ARCHIVE_OUTPUT_DIRECTORY` for object files

## Benefits

1. **Clean Source Tree**: No object files scattered in source directories
2. **Platform Separation**: Each platform has its own isolated build artifacts
3. **Easy Deployment**: Complete deployment packages in single directories
4. **Build Cache**: Object files preserved for incremental builds
5. **Cross-Platform**: Consistent structure across Linux, Windows, macOS

## Build Commands

### Full Build
```bash
make all                    # Build everything
make clean                  # Clean object files only
make clean-all             # Clean everything including deployment
```

### CMake Build
```bash
mkdir build && cd build
cmake ..
cmake --build .
```

## Executables

- **Aaryan**: Main transformer executable
- **preprocess_csv**: Data preprocessing utility

Both executables are placed in the appropriate `deploy/$(PLATFORM)/bin/` directory.
