# AryanAlpha Windows x64 Release

Windows 64-bit distribution of the AryanAlpha C Transformer Model.

## Contents

- `bin/Aaryan.exe` - Main transformer training and inference executable
- `bin/preprocess_csv.exe` - CSV data preprocessing utility
- `data/` - Sample training and test data files

## System Requirements

- Windows 7 or later (64-bit)
- No additional runtime dependencies required

## Quick Start

1. Open Command Prompt or PowerShell
2. Navigate to this directory
3. Run the main program:

```cmd
bin\Aaryan.exe --help
```

## Training a Model

To train with the included sample data:

```cmd
bin\Aaryan.exe --train --input data\test_training_data.txt --output model_weights.bin
```

## Text Generation

To generate text using a trained model:

```cmd
bin\Aaryan.exe --generate --model model_weights.bin --prompt "Hello world"
```

## Data Preprocessing

To preprocess CSV files for training:

```cmd
bin\preprocess_csv.exe input.csv output.txt
```

## Troubleshooting

### Missing DLL Errors
If you encounter missing DLL errors, this indicates an issue with the cross-compilation. The executables should be self-contained.

### Antivirus False Positives
Some antivirus software may flag the executable as suspicious because it's an unsigned binary. This is a false positive - you can safely add an exception.

### Performance Issues
- Ensure you have sufficient RAM (recommended: 4GB+)
- Close other resource-intensive applications
- Use SSD storage for better I/O performance

## Support

For issues specific to the Windows build, please check:
1. Windows version compatibility (Windows 7+ 64-bit required)
2. Available system memory
3. File permissions in the installation directory

Built with MinGW-w64 cross-compiler from Linux.
