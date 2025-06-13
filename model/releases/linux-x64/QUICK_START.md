# AryanAlpha Linux x64 Release

## Quick Start for Linux

This directory contains the Linux x64 release of AryanAlpha.

### Contents
- `bin/Aaryan` - Main transformer executable
- `bin/preprocess_csv` - CSV preprocessing utility
- `data/` - Sample training and test data

### Running the Program

1. Make executables runnable:
```bash
chmod +x bin/Aaryan bin/preprocess_csv
```

2. View help:
```bash
./bin/Aaryan --help
```

3. Train with sample data:
```bash
./bin/Aaryan --train --input data/test_training_data.txt
```

4. Generate text:
```bash
./bin/Aaryan --generate --prompt "Hello world"
```

### System Requirements
- Linux x86_64
- No additional dependencies required

For detailed documentation, see the main project README.
