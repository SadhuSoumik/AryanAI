# Benchmark Results Summary

This directory contains performance benchmark results for the AryanAi transformer model across different configurations and platforms.

## Benchmark Categories

### Training Benchmarks

- **Quick**: Fast training with minimal parameters for testing
- **Small**: Lightweight model configuration
- **Medium**: Balanced performance and quality configuration
- **Stress**: Maximum parameter stress test

### Generation Benchmarks

- **Quick**: Fast text generation testing
- **Small**: Small model text generation
- **Medium**: Medium model text generation

## Platform Testing

- **Linux**: Native compilation and execution
- **Windows**: Cross-compiled with MinGW-w64

## File Structure

- `*_time.txt`: Execution timing data
- `*.log`: Detailed training/generation logs

## Running Benchmarks

To generate new benchmark results:

```bash
# Run comprehensive benchmarks
./scripts/benchmark_test.sh

# View results
ls -la benchmark_results/
```

## Performance Summary

Based on the benchmark results:

| Configuration | Platform | Avg Training Time | Memory Usage | Model Quality |
| ------------- | -------- | ----------------- | ------------ | ------------- |
| Quick         | Linux    | < 2 minutes       | ~50MB        | Basic         |
| Small         | Linux    | ~5 minutes        | ~100MB       | Good          |
| Medium        | Linux    | ~15 minutes       | ~200MB       | High          |
| Stress        | Linux    | ~45+ minutes      | ~500MB       | Maximum       |

Cross-platform performance is generally consistent between Linux native and Windows cross-compiled builds.
