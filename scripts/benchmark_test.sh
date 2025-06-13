#!/bin/bash
# AryanAi Manual Benchmark Test Suite
# Tests both Linux and Windows versions with various configurations

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test configurations
BENCHMARK_DIR="/workspaces/AryanAlpha/benchmark_results"
LINUX_RELEASE="/workspaces/AryanAlpha/model/releases/linux-x64"
WINDOWS_RELEASE="/workspaces/AryanAlpha/model/releases/windows-x64"

# Create benchmark directory
mkdir -p "$BENCHMARK_DIR"

echo -e "${BLUE}===========================================${NC}"
echo -e "${BLUE}    AryanAi Cross-Platform Benchmark   ${NC}"
echo -e "${BLUE}===========================================${NC}"
echo
echo "Date: $(date)"
echo "Host: $(uname -a)"
echo "CPU: $(grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)"
echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
echo

# Function to run timed test
run_timed_test() {
    local test_name="$1"
    local command="$2"
    local platform="$3"
    
    echo -e "${YELLOW}Testing: $test_name ($platform)${NC}"
    echo "Command: $command"
    
    # Record start time
    start_time=$(date +%s.%N)
    
    # Run command and capture output
    if eval "$command" > "$BENCHMARK_DIR/${test_name}_${platform}.log" 2>&1; then
        # Record end time
        end_time=$(date +%s.%N)
        duration=$(echo "$end_time - $start_time" | bc)
        echo -e "${GREEN}✅ Success - Time: ${duration}s${NC}"
        echo "$duration" > "$BENCHMARK_DIR/${test_name}_${platform}_time.txt"
        return 0
    else
        end_time=$(date +%s.%N)
        duration=$(echo "$end_time - $start_time" | bc)
        echo -e "${RED}❌ Failed - Time: ${duration}s${NC}"
        echo "$duration" > "$BENCHMARK_DIR/${test_name}_${platform}_time.txt"
        return 1
    fi
}

# Function to test model configuration
test_model_config() {
    local config_name="$1"
    local epochs="$2"
    local batch_size="$3"
    local seq_len="$4"
    local d_model="$5"
    local n_layers="$6"
    local n_heads="$7"
    local platform="$8"
    local executable="$9"
    local wine_prefix="${10}"
    
    echo -e "\n${BLUE}--- Testing $config_name on $platform ---${NC}"
    
    local base_cmd="$wine_prefix $executable --mode train --data data/test_training_data.txt --epochs $epochs --batch_size $batch_size --seq_len $seq_len --d_model $d_model --n_layers $n_layers --n_heads $n_heads --lr 1e-4"
    
    # Training test
    local train_cmd="$base_cmd --weights_save models/bench_${config_name}_${platform}.bin --vocab_save models/bench_${config_name}_${platform}.vocab"
    run_timed_test "train_${config_name}" "$train_cmd" "$platform"
    
    # Generation test (if training succeeded)
    if [ -f "models/bench_${config_name}_${platform}.bin" ]; then
        local gen_cmd="$wine_prefix $executable --mode generate --weights_load models/bench_${config_name}_${platform}.bin --vocab_load models/bench_${config_name}_${platform}.vocab --prompt 'The quick brown fox' --max_new 20"
        run_timed_test "generate_${config_name}" "$gen_cmd" "$platform"
    fi
}

# Test 1: Quick Performance Test
echo -e "\n${BLUE}=== QUICK PERFORMANCE TEST ===${NC}"

cd "$LINUX_RELEASE"
echo -e "\n${YELLOW}Linux Native Performance:${NC}"
test_model_config "quick" 2 4 32 64 2 4 "linux" "./bin/Aaryan" ""

cd "$WINDOWS_RELEASE"
echo -e "\n${YELLOW}Windows (Wine) Performance:${NC}"
test_model_config "quick" 2 4 32 64 2 4 "windows" "./bin/Aaryan.exe" "wine64"

# Test 2: Small Model Configuration
echo -e "\n${BLUE}=== SMALL MODEL TEST ===${NC}"

cd "$LINUX_RELEASE"
echo -e "\n${YELLOW}Linux Small Model:${NC}"
test_model_config "small" 3 8 64 128 3 8 "linux" "./bin/Aaryan" ""

cd "$WINDOWS_RELEASE"
echo -e "\n${YELLOW}Windows Small Model:${NC}"
test_model_config "small" 3 8 64 128 3 8 "windows" "./bin/Aaryan.exe" "wine64"

# Test 3: Medium Model Configuration
echo -e "\n${BLUE}=== MEDIUM MODEL TEST ===${NC}"

cd "$LINUX_RELEASE"
echo -e "\n${YELLOW}Linux Medium Model:${NC}"
test_model_config "medium" 2 4 128 256 4 8 "linux" "./bin/Aaryan" ""

cd "$WINDOWS_RELEASE"
echo -e "\n${YELLOW}Windows Medium Model:${NC}"
test_model_config "medium" 2 4 128 256 4 8 "windows" "./bin/Aaryan.exe" "wine64"

# Test 4: Memory Stress Test
echo -e "\n${BLUE}=== MEMORY STRESS TEST ===${NC}"

cd "$LINUX_RELEASE"
echo -e "\n${YELLOW}Linux Memory Stress:${NC}"
test_model_config "stress" 1 2 256 512 6 16 "linux" "./bin/Aaryan" ""

cd "$WINDOWS_RELEASE"
echo -e "\n${YELLOW}Windows Memory Stress:${NC}"
test_model_config "stress" 1 2 256 512 6 16 "windows" "./bin/Aaryan.exe" "wine64"

# Test 5: Preprocessing Benchmark
echo -e "\n${BLUE}=== PREPROCESSING BENCHMARK ===${NC}"

# Create test CSV data
cat > "$BENCHMARK_DIR/test_large.csv" << 'EOF'
text,label
"This is a positive sentiment example that should be processed correctly",1
"This is a negative sentiment example for testing preprocessing speed",0
"Another positive example with more text to make processing take longer",1
"Negative example with additional content for benchmarking purposes only",0
"Medium length positive text example for comprehensive testing procedures",1
"Medium length negative text example for comprehensive testing procedures",0
"Long positive sentiment example with extended content designed to test the preprocessing capabilities and performance of the system under various load conditions",1
"Long negative sentiment example with extended content designed to test the preprocessing capabilities and performance of the system under various load conditions",0
EOF

# Replicate the data to make it larger
for i in {1..100}; do
    tail -n +2 "$BENCHMARK_DIR/test_large.csv" >> "$BENCHMARK_DIR/test_large_extended.csv"
done

cd "$LINUX_RELEASE"
echo -e "\n${YELLOW}Linux Preprocessing:${NC}"
run_timed_test "preprocess" "./bin/preprocess_csv $BENCHMARK_DIR/test_large_extended.csv $BENCHMARK_DIR/processed_linux.txt" "linux"

cd "$WINDOWS_RELEASE"
echo -e "\n${YELLOW}Windows Preprocessing:${NC}"
run_timed_test "preprocess" "wine64 ./bin/preprocess_csv.exe $BENCHMARK_DIR/test_large_extended.csv $BENCHMARK_DIR/processed_windows.txt" "windows"

# Generate Benchmark Report
echo -e "\n${BLUE}=== GENERATING BENCHMARK REPORT ===${NC}"

cat > "$BENCHMARK_DIR/benchmark_report.md" << 'EOF'
# AryanAi Cross-Platform Benchmark Report

## Test Environment
EOF

echo "- Date: $(date)" >> "$BENCHMARK_DIR/benchmark_report.md"
echo "- Host: $(uname -a)" >> "$BENCHMARK_DIR/benchmark_report.md"
echo "- CPU: $(grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)" >> "$BENCHMARK_DIR/benchmark_report.md"
echo "- Memory: $(free -h | grep Mem | awk '{print $2}')" >> "$BENCHMARK_DIR/benchmark_report.md"
echo "" >> "$BENCHMARK_DIR/benchmark_report.md"

cat >> "$BENCHMARK_DIR/benchmark_report.md" << 'EOF'
## Performance Results

| Test Configuration | Linux Time (s) | Windows Time (s) | Performance Ratio |
|-------------------|----------------|------------------|-------------------|
EOF

# Calculate performance comparisons
for test in quick small medium stress; do
    if [ -f "$BENCHMARK_DIR/train_${test}_linux_time.txt" ] && [ -f "$BENCHMARK_DIR/train_${test}_windows_time.txt" ]; then
        linux_time=$(cat "$BENCHMARK_DIR/train_${test}_linux_time.txt")
        windows_time=$(cat "$BENCHMARK_DIR/train_${test}_windows_time.txt")
        ratio=$(echo "scale=2; $windows_time / $linux_time" | bc)
        echo "| Training ($test) | $linux_time | $windows_time | ${ratio}x |" >> "$BENCHMARK_DIR/benchmark_report.md"
    fi
done

echo "" >> "$BENCHMARK_DIR/benchmark_report.md"
echo "## Model File Sizes" >> "$BENCHMARK_DIR/benchmark_report.md"
echo "" >> "$BENCHMARK_DIR/benchmark_report.md"

# Check model file sizes
cd "$LINUX_RELEASE"
echo "### Linux Models" >> "$BENCHMARK_DIR/benchmark_report.md"
for model in models/bench_*.bin; do
    if [ -f "$model" ]; then
        size=$(ls -lh "$model" | awk '{print $5}')
        name=$(basename "$model")
        echo "- $name: $size" >> "$BENCHMARK_DIR/benchmark_report.md"
    fi
done

cd "$WINDOWS_RELEASE"
echo "" >> "$BENCHMARK_DIR/benchmark_report.md"
echo "### Windows Models" >> "$BENCHMARK_DIR/benchmark_report.md"
for model in models/bench_*.bin; do
    if [ -f "$model" ]; then
        size=$(ls -lh "$model" | awk '{print $5}')
        name=$(basename "$model")
        echo "- $name: $size" >> "$BENCHMARK_DIR/benchmark_report.md"
    fi
done

echo "" >> "$BENCHMARK_DIR/benchmark_report.md"
echo "## Cross-Platform Compatibility" >> "$BENCHMARK_DIR/benchmark_report.md"
echo "- ✅ Both Linux and Windows executables functional" >> "$BENCHMARK_DIR/benchmark_report.md"
echo "- ✅ Model files saved in proper directory structure" >> "$BENCHMARK_DIR/benchmark_report.md"
echo "- ✅ Training and generation work on both platforms" >> "$BENCHMARK_DIR/benchmark_report.md"
echo "- ✅ Preprocessing utilities functional on both platforms" >> "$BENCHMARK_DIR/benchmark_report.md"

echo -e "\n${GREEN}===========================================${NC}"
echo -e "${GREEN}    BENCHMARK TESTING COMPLETED!          ${NC}"
echo -e "${GREEN}===========================================${NC}"
echo
echo -e "Results saved in: ${YELLOW}$BENCHMARK_DIR/${NC}"
echo -e "Full report: ${YELLOW}$BENCHMARK_DIR/benchmark_report.md${NC}"
echo
echo -e "${BLUE}Summary of tests performed:${NC}"
echo "✅ Quick performance comparison"
echo "✅ Small model configuration test"
echo "✅ Medium model configuration test"
echo "✅ Memory stress test"
echo "✅ Preprocessing performance test"
echo "✅ Cross-platform compatibility verification"
echo "✅ Model file size analysis"
echo
