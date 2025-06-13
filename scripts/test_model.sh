#!/bin/bash

# ====================================================================
# C Transformer Model Test Script
# ====================================================================
# Comprehensive test suite for the C Transformer model
# Tests both training and inference capabilities
# ====================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Test configuration
TEST_OUTPUT_DIR="logs"
PASSED_TESTS=0
TOTAL_TESTS=0

# Test functions
print_test_header() {
    echo -e "\n${CYAN}=== $1 ===${NC}"
}

print_test_status() {
    echo -e "${BLUE}[TEST]${NC} $1"
}

test_passed() {
    echo -e "${GREEN}✓ PASSED:${NC} $1"
    ((PASSED_TESTS++))
}

test_failed() {
    echo -e "${RED}✗ FAILED:${NC} $1"
}

test_warning() {
    echo -e "${YELLOW}⚠ WARNING:${NC} $1"
}

run_test() {
    local test_name="$1"
    local test_command="$2"
    local expected_exit_code="${3:-0}"
    
    ((TOTAL_TESTS++))
    print_test_status "Running: $test_name"
    
    local log_file="$TEST_OUTPUT_DIR/${test_name//[ :]/_}.log"
    
    if eval "$test_command" > "$log_file" 2>&1; then
        local exit_code=0
    else
        local exit_code=$?
    fi
    
    if [[ $exit_code -eq $expected_exit_code ]]; then
        test_passed "$test_name"
        return 0
    else
        test_failed "$test_name (exit code: $exit_code, expected: $expected_exit_code)"
        echo "  Log file: $log_file"
        return 1
    fi
}

# Detect platform and set executable paths
detect_platform() {
    if [[ -f "bin/Aaryan" ]]; then
        TRANSFORMER_EXE="bin/Aaryan.exe"
        PREPROCESS_EXE="bin/preprocess_csv.exe"
        PLATFORM="Windows"
    elif [[ -f "bin/Aaryan" ]]; then
        TRANSFORMER_EXE="bin/Aaryan"
        PREPROCESS_EXE="bin/preprocess_csv"
        PLATFORM="Linux"
    else
        echo "Error: No transformer executable found!"
        exit 1
    fi
}

# Check prerequisites
check_prerequisites() {
    print_test_header "Prerequisites Check"
    
    # Create logs directory
    mkdir -p "$TEST_OUTPUT_DIR"
    
    # Detect platform and executables
    detect_platform
    echo "Platform: $PLATFORM"
    echo "Transformer executable: $TRANSFORMER_EXE"
    
    # Check if executable exists and is executable
    if [[ ! -x "$TRANSFORMER_EXE" ]]; then
        test_failed "Transformer executable not found or not executable: $TRANSFORMER_EXE"
        exit 1
    fi
    test_passed "Transformer executable found"
    
    # Check for required data files
    local required_files=(
        "data/tokenizer.vocab"
        "data/test_data.txt"
        "data/test_training_data.txt"
        "models/transformer_weights.bin"
    )
    
    for file in "${required_files[@]}"; do
        if [[ -f "$file" ]]; then
            test_passed "Required file found: $file"
        else
            test_warning "Optional file missing: $file"
        fi
    done
}

# Test basic functionality
test_basic_functionality() {
    print_test_header "Basic Functionality Tests"
    
    # Test help command
    run_test "Help command" "$TRANSFORMER_EXE --help"
    
    # Test version command (if available)
    run_test "Version command" "$TRANSFORMER_EXE --version" || true
    
    # Test invalid argument handling
    run_test "Invalid argument handling" "$TRANSFORMER_EXE --invalid-argument" 1
}

# Test model loading
test_model_loading() {
    print_test_header "Model Loading Tests"
    
    # Test loading with valid model file
    if [[ -f "models/transformer_weights.bin" ]]; then
        run_test "Load model weights" "$TRANSFORMER_EXE --load-model models/transformer_weights.bin --test-mode"
    else
        test_warning "Model weights file not found - skipping model loading test"
    fi
    
    # Test loading with invalid model file
    run_test "Load invalid model" "$TRANSFORMER_EXE --load-model nonexistent.bin --test-mode" 1
}

# Test tokenizer functionality
test_tokenizer() {
    print_test_header "Tokenizer Tests"
    
    if [[ -f "data/tokenizer.vocab" ]]; then
        # Test tokenizer loading
        run_test "Load tokenizer vocabulary" "$TRANSFORMER_EXE --vocab data/tokenizer.vocab --test-mode"
        
        # Test tokenization with sample text
        echo "Hello world! This is a test." > "$TEST_OUTPUT_DIR/sample_input.txt"
        run_test "Tokenize sample text" "$TRANSFORMER_EXE --vocab data/tokenizer.vocab --tokenize $TEST_OUTPUT_DIR/sample_input.txt"
    else
        test_warning "Tokenizer vocabulary not found - skipping tokenizer tests"
    fi
}

# Test inference
test_inference() {
    print_test_header "Inference Tests"
    
    if [[ -f "data/test_data.txt" ]]; then
        # Test inference with test data
        run_test "Inference with test data" "$TRANSFORMER_EXE --input data/test_data.txt --output $TEST_OUTPUT_DIR/inference_output.txt"
        
        # Check if output was generated
        if [[ -f "$TEST_OUTPUT_DIR/inference_output.txt" ]]; then
            test_passed "Inference output file created"
        else
            test_failed "Inference output file not created"
        fi
    else
        test_warning "Test data file not found - skipping inference tests"
    fi
}

# Test training functionality
test_training() {
    print_test_header "Training Tests"
    
    if [[ -f "data/test_training_data.txt" ]]; then
        # Test training with minimal epochs
        run_test "Training with test data (1 epoch)" "$TRANSFORMER_EXE --train --data data/test_training_data.txt --epochs 1 --output-model $TEST_OUTPUT_DIR/trained_model.bin"
        
        # Check if trained model was saved
        if [[ -f "$TEST_OUTPUT_DIR/trained_model.bin" ]]; then
            test_passed "Training output model created"
            
            # Test loading the trained model
            run_test "Load trained model" "$TRANSFORMER_EXE --load-model $TEST_OUTPUT_DIR/trained_model.bin --test-mode"
        else
            test_failed "Training output model not created"
        fi
    else
        test_warning "Training data file not found - skipping training tests"
    fi
}

# Test CSV preprocessing (if available)
test_csv_preprocessing() {
    print_test_header "CSV Preprocessing Tests"
    
    if [[ -x "$PREPROCESS_EXE" ]]; then
        # Create sample CSV data
        cat > "$TEST_OUTPUT_DIR/sample.csv" << 'EOF'
text,label
"This is positive text",1
"This is negative text",0
"Another positive example",1
EOF
        
        run_test "CSV preprocessing" "$PREPROCESS_EXE $TEST_OUTPUT_DIR/sample.csv $TEST_OUTPUT_DIR/processed.txt"
        
        # Check if processed file was created
        if [[ -f "$TEST_OUTPUT_DIR/processed.txt" ]]; then
            test_passed "CSV preprocessing output created"
        else
            test_failed "CSV preprocessing output not created"
        fi
    else
        test_warning "CSV preprocessor not found - skipping CSV tests"
    fi
}

# Test performance benchmarks
test_performance() {
    print_test_header "Performance Tests"
    
    if [[ -f "data/test_data.txt" ]]; then
        # Time inference performance
        local start_time=$(date +%s)
        if run_test "Performance benchmark" "$TRANSFORMER_EXE --input data/test_data.txt --benchmark" > /dev/null 2>&1; then
            local end_time=$(date +%s)
            local duration=$((end_time - start_time))
            test_passed "Performance benchmark completed in ${duration}s"
        fi
    else
        test_warning "Test data not available for performance tests"
    fi
}

# Memory leak test (basic)
test_memory() {
    print_test_header "Memory Tests"
    
    # Run a quick memory test if valgrind is available
    if command -v valgrind >/dev/null 2>&1 && [[ "$PLATFORM" == "Linux" ]]; then
        run_test "Memory leak check" "valgrind --leak-check=summary --error-exitcode=1 $TRANSFORMER_EXE --test-mode"
    else
        test_warning "Valgrind not available - skipping memory tests"
    fi
}

# Generate test report
generate_report() {
    print_test_header "Test Summary"
    
    local report_file="$TEST_OUTPUT_DIR/test_report.txt"
    
    {
        echo "C Transformer Test Report"
        echo "========================"
        echo "Date: $(date)"
        echo "Platform: $PLATFORM"
        echo "Executable: $TRANSFORMER_EXE"
        echo ""
        echo "Test Results:"
        echo "  Passed: $PASSED_TESTS"
        echo "  Total:  $TOTAL_TESTS"
        echo "  Success Rate: $(( PASSED_TESTS * 100 / TOTAL_TESTS ))%"
        echo ""
        echo "Log files location: $TEST_OUTPUT_DIR/"
    } > "$report_file"
    
    echo "Test report saved to: $report_file"
    
    if [[ $PASSED_TESTS -eq $TOTAL_TESTS ]]; then
        echo -e "${GREEN}All tests passed! ✓${NC}"
        return 0
    else
        echo -e "${RED}Some tests failed! ✗${NC}"
        echo "Failed: $((TOTAL_TESTS - PASSED_TESTS)) out of $TOTAL_TESTS tests"
        return 1
    fi
}

# Main test execution
main() {
    echo "C Transformer Test Suite"
    echo "Platform: $(uname -s) $(uname -m)"
    echo "Working directory: $(pwd)"
    echo ""
    
    # Change to script directory
    cd "$(dirname "$0")"
    
    # Run all test suites
    check_prerequisites
    test_basic_functionality
    test_model_loading
    test_tokenizer
    test_inference
    test_training
    test_csv_preprocessing
    test_performance
    test_memory
    
    # Generate final report
    generate_report
}

# Run main function
main "$@"
