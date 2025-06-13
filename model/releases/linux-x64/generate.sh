#!/bin/bash
# Enhanced Text Generation Script for AryanAlpha Linux
echo "Starting AryanAlpha Text Generation..."
echo

# Check if trained model exists
if [ ! -f "models/aryan_model.bin" ]; then
    echo "❌ Error: No trained model found!"
    echo "Please run ./train.sh first to create a model."
    exit 1
fi

# Use provided prompt or default
PROMPT="${1:-The quick brown fox}"
MAX_TOKENS="${2:-50}"

echo "Using trained model: models/aryan_model.bin"
echo "Vocabulary: models/aryan_vocab.vocab"
echo "Prompt: \"$PROMPT\""
echo "Max new tokens: $MAX_TOKENS"
echo

./bin/Aaryan --mode generate --weights_load models/aryan_model.bin --vocab_load models/aryan_vocab.vocab --prompt "$PROMPT" --max_new $MAX_TOKENS

if [ $? -eq 0 ]; then
    echo
    echo "✅ Text generation completed!"
else
    echo
    echo "❌ Text generation failed with error code: $?"
    exit 1
fi
