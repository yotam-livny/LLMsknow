#!/bin/bash
# Full pipeline: Generate answers -> Extract exact answers -> Train and save probe
# Usage: ./run_pipeline.sh [n_samples] (default: 1000)

set -e  # Exit on error

# Configuration
MODEL="mistralai/Mistral-7B-Instruct-v0.2"
DATASET="movies"
N_SAMPLES="${1:-1000}"  # Default to 1000 if not provided
PROBE_LAYER=15
PROBE_TOKEN="exact_answer_last_token"
PROBE_LOCATION="mlp_last_layer_only_input"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Change to src directory
cd "$(dirname "$0")/src"

echo ""
echo "=========================================="
echo "  LLMsKnow Full Pipeline"
echo "=========================================="
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Samples: $N_SAMPLES"
echo "Probe: Layer $PROBE_LAYER, Token $PROBE_TOKEN"
echo "=========================================="
echo ""

# Step 1: Generate model answers
echo -e "${BLUE}Step 1/3: Generating model answers${NC}"
echo "----------------------------------------"
echo "This will process $N_SAMPLES samples..."
echo "Estimated time: ~$((N_SAMPLES * 12 / 60)) minutes"
echo ""

WANDB_MODE=offline python3 generate_model_answers.py \
    --model $MODEL \
    --dataset $DATASET \
    --n_samples $N_SAMPLES

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Step 1 complete${NC}"
    echo ""
else
    echo -e "${YELLOW}✗ Step 1 failed${NC}"
    exit 1
fi

# Check output file was created
# Convert model name to friendly name (e.g., "mistralai/Mistral-7B-Instruct-v0.2" -> "mistral-7b-instruct")
MODEL_FRIENDLY=$(echo "$MODEL" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]' | sed 's/-v[0-9.]*$//')
OUTPUT_FILE="../output/${MODEL_FRIENDLY}-answers-${DATASET}.csv"

if [ -f "$OUTPUT_FILE" ]; then
    LINES=$(wc -l < "$OUTPUT_FILE")
    echo "Generated answers file: $OUTPUT_FILE ($LINES lines)"
    echo ""
else
    echo "Warning: Output file not found"
fi

# Step 2: Extract exact answers
echo -e "${BLUE}Step 2/3: Extracting exact answers${NC}"
echo "----------------------------------------"
echo ""

WANDB_MODE=offline python3 extract_exact_answer.py \
    --model $MODEL \
    --dataset $DATASET \
    --extraction_model $MODEL

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Step 2 complete${NC}"
    echo ""
else
    echo -e "${YELLOW}✗ Step 2 failed${NC}"
    exit 1
fi

# Step 3: Train and save probe
echo -e "${BLUE}Step 3/3: Training and saving probe${NC}"
echo "----------------------------------------"
echo "Training classifier at:"
echo "  - Location: $PROBE_LOCATION"
echo "  - Layer: $PROBE_LAYER"
echo "  - Token: $PROBE_TOKEN"
echo ""

WANDB_MODE=offline python3 probe.py \
    --model $MODEL \
    --probe_at $PROBE_LOCATION \
    --seeds 0 \
    --n_samples all \
    --dataset $DATASET \
    --layer $PROBE_LAYER \
    --token $PROBE_TOKEN \
    --save_clf

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Step 3 complete${NC}"
    echo ""
else
    echo -e "${YELLOW}✗ Step 3 failed${NC}"
    exit 1
fi

# Final summary
echo ""
echo "=========================================="
echo -e "${GREEN}Pipeline Complete!${NC}"
echo "=========================================="
echo ""
echo "Output files:"
echo "  - Answers: ../output/mistral-7b-instruct-answers-${DATASET}.csv"
echo "  - Input/Output IDs: ../output/mistral-7b-instruct-input_output_ids-${DATASET}.pt"
echo "  - Scores: ../output/mistral-7b-instruct-scores-${DATASET}.pt"
echo ""
echo "Probe saved to:"
echo "  ../checkpoints/clf_mistral-7b-instruct_${DATASET}_layer-${PROBE_LAYER}_token-${PROBE_TOKEN}.pkl"
echo ""
