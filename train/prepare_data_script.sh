#!/bin/bash
# Prepare AnghaBench data for C to Assembly training

set -e

echo "=== Data Preparation for C to Assembly Training ==="

# Input and output paths
INPUT_FILE=${1:-"AnghaBench_compile.jsonl"}
OUTPUT_DIR=${2:-"./prepared_data"}

# Create output directory
mkdir -p $OUTPUT_DIR

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file $INPUT_FILE not found!"
    exit 1
fi

echo "Input file: $INPUT_FILE"
echo "Output directory: $OUTPUT_DIR"

# First, analyze the dataset
echo -e "\n=== Analyzing Dataset ==="
python3 data_converter.py "$INPUT_FILE" dummy.jsonl --analyze

# Convert for different optimization levels
echo -e "\n=== Converting Dataset ==="

# Option 1: Create separate datasets for each optimization level
echo -e "\n1. Creating optimization-level-specific datasets..."
for opt_level in "opt-state-O0" "opt-state-O1" "opt-state-O2" "opt-state-O3"; do
    output_file="$OUTPUT_DIR/train_${opt_level#opt-state-}.jsonl"
    echo "Converting for $opt_level -> $output_file"
    python3 data_converter.py "$INPUT_FILE" "$output_file" \
        --optimization-level "$opt_level" \
        --include-function-name \
        --shuffle \
        --validation-split 0.05 \
        --show-samples 2
done

# Option 2: Create a combined dataset with all optimization levels
echo -e "\n2. Creating combined dataset with all optimization levels..."
output_file="$OUTPUT_DIR/train_all_levels.jsonl"
python3 data_converter.py "$INPUT_FILE" "$output_file" \
    --optimization-level "all" \
    --include-function-name \
    --shuffle \
    --validation-split 0.05 \
    --show-samples 3

# Option 3: Create a default O2-optimized dataset (recommended for initial training)
echo -e "\n3. Creating default O2-optimized dataset..."
output_file="$OUTPUT_DIR/l4av_train_O2.jsonl"
python3 data_converter.py "$INPUT_FILE" "$output_file" \
    --optimization-level "opt-state-O2" \
    --include-function-name \
    --shuffle \
    --validation-split 0.05 \
    --show-samples 3

# Summary
echo -e "\n=== Data Preparation Complete ==="
echo "Generated datasets:"
ls -lh $OUTPUT_DIR/*.jsonl

# Count examples in each file
echo -e "\nExample counts:"
for file in $OUTPUT_DIR/*.jsonl; do
    count=$(wc -l < "$file")
    echo "  $(basename $file): $count examples"
done

# Create a README for the dataset
cat > $OUTPUT_DIR/README.md << EOF
# Prepared C to Assembly Training Data

Generated from AnghaBench dataset on $(date)

## Files

- **l4av_train_O2.jsonl**: Default training set with O2 optimization (recommended)
- **l4av_train_O2_val.jsonl**: Validation set for O2 optimization
- **train_O0.jsonl**: Training set with O0 optimization (no optimization)
- **train_O1.jsonl**: Training set with O1 optimization
- **train_O2.jsonl**: Training set with O2 optimization
- **train_O3.jsonl**: Training set with O3 optimization
- **train_all_levels.jsonl**: Combined dataset with all optimization levels

## Format

Each line is a JSON object with:
- **instruction**: The C code with optimization level prefix
- **output**: The corresponding assembly code

## Usage

Use with the finetune.py script:
\`\`\`bash
python finetune.py \\
    --data_path ./prepared_data/l4av_train_O2.jsonl \\
    --model_name_or_path codellama/CodeLlama-34b-hf \\
    --output_dir ./output
\`\`\`
EOF

echo -e "\nREADME created at: $OUTPUT_DIR/README.md"

# Recommend next steps
echo -e "\n=== Next Steps ==="
echo "1. Review the generated datasets in $OUTPUT_DIR"
echo "2. Choose a dataset for training:"
echo "   - For general C to Assembly: use l4av_train_O2.jsonl"
echo "   - For specific optimization level: use train_O{0,1,2,3}.jsonl"
echo "   - For multi-optimization training: use train_all_levels.jsonl"
echo "3. Run training with:"
echo "   ./optimized_hetero_script.sh zero3_optim codellama/CodeLlama-34b-hf $OUTPUT_DIR/l4av_train_O2.jsonl"
