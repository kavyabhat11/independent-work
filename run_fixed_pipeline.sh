#!/bin/bash
set -e  # Exit on any error

echo "=========================================="
echo "MOZART FINETUNING PIPELINE (FIXED VERSION)"
echo "=========================================="
echo ""

# Step 0: Convert .krn files to .xml (if any exist)
echo "[0/5] Converting Kern files to MusicXML..."
conda run -n chordgpu python3 convert_krn_to_xml.py --data_dir ./data/mozart
echo ""

# Step 1: Run tests to verify conversion logic
echo "[1/5] Running conversion tests..."
conda run -n chordgpu python3 test_conversion.py
if [ $? -ne 0 ]; then
    echo "ERROR: Tests failed! Aborting."
    exit 1
fi
echo ""

# Step 2: Convert Mozart data with FIXED conversion script
echo "[2/5] Converting Mozart XML+TXT to TSV (with proper voice leading, etc.)..."
conda run -n chordgpu python3 convert_mozart_to_tsv_FIXED.py \
    --data_dir ./data/mozart \
    --output_dir ./mozart_tsv_fixed \
    --grid 0.5
echo ""

# Step 3: Split into train/val/test BY WORK (keeps movements together)
echo "[3/5] Splitting data into train/val/test..."
# Will auto-adjust based on actual number of files found
conda run -n chordgpu python3 split_mozart_data_FIXED.py \
    --tsv_dir ./mozart_tsv_fixed \
    --output_dir ./mozart_dataset_fixed \
    --n_train 34 \
    --n_val 20
echo ""

# Step 4: Run finetuning
echo "[4/5] Starting finetuning with fixed data..."
echo "NOTE: This will take a while. Monitor with wandb."
MOZART_ROOT=./mozart_dataset_fixed conda run -n chordgpu python3 finetune_from_checkpoint_FIXED.py
echo ""

echo "=========================================="
echo "PIPELINE COMPLETE!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Check W&B for training curves"
echo "  2. Run batch_analyse_mozart.py with the new finetuned model"
echo "  3. Compare results with batch_compare_mozart_aggregate.py"
echo ""
