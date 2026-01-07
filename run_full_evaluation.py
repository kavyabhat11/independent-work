#!/usr/bin/env python3
"""
Complete evaluation script:
1. Run ChordGNN analysis on test files using PRETRAINED and FINETUNED models
2. Convert MusicXML outputs to RomanText format
3. Run Dmitri's comparison script on both
"""

import os
import glob
import subprocess
import shutil

# ===============================================================
# CONFIGURATION
# ===============================================================

# Checkpoint paths
PRETRAINED_CKPT = "melkisedeath/chord_rec/model-kvd0jic5:v0"  # Original pretrained
FINETUNED_CKPT = "kb9520-princeton-university/chord_rec/mozart-finetuned-model-FIXED:v1"  # Finetuned

# Directories
TEST_XML_DIR = "data/mozart_test"  # Contains .xml test files
GT_TXT_DIR = "data/mozart_test"    # Contains ground truth .txt files

# Output directories
PRETRAINED_XML_DIR = "analysis/pretrained_musicxml"
FINETUNED_XML_DIR = "analysis/finetuned_musicxml"
PRETRAINED_TXT_DIR = "txt_analysis/pretrained"
FINETUNED_TXT_DIR = "txt_analysis/finetuned"

# Scripts
ANALYSE_SCRIPT = "analyse_score.py"
CONVERT_SCRIPT = "convert_musicxml_to_analysis.py"
COMPARE_MODULE = "compare"


# ===============================================================
# STEP 1: ANALYZE WITH PRETRAINED MODEL
# ===============================================================

def analyze_with_model(model_name, ckpt_path, output_dir):
    """Run ChordGNN analysis on all test XML files using specified checkpoint."""
    os.makedirs(output_dir, exist_ok=True)

    xml_files = sorted([f for f in os.listdir(TEST_XML_DIR)
                       if f.endswith(".xml") and "-analysis" not in f])

    print(f"\n{'='*70}")
    print(f"ANALYZING WITH {model_name.upper()}")
    print(f"{'='*70}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Test files: {len(xml_files)}")
    print(f"Output dir: {output_dir}\n")

    successful = 0
    failed = []

    for i, fname in enumerate(xml_files, 1):
        base = os.path.splitext(fname)[0]
        in_path = os.path.join(TEST_XML_DIR, fname)

        print(f"[{i}/{len(xml_files)}] Analyzing {base}...")

        cmd = [
            "python3", ANALYSE_SCRIPT,
            "--score_path", in_path,
            "--use_ckpt", ckpt_path,
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            # Check if output file was created
            expected_output = f"{base}-analysis.musicxml"
            output_path = os.path.join(TEST_XML_DIR, expected_output)

            if os.path.exists(output_path):
                # Move to model-specific directory
                dest_path = os.path.join(output_dir, expected_output)
                shutil.move(output_path, dest_path)
                print(f"  ✓ Success → {expected_output}")
                successful += 1
            else:
                print(f"  ✗ Failed: output file not created")
                failed.append(base)

        except subprocess.TimeoutExpired:
            print(f"  ✗ Failed: timeout")
            failed.append(base)
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            failed.append(base)

    print(f"\n{model_name.upper()} Analysis Complete:")
    print(f"  Successful: {successful}/{len(xml_files)}")
    if failed:
        print(f"  Failed: {failed}")

    return successful, failed


# ===============================================================
# STEP 2: CONVERT MUSICXML TO ROMANTEXT
# ===============================================================

def convert_to_text(xml_dir, txt_dir, model_name):
    """Convert MusicXML analysis files to RomanText format."""
    os.makedirs(txt_dir, exist_ok=True)

    xml_files = sorted([f for f in os.listdir(xml_dir) if f.endswith(".musicxml")])

    print(f"\n{'='*70}")
    print(f"CONVERTING {model_name.upper()} TO ROMANTEXT")
    print(f"{'='*70}")
    print(f"Input dir: {xml_dir}")
    print(f"Output dir: {txt_dir}")
    print(f"Files: {len(xml_files)}\n")

    successful = 0
    failed = []

    for i, fname in enumerate(xml_files, 1):
        base = os.path.splitext(fname)[0]  # e.g., "K279-2-analysis"
        in_path = os.path.join(xml_dir, fname)
        out_name = f"{base}.txt"
        out_path = os.path.join(txt_dir, out_name)

        print(f"[{i}/{len(xml_files)}] Converting {fname}...")

        cmd = ["python3", CONVERT_SCRIPT, in_path, out_path]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if os.path.exists(out_path):
                print(f"  ✓ Success → {out_name}")
                successful += 1
            else:
                print(f"  ✗ Failed: output not created")
                failed.append(base)

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            failed.append(base)

    print(f"\n{model_name.upper()} Conversion Complete:")
    print(f"  Successful: {successful}/{len(xml_files)}")
    if failed:
        print(f"  Failed: {failed}")

    return successful, failed


# ===============================================================
# STEP 3: RUN COMPARISON
# ===============================================================

def run_comparison(pred_dir, model_name):
    """Run Dmitri's comparison script."""
    print(f"\n{'='*70}")
    print(f"RUNNING COMPARISON FOR {model_name.upper()}")
    print(f"{'='*70}")
    print(f"Ground truth: {GT_TXT_DIR}")
    print(f"Predictions: {pred_dir}\n")

    # Import compare module
    import compare

    # Reset globals
    compare.globalIgnoreSixFour = False
    compare.globalIgnoreInversions = False
    compare.globalSave = False
    compare._USEANALYSISOFFSETS = False
    compare._PRINTINDIVIDUAL = False

    compare.keyErrors = 0
    compare.chordErrors = 0
    compare.totalCorrectEighths = 0
    compare.inversionErrors = 0
    compare.wrongKeyRightChord = 0
    compare.confusionMatrix = {}
    compare.confusions = {}

    from music21 import converter

    aggregate_pcts = []
    aggregate_wrongKey = []
    aggregate_wrongChord = []
    error_log = []

    # Get ground truth files
    gt_files = sorted([f for f in os.listdir(GT_TXT_DIR) if f.endswith(".txt")])

    for fname in gt_files:
        base = fname[:-4]
        gt_path = os.path.join(GT_TXT_DIR, fname)

        # Prediction file is named "{base}-analysis.txt"
        pred_name = f"{base}-analysis.txt"
        pred_path = os.path.join(pred_dir, pred_name)

        if not os.path.exists(pred_path):
            error_log.append(f"{base}: Missing prediction {pred_name}")
            continue

        print(f"Comparing {base}...")

        try:
            compare.analyzed_file1 = converter.parse(gt_path, format='romantext')
        except Exception as e:
            error_log.append(f"{base}: GT parsing error - {str(e)[:80]}")
            continue

        try:
            compare.analyzed_file2 = converter.parse(pred_path, format='romantext')
        except Exception as e:
            error_log.append(f"{base}: PRED parsing error - {str(e)[:80]}")
            continue

        try:
            pct, wrongKey, wrongChord = compare.dochorale(base)
            aggregate_pcts.append(pct)
            aggregate_wrongKey.append(wrongKey)
            aggregate_wrongChord.append(wrongChord)
            print(f"  ✓ {pct:.2f}%")
        except Exception as e:
            error_log.append(f"{base}: Comparison failed - {str(e)[:100]}")
            print(f"  ✗ Comparison failed")

    # Print results
    print(f"\n{'='*70}")
    print(f"{model_name.upper()} RESULTS")
    print(f"{'='*70}")

    if len(aggregate_pcts) == 0:
        print("No successful comparisons.\n")
        return

    overall_pct = sum(aggregate_pcts) / len(aggregate_pcts)
    overall_wrongKey = sum(aggregate_wrongKey) / len(aggregate_wrongKey)
    overall_wrongChord = sum(aggregate_wrongChord) / len(aggregate_wrongChord)

    total_eighths = (
        compare.totalCorrectEighths +
        compare.keyErrors +
        compare.chordErrors
    )

    if total_eighths > 0:
        overall_accuracy = 100 * compare.totalCorrectEighths / total_eighths
        key_accuracy = 100 * (1 - compare.keyErrors / total_eighths)
        chord_accuracy = 100 * (1 - compare.chordErrors / total_eighths)
        inversion_pct = 100 * (compare.inversionErrors / total_eighths)
    else:
        overall_accuracy = key_accuracy = chord_accuracy = inversion_pct = 0.0

    print(f"Successful files:                {len(aggregate_pcts)}")
    print(f"Avg correctness (simple avg):    {overall_pct:.2f}%")
    print(f"Avg wrong key (simple avg):      {overall_wrongKey:.2f}%")
    print(f"Avg wrong chord (simple avg):    {overall_wrongChord:.2f}%")
    print("-" * 70)
    print(f"Overall Accuracy (eighths):      {overall_accuracy:6.2f}%")
    print(f"Key Accuracy (eighths):          {key_accuracy:6.2f}%")
    print(f"Chord Accuracy (eighths):        {chord_accuracy:6.2f}%")
    print(f"Inversion Error %:               {inversion_pct:6.2f}%")
    print(f"Wrong-key-but-right-chord:       {compare.wrongKeyRightChord}")
    print(f"Total Eighth Notes Compared:     {total_eighths}")

    print("\n" + "-" * 70)
    print("Top Confusions")
    print("-" * 70)

    if compare.confusionMatrix:
        sorted_conf = sorted(
            compare.confusionMatrix.items(),
            key=lambda x: -x[1]
        )
        for label, count in sorted_conf[:15]:
            print(f"  {label:<30} {count} times")

    if error_log:
        print("\n" + "-" * 70)
        print("Errors:")
        for err in error_log:
            print(f"  • {err}")

    print()


# ===============================================================
# MAIN
# ===============================================================

def main():
    print("\n" + "="*70)
    print("FULL CHORDGNN EVALUATION PIPELINE")
    print("="*70)

    # Step 1: Analyze with pretrained model
    analyze_with_model("pretrained", PRETRAINED_CKPT, PRETRAINED_XML_DIR)

    # Step 2: Analyze with finetuned model
    analyze_with_model("finetuned", FINETUNED_CKPT, FINETUNED_XML_DIR)

    # Step 3: Convert pretrained to text
    convert_to_text(PRETRAINED_XML_DIR, PRETRAINED_TXT_DIR, "pretrained")

    # Step 4: Convert finetuned to text
    convert_to_text(FINETUNED_XML_DIR, FINETUNED_TXT_DIR, "finetuned")

    # Step 5: Run comparison for pretrained
    run_comparison(PRETRAINED_TXT_DIR, "pretrained")

    # Step 6: Run comparison for finetuned
    run_comparison(FINETUNED_TXT_DIR, "finetuned")

    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"Pretrained results in: {PRETRAINED_TXT_DIR}")
    print(f"Finetuned results in: {FINETUNED_TXT_DIR}")
    print()


if __name__ == "__main__":
    main()
