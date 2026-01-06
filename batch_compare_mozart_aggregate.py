#!/usr/bin/env python3
import os
from music21 import converter
import compare

# ===============================================================
# DEFINE ALL GLOBALS HERE SO compare.dochorale NEVER BREAKS
# ===============================================================
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

# analyzed_file1 and analyzed_file2 are set each iteration
compare.analyzed_file1 = None
compare.analyzed_file2 = None


# ===============================================================
# PATHS
# ===============================================================
GT_DIR = "data/mozart_test"          # ground truth .txt files
PRED_DIR = "txt_analysis/mozart"   # predictions

# results
aggregate_pcts = []
aggregate_wrongKey = []
aggregate_wrongChord = []
error_log = []


# ===============================================================
# PROCESS EACH FILE
# ===============================================================
for fname in sorted(os.listdir(GT_DIR)):
    if not fname.endswith(".txt"):
        continue

    base = fname[:-4]    # remove .txt
    gt_path = os.path.join(GT_DIR, fname)
    pred_name = base + "-analysis.txt"      # "K331-1-analysis.txt"

    pred_path = os.path.join(PRED_DIR, pred_name)  # must match name

    print(f"\nComparing {base}.txt <--> {base}-analysis.txt")

    # -----------------------------------------------------------
    # LOAD PREDICTED FILE
    # -----------------------------------------------------------
    if not os.path.exists(pred_path):
        error_log.append(f"{base}: Missing predicted {pred_path}")
        print("  -> missing pred file, skipping")
        continue

    try:
        compare.analyzed_file1 = converter.parse(gt_path, format='romantext')
    except Exception as e:
        error_log.append(f"{base}: Could not parse GT ({e})")
        print("  -> GT load error, skipping")
        continue

    try:
        compare.analyzed_file2 = converter.parse(pred_path, format='romantext')
    except Exception as e:
        error_log.append(f"{base}: Could not parse PRED ({e})")
        print("  -> prediction load error, skipping")
        continue

    # -----------------------------------------------------------
    # RUN COMPARISON
    # -----------------------------------------------------------
    try:
        pct, wrongKey, wrongChord = compare.dochorale(base)
        aggregate_pcts.append(pct)
        aggregate_wrongKey.append(wrongKey)
        aggregate_wrongChord.append(wrongChord)
        print(f"  ✓ success: {pct:.2f}%")
    except Exception as e:
        error_log.append(f"{base}: EXCEPTION → {e}")
        print("  -> comparison crash (logged), continuing")
        continue


# ===============================================================
# FINAL DETAILED REPORT (compare.print_report-style)
# ===============================================================
print("\n" + "="*70)
print("FINAL MOZART AGGREGATE RESULTS (DETAILED)")
print("="*70)

if len(aggregate_pcts) == 0:
    print("No successful comparisons.\n")
else:
    # ---- basic aggregate stats ----
    overall_pct = sum(aggregate_pcts) / len(aggregate_pcts)
    overall_wrongKey = sum(aggregate_wrongKey) / len(aggregate_wrongKey)
    overall_wrongChord = sum(aggregate_wrongChord) / len(aggregate_wrongChord)

    # pull globals from compare.py
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

    # ---- PRINT section ----
    print(f"Successful files:                {len(aggregate_pcts)}")
    print(f"Avg correctness (simple avg):    {overall_pct:.2f}%")
    print(f"Avg wrong key (simple avg):      {overall_wrongKey:.2f}%")
    print(f"Avg wrong chord (simple avg):    {overall_wrongChord:.2f}%")
    print("-"*70)

    print(f"Overall Accuracy (eighths):      {overall_accuracy:6.2f}%")
    print(f"Key Accuracy (eighths):          {key_accuracy:6.2f}%")
    print(f"Chord Accuracy (eighths):        {chord_accuracy:6.2f}%")
    print(f"Inversion Error %:               {inversion_pct:6.2f}%")
    print(f"Wrong-key-but-right-chord:       {compare.wrongKeyRightChord}")
    print(f"Total Eighth Notes Compared:     {total_eighths}")

    print("\n" + "-"*70)
    print("Top Confusions (Across All Files)")
    print("-"*70)

    if compare.confusionMatrix:
        sorted_conf = sorted(
            compare.confusionMatrix.items(),
            key=lambda x: -x[1]
        )
        top = sorted_conf[:15]

        for label, count in top:
            print(f"  {label:<30} {count} times")
    else:
        print("  No confusions recorded.")

# ===============================================================
# ERRORS
# ===============================================================
print("\n" + "="*70)
print("ERROR LOG")
print("="*70)
if not error_log:
    print("No errors.")
else:
    for err in error_log:
        print(" •", err)
