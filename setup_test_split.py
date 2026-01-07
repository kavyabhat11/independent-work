#!/usr/bin/env python3
"""
Set up the test split directory based on mozart_dataset_final/test/ TSVs.
This copies only the test set XML and TXT files to a clean directory.
"""

import os
import shutil
import glob

# ===============================================================
# CONFIGURATION
# ===============================================================

TSV_TEST_DIR = "mozart_dataset_final/test"  # TSVs define the test split
SOURCE_DIR = "data/mozart"                   # All XML/TXT files
TEST_DIR = "data/test_split"                 # Clean test directory

# ===============================================================
# MAIN
# ===============================================================

def main():
    print("\n" + "="*70)
    print("SETTING UP TEST SPLIT DIRECTORY")
    print("="*70)

    # Get list of test file basenames from TSVs
    tsv_files = glob.glob(os.path.join(TSV_TEST_DIR, "*.tsv"))
    if not tsv_files:
        print(f"ERROR: No TSV files found in {TSV_TEST_DIR}")
        return

    test_basenames = sorted({os.path.splitext(os.path.basename(f))[0] for f in tsv_files})

    print(f"\nTest set size: {len(test_basenames)} files")
    print(f"Examples: {test_basenames[:5]}")

    # Create clean test directory
    if os.path.exists(TEST_DIR):
        print(f"\nRemoving existing {TEST_DIR}...")
        shutil.rmtree(TEST_DIR)

    os.makedirs(TEST_DIR, exist_ok=True)
    print(f"Created {TEST_DIR}/")

    # Copy XML and TXT files
    print(f"\nCopying files from {SOURCE_DIR}...")

    copied_xml = 0
    copied_txt = 0
    missing_xml = []
    missing_txt = []

    for basename in test_basenames:
        # Copy XML
        src_xml = os.path.join(SOURCE_DIR, f"{basename}.xml")
        if os.path.exists(src_xml):
            shutil.copy2(src_xml, os.path.join(TEST_DIR, f"{basename}.xml"))
            copied_xml += 1
        else:
            missing_xml.append(basename)

        # Copy TXT (ground truth)
        src_txt = os.path.join(SOURCE_DIR, f"{basename}.txt")
        if os.path.exists(src_txt):
            shutil.copy2(src_txt, os.path.join(TEST_DIR, f"{basename}.txt"))
            copied_txt += 1
        else:
            missing_txt.append(basename)

    print(f"\nResults:")
    print(f"  XML files copied: {copied_xml}/{len(test_basenames)}")
    print(f"  TXT files copied: {copied_txt}/{len(test_basenames)}")

    if missing_xml:
        print(f"\n  WARNING: Missing {len(missing_xml)} XML files:")
        for name in missing_xml[:10]:
            print(f"    - {name}.xml")
        if len(missing_xml) > 10:
            print(f"    ... and {len(missing_xml) - 10} more")

    if missing_txt:
        print(f"\n  WARNING: Missing {len(missing_txt)} TXT files:")
        for name in missing_txt[:10]:
            print(f"    - {name}.txt")
        if len(missing_txt) > 10:
            print(f"    ... and {len(missing_txt) - 10} more")

    print(f"\n{'='*70}")
    print("SETUP COMPLETE")
    print(f"{'='*70}")
    print(f"Test directory: {TEST_DIR}/")
    print(f"Ready to run evaluation script.\n")

if __name__ == "__main__":
    main()
