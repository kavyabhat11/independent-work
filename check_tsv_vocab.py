#!/usr/bin/env python3
"""
Check which Roman numeral and quality labels in TSVs are not in 31-class vocab.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from chordgnn.utils.chord_representations_latest import COMMON_ROMAN_NUMERALS, CHORD_QUALITIES

def check_tsv_vocab(tsv_dir):
    """Check all TSVs for vocab compatibility."""
    vocab_rn = set(COMMON_ROMAN_NUMERALS)
    vocab_quality = set(CHORD_QUALITIES)

    tsv_path = Path(tsv_dir)
    all_tsvs = list(tsv_path.rglob("*.tsv"))

    print(f"Checking {len(all_tsvs)} TSV files...")
    print(f"RomanNumeral vocab: {len(vocab_rn)} labels")
    print(f"Quality vocab: {len(vocab_quality)} labels\n")

    all_invalid_rn = set()
    all_invalid_quality = set()
    files_with_issues = []
    nan_files_rn = []
    nan_files_quality = []

    for tsv_file in all_tsvs:
        try:
            df = pd.read_csv(tsv_file, sep='\t')
            has_issues = False

            # Check a_romanNumeral
            if 'a_romanNumeral' not in df.columns:
                print(f"⚠ {tsv_file.name}: No a_romanNumeral column")
                has_issues = True
            else:
                unique_labels = df['a_romanNumeral'].unique()

                # Separate NaNs from invalid labels
                has_nan = any(pd.isna(label) for label in unique_labels)
                invalid = [label for label in unique_labels
                          if not pd.isna(label) and label not in vocab_rn]

                if has_nan:
                    nan_files_rn.append(tsv_file.name)
                    has_issues = True
                    print(f"⚠ {tsv_file.name}: Contains NaN/missing values in a_romanNumeral")

                if invalid:
                    all_invalid_rn.update(invalid)
                    has_issues = True
                    print(f"✗ {tsv_file.name}: {len(invalid)} invalid romanNumeral labels")
                    for label in invalid:
                        print(f"    - '{label}'")

            # Check a_quality
            if 'a_quality' not in df.columns:
                print(f"⚠ {tsv_file.name}: No a_quality column")
                has_issues = True
            else:
                unique_qualities = df['a_quality'].unique()

                # Separate NaNs from invalid qualities
                has_nan = any(pd.isna(q) for q in unique_qualities)
                invalid = [q for q in unique_qualities
                          if not pd.isna(q) and q not in vocab_quality]

                if has_nan:
                    nan_files_quality.append(tsv_file.name)
                    has_issues = True
                    print(f"⚠ {tsv_file.name}: Contains NaN/missing values in a_quality")

                if invalid:
                    all_invalid_quality.update(invalid)
                    has_issues = True
                    print(f"✗ {tsv_file.name}: {len(invalid)} invalid quality labels")
                    for label in invalid:
                        print(f"    - '{label}'")

            if has_issues:
                files_with_issues.append(tsv_file.name)

        except Exception as e:
            print(f"ERROR reading {tsv_file.name}: {e}")
            files_with_issues.append(tsv_file.name)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Files checked: {len(all_tsvs)}")
    print(f"Files with issues: {len(files_with_issues)}")

    if nan_files_rn:
        print(f"\nFiles with NaN romanNumeral values: {len(nan_files_rn)}")
        for fname in nan_files_rn[:5]:
            print(f"  - {fname}")
        if len(nan_files_rn) > 5:
            print(f"  ... and {len(nan_files_rn) - 5} more")

    if nan_files_quality:
        print(f"\nFiles with NaN quality values: {len(nan_files_quality)}")
        for fname in nan_files_quality[:5]:
            print(f"  - {fname}")
        if len(nan_files_quality) > 5:
            print(f"  ... and {len(nan_files_quality) - 5} more")

    if all_invalid_rn:
        print(f"\nInvalid romanNumeral labels ({len(all_invalid_rn)} unique):")
        for label in sorted(all_invalid_rn, key=str):
            print(f"  - '{label}'")

    if all_invalid_quality:
        print(f"\nInvalid quality labels ({len(all_invalid_quality)} unique):")
        for label in sorted(all_invalid_quality, key=str):
            print(f"  - '{label}'")

    if not files_with_issues:
        print("\n✓ All labels are valid!")

    return len(files_with_issues) == 0

if __name__ == "__main__":
    tsv_dir = sys.argv[1] if len(sys.argv) > 1 else "./mozart_dataset"
    check_tsv_vocab(tsv_dir)
