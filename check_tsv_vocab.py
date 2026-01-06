#!/usr/bin/env python3
"""
Check which Roman numeral labels in TSVs are not in 31-class vocab.
"""

import pandas as pd
import sys
from pathlib import Path
from chordgnn.utils.chord_representations_latest import COMMON_ROMAN_NUMERALS

def check_tsv_vocab(tsv_dir):
    """Check all TSVs for vocab compatibility."""
    vocab_31 = set(COMMON_ROMAN_NUMERALS)

    tsv_path = Path(tsv_dir)
    all_tsvs = list(tsv_path.rglob("*.tsv"))

    print(f"Checking {len(all_tsvs)} TSV files...")
    print(f"31-class vocab has {len(vocab_31)} labels\n")

    all_invalid = set()
    files_with_issues = []

    for tsv_file in all_tsvs:
        try:
            df = pd.read_csv(tsv_file, sep='\t')

            if 'a_romanNumeral' not in df.columns:
                print(f"⚠ {tsv_file.name}: No a_romanNumeral column")
                continue

            unique_labels = df['a_romanNumeral'].unique()
            invalid = [label for label in unique_labels if label not in vocab_31]

            if invalid:
                all_invalid.update(invalid)
                files_with_issues.append(tsv_file.name)
                print(f"✗ {tsv_file.name}: {len(invalid)} invalid labels")
                for label in invalid:
                    print(f"    - '{label}'")

        except Exception as e:
            print(f"ERROR reading {tsv_file.name}: {e}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Files checked: {len(all_tsvs)}")
    print(f"Files with issues: {len(files_with_issues)}")

    if all_invalid:
        print(f"\nAll invalid labels found ({len(all_invalid)} unique):")
        for label in sorted(all_invalid):
            print(f"  - '{label}'")
    else:
        print("\n✓ All labels are valid 31-class vocab!")

    return len(all_invalid) == 0

if __name__ == "__main__":
    tsv_dir = sys.argv[1] if len(sys.argv) > 1 else "./mozart_dataset"
    check_tsv_vocab(tsv_dir)
