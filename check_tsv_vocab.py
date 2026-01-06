#!/usr/bin/env python3
"""
Check all TSV fields against their respective encoder vocabularies.
"""

import pandas as pd
import numpy as np
import sys
import ast
from pathlib import Path
from chordgnn.utils.chord_representations_latest import (
    RomanNumeral31, ChordQuality11, LocalKey38, TonicizedKey38,
    ChordRoot35, Inversion4, PrimaryDegree22, SecondaryDegree22,
    Bass35, HarmonicRhythm7, PitchClassSet121,
    Tenor35, Alto35, Soprano35
)

def check_tsv_vocab(tsv_dir):
    """Check all TSVs for vocab compatibility."""
    # Map encoder classes to their dfFeature (column name) and classList (vocab)
    encoders = [
        (RomanNumeral31, 'a_romanNumeral'),
        (ChordQuality11, 'a_quality'),
        (LocalKey38, 'a_localKey'),
        (TonicizedKey38, 'a_tonicizedKey'),
        (ChordRoot35, 'a_root'),
        (Bass35, 'a_bass'),
        (Inversion4, 'a_inversion'),
        (PrimaryDegree22, 'a_degree1'),
        (SecondaryDegree22, 'a_degree2'),
        (HarmonicRhythm7, 'a_harmonicRhythm'),
        (PitchClassSet121, 'a_pcset'),
        (Tenor35, 'a_tenor'),
        (Alto35, 'a_alto'),
        (Soprano35, 'a_soprano'),
    ]

    tsv_path = Path(tsv_dir)
    all_tsvs = list(tsv_path.rglob("*.tsv"))

    print(f"Checking {len(all_tsvs)} TSV files against {len(encoders)} encoder vocabularies...\n")

    # Print vocab sizes
    for encoder_class, col_name in encoders:
        print(f"  {col_name:20s}: {len(encoder_class.classList):3d} valid values")
    print()

    # Track issues per field
    invalid_by_field = {col: set() for _, col in encoders}
    nan_by_field = {col: [] for _, col in encoders}
    files_with_issues = []

    for tsv_file in all_tsvs:
        try:
            # Force degree columns to be read as strings (inversion is int)
            dtype_spec = {
                'a_degree1': str,
                'a_degree2': str,
            }
            df = pd.read_csv(tsv_file, sep='\t', dtype=dtype_spec, keep_default_na=False)
            has_issues = False

            # Check each encoder field
            for encoder_class, col_name in encoders:
                vocab = set(encoder_class.classList)

                if col_name not in df.columns:
                    print(f"⚠ {tsv_file.name}: Missing column '{col_name}'")
                    has_issues = True
                    continue

                unique_values = df[col_name].unique()

                # Separate NaNs from invalid values
                # Note: 'None' (string) is valid in PrimaryDegree22/SecondaryDegree22 vocabs
                has_nan = any(pd.isna(v) for v in unique_values)

                # Special handling for a_pcset: parse tuple strings
                if col_name == 'a_pcset':
                    invalid = []
                    for v in unique_values:
                        if pd.isna(v):
                            continue
                        try:
                            # Parse string like "(0, 4, 7)" to tuple (0, 4, 7)
                            parsed_tuple = ast.literal_eval(v)
                            if parsed_tuple not in vocab:
                                invalid.append(v)
                        except (ValueError, SyntaxError):
                            # Parsing failed - invalid format
                            invalid.append(v)
                else:
                    invalid = [v for v in unique_values
                              if not pd.isna(v) and v not in vocab]

                if has_nan:
                    nan_by_field[col_name].append(tsv_file.name)
                    has_issues = True
                    print(f"⚠ {tsv_file.name}: NaN values in '{col_name}'")

                if invalid:
                    invalid_by_field[col_name].update(invalid)
                    has_issues = True
                    print(f"✗ {tsv_file.name}: {len(invalid)} invalid '{col_name}' values")
                    for val in invalid[:5]:  # Show first 5
                        print(f"    - '{val}'")
                    if len(invalid) > 5:
                        print(f"    ... and {len(invalid) - 5} more")

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

    # Report NaN issues by field
    for col_name in [col for _, col in encoders]:
        if nan_by_field[col_name]:
            print(f"\nFiles with NaN '{col_name}' values: {len(nan_by_field[col_name])}")
            for fname in nan_by_field[col_name][:3]:
                print(f"  - {fname}")
            if len(nan_by_field[col_name]) > 3:
                print(f"  ... and {len(nan_by_field[col_name]) - 3} more")

    # Report invalid values by field
    for col_name in [col for _, col in encoders]:
        if invalid_by_field[col_name]:
            print(f"\nInvalid '{col_name}' values ({len(invalid_by_field[col_name])} unique):")
            for val in sorted(invalid_by_field[col_name], key=str)[:10]:
                print(f"  - '{val}'")
            if len(invalid_by_field[col_name]) > 10:
                print(f"  ... and {len(invalid_by_field[col_name]) - 10} more")

    if not files_with_issues:
        print("\n✓ All fields are valid!")

    return len(files_with_issues) == 0

if __name__ == "__main__":
    tsv_dir = sys.argv[1] if len(sys.argv) > 1 else "./mozart_dataset"
    check_tsv_vocab(tsv_dir)
