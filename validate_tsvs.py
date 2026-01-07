#!/usr/bin/env python3
"""
Validate TSV conversion to ensure no oversimplification.
Checks key variety, voice leading, secondary dominants, etc.
"""

import pandas as pd
from pathlib import Path
from collections import Counter

def validate_tsv_quality(tsv_dir):
    """Check that TSVs have rich, varied data (not oversimplified)."""

    tsv_files = list(Path(tsv_dir).glob("*.tsv"))
    print(f"Validating {len(tsv_files)} TSV files...\n")

    # Aggregate statistics across all files
    all_local_keys = []
    all_ton_keys = []
    all_roman_numerals = []
    all_soprano = []
    all_alto = []
    all_tenor = []
    all_bass = []
    all_degree2 = []
    all_harmonic_rhythm = []

    issues = []

    for tsv_file in tsv_files[:5]:  # Check first 5 files in detail
        try:
            df = pd.read_csv(tsv_file, sep='\t')

            # Check for required columns
            required_cols = ['a_localKey', 'a_tonicizedKey', 'a_romanNumeral',
                           'a_soprano', 'a_alto', 'a_tenor', 'a_bass',
                           'a_degree2', 'a_harmonicRhythm']

            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                issues.append(f"{tsv_file.name}: Missing columns {missing}")
                continue

            # Check key variety
            local_keys = df['a_localKey'].unique()
            ton_keys = df['a_tonicizedKey'].unique()

            if len(local_keys) == 1 and local_keys[0] == 'C' and len(df) > 100:
                issues.append(f"{tsv_file.name}: Only key 'C' found (suspicious for long piece)")

            # Check voice leading variety
            soprano_unique = df['a_soprano'].nunique()
            alto_unique = df['a_alto'].nunique()
            tenor_unique = df['a_tenor'].nunique()
            bass_unique = df['a_bass'].nunique()

            if soprano_unique == 1 or alto_unique == 1 or tenor_unique == 1 or bass_unique == 1:
                issues.append(f"{tsv_file.name}: Voice leading has no variety (all same note)")

            # Check if voices are all identical (bad)
            same_voices = (df['a_soprano'] == df['a_alto']).all() and \
                         (df['a_alto'] == df['a_tenor']).all() and \
                         (df['a_tenor'] == df['a_bass']).all()
            if same_voices:
                issues.append(f"{tsv_file.name}: All voices identical (SATB all same)")

            # Aggregate for statistics
            all_local_keys.extend(df['a_localKey'].tolist())
            all_ton_keys.extend(df['a_tonicizedKey'].tolist())
            all_roman_numerals.extend(df['a_romanNumeral'].tolist())
            all_soprano.extend(df['a_soprano'].tolist())
            all_alto.extend(df['a_alto'].tolist())
            all_tenor.extend(df['a_tenor'].tolist())
            all_bass.extend(df['a_bass'].tolist())
            all_degree2.extend(df['a_degree2'].tolist())
            all_harmonic_rhythm.extend(df['a_harmonicRhythm'].tolist())

        except Exception as e:
            issues.append(f"{tsv_file.name}: ERROR reading file - {e}")

    # Print validation results
    print("="*70)
    print("VALIDATION RESULTS")
    print("="*70)

    if issues:
        print("\n⚠ ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✓ No critical issues found")

    # Print statistics
    print("\n" + "="*70)
    print("DATA RICHNESS STATISTICS (first 5 files)")
    print("="*70)

    print(f"\nLocal Keys:")
    key_counts = Counter(all_local_keys)
    for key, count in key_counts.most_common(10):
        print(f"  {key}: {count}")
    print(f"  → Total unique keys: {len(key_counts)}")

    print(f"\nTonicized Keys:")
    ton_counts = Counter(all_ton_keys)
    for key, count in ton_counts.most_common(10):
        print(f"  {key}: {count}")
    print(f"  → Total unique keys: {len(ton_counts)}")

    print(f"\nRoman Numerals:")
    rn_counts = Counter(all_roman_numerals)
    for rn, count in rn_counts.most_common(10):
        print(f"  {rn}: {count}")
    print(f"  → Total unique RNs: {len(rn_counts)}")

    print(f"\nVoice Leading Variety:")
    print(f"  Unique soprano notes: {len(set(all_soprano))}")
    print(f"  Unique alto notes: {len(set(all_alto))}")
    print(f"  Unique tenor notes: {len(set(all_tenor))}")
    print(f"  Unique bass notes: {len(set(all_bass))}")

    print(f"\nSecondary Dominants (degree2):")
    deg2_counts = Counter(all_degree2)
    for deg, count in deg2_counts.most_common(10):
        print(f"  {deg}: {count}")
    non_none = sum(1 for d in all_degree2 if d != 'None')
    if len(all_degree2) > 0:
        print(f"  → Applied chords (degree2 != None): {non_none}/{len(all_degree2)} ({100*non_none/len(all_degree2):.1f}%)")
    else:
        print(f"  → Applied chords (degree2 != None): 0/0")

    print(f"\nHarmonic Rhythm:")
    hr_counts = Counter(all_harmonic_rhythm)
    for hr, count in sorted(hr_counts.items()):
        print(f"  Category {hr}: {count}")
    print(f"  → Variety (not all 0): {len(hr_counts) > 1}")

    print("\n" + "="*70)
    print("SAMPLE DATA FROM FIRST FILE")
    print("="*70)

    # Show sample rows from first file
    sample_file = tsv_files[0]
    df_sample = pd.read_csv(sample_file, sep='\t')
    print(f"\n{sample_file.name} (showing rows 10-15):\n")

    cols_to_show = ['a_romanNumeral', 'a_localKey', 'a_tonicizedKey',
                    'a_soprano', 'a_alto', 'a_tenor', 'a_bass', 'a_degree2']

    print(df_sample[cols_to_show].iloc[10:16].to_string(index=False))

    print("\n" + "="*70)
    if not issues:
        print("✓ TSVs LOOK GOOD - Rich, varied data present")
    else:
        print("⚠ REVIEW ISSUES ABOVE")
    print("="*70)

if __name__ == "__main__":
    validate_tsv_quality("./mozart_tsv_validation")
