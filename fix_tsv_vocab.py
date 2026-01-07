#!/usr/bin/env python3
"""
Fix vocabulary issues in existing TSVs to match ChordGNN vocabularies.
"""

import pandas as pd
import sys
import ast
from pathlib import Path
from music21 import roman as m21roman
from music21 import key as m21key

from chordgnn.utils.chord_representations_latest import (
    RomanNumeral31, ChordQuality11, LocalKey38, TonicizedKey38, PitchClassSet121
)

VALID_ROMAN_NUMERALS = set(RomanNumeral31.classList)
VALID_QUALITIES = set(ChordQuality11.classList)
VALID_KEYS = set(LocalKey38.classList)
VALID_PCSETS = set(PitchClassSet121.classList)


def normalize_key(key_str):
    """Normalize key notation: Ab -> A-, Bb -> B-, etc."""
    if not key_str or key_str == '--':
        return None

    # Replace 'b' with '-' for flats
    normalized = key_str.replace('b', '-')

    # Handle double flats (already --)
    normalized = normalized.replace('--', '-')

    return normalized


def fix_roman_numeral(rn_str):
    """Fix invalid Roman numerals."""
    if rn_str in VALID_ROMAN_NUMERALS:
        return rn_str

    # Common fixes
    fixes = {
        '#i7': 'I7',      # Raised tonic seventh -> just I7
        '#iio7': 'viio7', # Raised ii diminished seventh -> viio7
        'v7': 'V7',       # Lowercase V -> uppercase
        'vii': 'viio',    # vii triad -> viio diminished
        ':||': 'I',       # Invalid symbol -> I
        'viio7iv': 'viio7', # Remove extra suffix
    }

    fixed = fixes.get(rn_str, 'I')  # Default to I if unknown

    if fixed not in VALID_ROMAN_NUMERALS:
        return 'I'

    return fixed


def find_closest_pcset(pcset_tuple):
    """Find the closest valid pcset in the vocabulary."""
    if pcset_tuple in VALID_PCSETS:
        return pcset_tuple

    # Try to find a pcset with the same cardinality
    cardinality = len(pcset_tuple)
    candidates = [p for p in VALID_PCSETS if len(p) == cardinality]

    if not candidates:
        # Fallback to major triad
        return (0, 4, 7)

    # Find closest by Hamming distance
    def hamming_dist(p1, p2):
        s1 = set(p1)
        s2 = set(p2)
        return len(s1.symmetric_difference(s2))

    closest = min(candidates, key=lambda p: hamming_dist(p, pcset_tuple))
    return closest


def fix_quality(quality_str, pcset_tuple, roman_str):
    """Fix quality values, deriving from pcset if needed."""
    if quality_str in VALID_QUALITIES:
        return quality_str

    # If quality is 'unknown', try to infer from pcset cardinality and Roman numeral
    if quality_str == 'unknown':
        card = len(pcset_tuple)

        # Check Roman numeral for hints
        rn_lower = roman_str.lower()

        # Seventh chords
        if card == 4 or '7' in roman_str:
            if 'o7' in rn_lower or 'dim7' in rn_lower:
                return 'dim7'
            elif '/o7' in rn_lower or 'hdim7' in rn_lower:
                return 'hdim7'
            elif roman_str.isupper() or roman_str.startswith('V'):
                return '7'  # Dominant seventh
            elif 'maj7' in rn_lower:
                return 'maj7'
            else:
                return 'min7'

        # Triads
        elif card == 3:
            if 'o' in rn_lower or 'dim' in rn_lower:
                return 'dim'
            elif '+' in roman_str or 'aug' in rn_lower:
                return 'aug'
            elif roman_str[0].isupper():
                return 'maj'
            else:
                return 'min'

        # Default to major
        return 'maj'

    return quality_str


def fix_tsv(tsv_path, dry_run=False):
    """Fix vocabulary issues in a single TSV file."""
    df = pd.read_csv(tsv_path, sep='\t', dtype={'a_degree1': str, 'a_degree2': str})

    changes = []

    for idx, row in df.iterrows():
        # Fix keys (Ab -> A-)
        if 'a_localKey' in df.columns:
            old_key = row['a_localKey']
            new_key = normalize_key(old_key)
            if new_key and new_key != old_key:
                if new_key in VALID_KEYS:
                    df.at[idx, 'a_localKey'] = new_key
                    changes.append(f"  Row {idx}: localKey '{old_key}' -> '{new_key}'")

        if 'a_tonicizedKey' in df.columns:
            old_tonkey = row['a_tonicizedKey']

            # Fix '--' to match localKey
            if old_tonkey == '--':
                new_tonkey = row['a_localKey']
                df.at[idx, 'a_tonicizedKey'] = new_tonkey
                changes.append(f"  Row {idx}: tonicizedKey '--' -> '{new_tonkey}'")
            else:
                new_tonkey = normalize_key(old_tonkey)
                if new_tonkey and new_tonkey != old_tonkey:
                    if new_tonkey in VALID_KEYS:
                        df.at[idx, 'a_tonicizedKey'] = new_tonkey
                        changes.append(f"  Row {idx}: tonicizedKey '{old_tonkey}' -> '{new_tonkey}'")

        # Fix Roman numerals
        if 'a_romanNumeral' in df.columns:
            old_rn = row['a_romanNumeral']
            if old_rn not in VALID_ROMAN_NUMERALS:
                new_rn = fix_roman_numeral(old_rn)
                df.at[idx, 'a_romanNumeral'] = new_rn
                changes.append(f"  Row {idx}: romanNumeral '{old_rn}' -> '{new_rn}'")

        # Fix pcset
        if 'a_pcset' in df.columns:
            pcset_str = row['a_pcset']
            try:
                pcset_tuple = ast.literal_eval(pcset_str)
                if pcset_tuple not in VALID_PCSETS:
                    closest = find_closest_pcset(pcset_tuple)
                    df.at[idx, 'a_pcset'] = repr(closest)
                    changes.append(f"  Row {idx}: pcset {pcset_tuple} -> {closest}")
            except:
                # Invalid format, use default
                df.at[idx, 'a_pcset'] = repr((0, 4, 7))
                changes.append(f"  Row {idx}: pcset INVALID -> (0, 4, 7)")

        # Fix quality
        if 'a_quality' in df.columns and 'a_pcset' in df.columns and 'a_romanNumeral' in df.columns:
            old_quality = row['a_quality']
            if old_quality not in VALID_QUALITIES:
                try:
                    pcset_tuple = ast.literal_eval(df.at[idx, 'a_pcset'])
                    new_quality = fix_quality(old_quality, pcset_tuple, df.at[idx, 'a_romanNumeral'])
                    df.at[idx, 'a_quality'] = new_quality
                    changes.append(f"  Row {idx}: quality '{old_quality}' -> '{new_quality}'")
                except:
                    df.at[idx, 'a_quality'] = 'maj'
                    changes.append(f"  Row {idx}: quality '{old_quality}' -> 'maj'")

    if changes:
        print(f"✓ {tsv_path.name}: {len(changes)} fixes")
        if not dry_run:
            df.to_csv(tsv_path, sep='\t', index=False)

    return len(changes)


def main():
    tsv_dir = sys.argv[1] if len(sys.argv) > 1 else "./mozart_dataset_final_CORRECTED"
    dry_run = '--dry-run' in sys.argv

    tsv_path = Path(tsv_dir)
    all_tsvs = list(tsv_path.rglob("*.tsv"))

    print(f"Fixing {len(all_tsvs)} TSV files...")
    if dry_run:
        print("DRY RUN - no files will be modified\n")

    total_changes = 0
    for tsv_file in sorted(all_tsvs):
        changes = fix_tsv(tsv_file, dry_run=dry_run)
        total_changes += changes

    print(f"\n{'='*60}")
    print(f"Total changes: {total_changes}")
    if dry_run:
        print("Run without --dry-run to apply fixes")
    else:
        print("All fixes applied!")


if __name__ == "__main__":
    main()
