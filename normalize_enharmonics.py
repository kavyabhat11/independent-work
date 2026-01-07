#!/usr/bin/env python3
"""
Normalize enharmonic spellings in TSVs to avoid double sharps/flats.
F## -> G, E## -> F#, etc.
"""

import pandas as pd
import sys
from pathlib import Path

# Enharmonic equivalents for double sharps and double flats
ENHARMONIC_MAP = {
    # Double sharps
    'C##': 'D',
    'D##': 'E',
    'E##': 'F#',
    'F##': 'G',
    'G##': 'A',
    'A##': 'B',
    'B##': 'C#',
    
    # Double flats
    'C--': 'B-',
    'D--': 'C',
    'E--': 'D',
    'F--': 'E-',
    'G--': 'F',
    'A--': 'G',
    'B--': 'A',
}

def normalize_note(note):
    """Normalize a single note name."""
    if note in ENHARMONIC_MAP:
        return ENHARMONIC_MAP[note]
    return note

def normalize_pitch_names(pitch_str):
    """Normalize a pitchNames tuple string."""
    if not pitch_str or pitch_str == "":
        return pitch_str
    
    try:
        # Parse the tuple
        pitch_str = pitch_str.strip()
        if not pitch_str.startswith('('):
            return pitch_str
        
        # Extract pitch names
        inner = pitch_str.strip('()').replace("'", "").replace('"', '')
        if not inner:
            return pitch_str
        
        pitches = [p.strip() for p in inner.split(',')]
        normalized = [normalize_note(p) for p in pitches]
        
        return repr(tuple(normalized))
    except:
        return pitch_str

def normalize_tsv(tsv_path):
    """Normalize enharmonics in a TSV file."""
    df = pd.read_csv(tsv_path, sep='\t', dtype={'a_degree1': str, 'a_degree2': str})
    
    changes = 0
    
    # Normalize bass, root, soprano, alto, tenor
    for col in ['a_bass', 'a_root', 'a_soprano', 'a_alto', 'a_tenor']:
        if col in df.columns:
            original = df[col].copy()
            df[col] = df[col].apply(normalize_note)
            changes += (df[col] != original).sum()
    
    # Normalize pitchNames
    if 'a_pitchNames' in df.columns:
        original = df['a_pitchNames'].copy()
        df['a_pitchNames'] = df['a_pitchNames'].apply(normalize_pitch_names)
        changes += (df['a_pitchNames'] != original).sum()
    
    if changes > 0:
        df.to_csv(tsv_path, sep='\t', index=False)
        print(f"✓ {tsv_path.name}: {changes} normalizations")
    
    return changes

def main():
    tsv_dir = sys.argv[1] if len(sys.argv) > 1 else "./mozart_dataset_final_CORRECTED"
    
    tsv_path = Path(tsv_dir)
    all_tsvs = list(tsv_path.rglob("*.tsv"))
    
    print(f"Normalizing enharmonics in {len(all_tsvs)} TSV files...\n")
    
    total_changes = 0
    for tsv_file in sorted(all_tsvs):
        changes = normalize_tsv(tsv_file)
        total_changes += changes
    
    print(f"\n{'='*60}")
    print(f"Total normalizations: {total_changes}")
    print("All double sharps/flats normalized!")

if __name__ == "__main__":
    main()
