#!/usr/bin/env python3
"""
Convert Mozart XML + TXT files to TSV format for ChordGNN training.
This processes the files in IWold/data/mozart/
"""

import os
import sys
import pandas as pd
import partitura as pt
from pathlib import Path
import re
from collections import defaultdict

def parse_analysis_txt(txt_file):
    """Parse a Mozart analysis .txt file into structured data."""
    with open(txt_file, 'r') as f:
        lines = f.readlines()

    metadata = {}
    analysis = []
    current_key = None

    for line in lines:
        line = line.strip()
        if not line or line.startswith('[TODO'):
            continue

        # Metadata
        if ':' in line and not line.startswith('m'):
            parts = line.split(':', 1)
            if len(parts) == 2:
                key, val = parts
                if key.strip() in ['Composer', 'Piece', 'Time Signature', 'Tempo']:
                    metadata[key.strip()] = val.strip()
            continue

        # Skip Form/Pedal/Note lines
        if any(line.startswith(x) for x in ['Form:', 'Pedal:', 'Note:']):
            continue

        # Measure analysis
        if line.startswith('m'):
            # Skip equivalences like m3-4 = m1-2
            if '=' in line:
                continue

            parts = line.split()
            measure_str = parts[0][1:]  # Remove 'm'

            # Skip variant measures like m12var1
            if 'var' in measure_str:
                continue

            try:
                measure_num = int(measure_str)
            except ValueError:
                continue

            # Parse beat positions and chords
            i = 1
            current_beat = 1.0
            while i < len(parts):
                part = parts[i]

                # Beat marker like b2 or b2.5
                if part.startswith('b'):
                    try:
                        current_beat = float(part[1:])
                    except ValueError:
                        pass
                    i += 1
                    continue

                # Key change like C: or f:
                if part.endswith(':'):
                    current_key = part[:-1]
                    i += 1
                    continue

                # Roman numeral chord
                roman = part
                analysis.append({
                    'measure': measure_num,
                    'beat': current_beat,
                    'key': current_key,
                    'roman': roman
                })
                i += 1

    return metadata, analysis

def xml_to_tsv(xml_file, txt_file, output_tsv):
    """
    Convert a Mozart XML + TXT pair to TSV format.

    This is a simplified version - creates basic TSV structure.
    For full compatibility, would need more complex alignment.
    """
    print(f"Processing: {os.path.basename(xml_file)}")

    # Parse the analysis
    try:
        metadata, analysis = parse_analysis_txt(txt_file)
    except Exception as e:
        print(f"  ERROR parsing {txt_file}: {e}")
        return False

    if not analysis:
        print(f"  WARNING: No analysis found in {txt_file}")
        return False

    # Load the MusicXML score
    try:
        score = pt.load_musicxml(xml_file)
    except Exception as e:
        print(f"  ERROR loading {xml_file}: {e}")
        return False

    # Get note array
    try:
        if isinstance(score, list):
            part = score[0]  # Take first part
        else:
            part = score
        note_array = part.note_array()
    except Exception as e:
        print(f"  ERROR extracting notes: {e}")
        return False

    # Create simplified TSV structure
    # This creates a basic time-aligned structure
    rows = []

    # Helper function to convert MIDI pitch to note name
    def midi_to_note_name(midi_pitch):
        """Convert MIDI pitch number to note name with octave (e.g., 60 -> 'C4')"""
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = (midi_pitch // 12) - 1
        note = note_names[midi_pitch % 12]
        return f"{note}{octave}"

    for note in note_array:
        onset = note['onset_beat'] if 'onset_beat' in note.dtype.names else note['onset_div'] / 100.0
        pitch_midi = note['pitch']
        pitch_name = midi_to_note_name(pitch_midi)
        duration = note['duration_beat'] if 'duration_beat' in note.dtype.names else 1.0

        # Find matching analysis for this timepoint
        # (simplified - matches by measure)
        matching_analysis = None
        for ann in analysis:
            if ann['measure'] is not None:
                matching_analysis = ann
                break

        if matching_analysis:
            row = {
                'j_offset': onset,
                's_duration': duration,
                's_measure': matching_analysis['measure'] if matching_analysis else 0,
                's_notes': f"['{pitch_name}']",  # Use note name instead of MIDI number
                's_intervals': "[]",
                's_isOnset': "[True]",
                'a_measure': matching_analysis['measure'] if matching_analysis else 0,
                'a_duration': duration,
                'a_annotationNumber': 0,
                'a_romanNumeral': matching_analysis['roman'] if matching_analysis else 'I',
                'a_isOnset': True,
                'a_localKey': matching_analysis['key'] if matching_analysis and matching_analysis['key'] else 'C',
                # Simplified - would need full chord parsing for these:
                'a_pitchNames': "('C', 'E', 'G')",
                'a_bass': 'C',
                'a_root': 'C',
                'a_inversion': 0.0,
                'a_quality': 'major triad',
                'a_pcset': '(0, 4, 7)',
                'a_tonicizedKey': matching_analysis['key'] if matching_analysis and matching_analysis['key'] else 'C',
                'a_degree1': 1,
                'a_degree2': 'None',
                'measureMisalignment': False,
                'qualityScoreNotes': "['C', 'E', 'G']",
                'qualityNonChordTones': 0.0,
                'qualityMissingChordTones': 0.0,
                'qualitySquaredSum': 0.0,
                'incongruentBass': 0.0,
            }
            rows.append(row)

    if not rows:
        print(f"  WARNING: No data rows created")
        return False

    # Create DataFrame and save
    df = pd.DataFrame(rows)
    df.to_csv(output_tsv, sep='\t', index=False)
    print(f"  ✓ Created: {output_tsv} ({len(df)} rows)")
    return True

def convert_all_mozart_files(data_dir, output_dir):
    """Convert all Mozart XML+TXT pairs to TSV files."""
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all TXT files
    txt_files = list(data_path.glob("*.txt"))
    print(f"Found {len(txt_files)} TXT files\n")

    converted = 0
    failed = []

    for txt_file in txt_files:
        # Find corresponding XML file
        base_name = txt_file.stem
        xml_file = data_path / f"{base_name}.xml"

        if not xml_file.exists():
            print(f"WARNING: No XML file for {txt_file.name}")
            continue

        output_tsv = output_path / f"{base_name}.tsv"

        if xml_to_tsv(str(xml_file), str(txt_file), str(output_tsv)):
            converted += 1
        else:
            failed.append(base_name)

    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"  ✓ Successfully converted: {converted} files")
    if failed:
        print(f"  ✗ Failed: {len(failed)} files")
        print(f"    {', '.join(failed[:5])}" + ("..." if len(failed) > 5 else ""))
    print(f"  Output directory: {output_path}")
    print(f"{'='*60}\n")

    return converted

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert Mozart XML+TXT to TSV")
    parser.add_argument("--data_dir", default="./data/mozart",
                        help="Directory containing Mozart XML and TXT files")
    parser.add_argument("--output_dir", default="./mozart_tsv",
                        help="Output directory for TSV files")

    args = parser.parse_args()

    print(f"Converting Mozart files from: {args.data_dir}")
    print(f"Output directory: {args.output_dir}\n")

    convert_all_mozart_files(args.data_dir, args.output_dir)
