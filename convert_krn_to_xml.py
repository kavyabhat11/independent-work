#!/usr/bin/env python3
"""
Convert Humdrum Kern (.krn) files to MusicXML (.xml) files.
"""

import os
import sys
from pathlib import Path

try:
    from music21 import converter
except ImportError:
    print("ERROR: music21 not found. Install with: pip install music21")
    sys.exit(1)


def convert_krn_to_xml(krn_file, xml_file):
    """Convert a single .krn file to .xml (MusicXML)."""
    try:
        # Parse kern file
        score = converter.parse(krn_file)

        # Write as MusicXML
        score.write('musicxml', fp=xml_file)

        return True
    except Exception as e:
        print(f"  ERROR converting {krn_file}: {e}")
        return False


def batch_convert(data_dir):
    """Convert all .krn files in a directory to .xml files."""

    data_path = Path(data_dir)
    krn_files = list(data_path.glob("*.krn"))

    print(f"Found {len(krn_files)} .krn files in {data_dir}\n")

    if not krn_files:
        print("No .krn files found!")
        return

    converted = 0
    failed = []

    for krn_file in krn_files:
        base = krn_file.stem
        xml_file = data_path / f"{base}.xml"

        # Skip if XML already exists
        if xml_file.exists():
            print(f"✓ {base}.xml already exists, skipping")
            continue

        print(f"Converting: {krn_file.name} → {xml_file.name}")

        if convert_krn_to_xml(str(krn_file), str(xml_file)):
            print(f"  ✓ Created: {xml_file.name}")
            converted += 1
        else:
            failed.append(krn_file.name)

    print(f"\n{'='*60}")
    print("CONVERSION COMPLETE")
    print(f"{'='*60}")
    print(f"Converted: {converted} files")
    if failed:
        print(f"Failed: {len(failed)} files")
        print(f"  {', '.join(failed[:5])}" + ("..." if len(failed) > 5 else ""))
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert Kern (.krn) to MusicXML (.xml)")
    parser.add_argument("--data_dir", default="./data/mozart",
                        help="Directory containing .krn files")

    args = parser.parse_args()

    batch_convert(args.data_dir)
