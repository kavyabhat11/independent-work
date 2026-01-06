#!/usr/bin/env python3
"""
Convert between MusicXML harmonic analysis and plain text (RomanText-like) format.

Usage:
    # MusicXML to text:
    python convert_musicxml_to_analysis.py <input.musicxml> [output.txt]

    # Text to MusicXML:
    python convert_musicxml_to_analysis.py <input.txt> [output.musicxml]

Notes:
- This script fixes a common bug: in MusicXML, <divisions> is "divisions per quarter note",
  not "divisions per beat". Beat conversion MUST account for the time signature's beat-type.
- It also supports time-signature changes per measure (reads <attributes><time> inside measures).
- A small safety valve clamps / skips beat markers that would exceed the current measure's beats.
"""

import xml.etree.ElementTree as ET
import sys
import re
from fractions import Fraction
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any


# ======================================================================
# FIGURED BASS TRANSFORMATIONS
# ======================================================================

def transform_figured_bass(harmony: str) -> str:
    """Convert MusicXML-style figured bass (64, 65, 43, etc.) into slash form."""
    harmony = harmony.replace('ø', '/o')

    patterns = [
        (r'(\w+?)65\b', r'\g<1>6/5'),
        (r'(\w+?)43\b', r'\g<1>4/3'),
        (r'(\w+?)42\b', r'\g<1>4/2'),
        (r'(\w+?)64\b', r'\g<1>6/4'),
        (r'(\w+?)32\b', r'\g<1>3/2'),
    ]

    for pattern, repl in patterns:
        harmony = re.sub(pattern, repl, harmony)

    return harmony


def reverse_transform_figured_bass(harmony: str) -> str:
    """Convert slash form (6/5, 4/3) back into MusicXML-style figured bass."""
    patterns = [
        (r'(\w+?)6/5\b', r'\g<1>65'),
        (r'(\w+?)4/3\b', r'\g<1>43'),
        (r'(\w+?)4/2\b', r'\g<1>42'),
        (r'(\w+?)6/4\b', r'\g<1>64'),
        (r'(\w+?)3/2\b', r'\g<1>32'),
    ]

    for pattern, repl in patterns:
        harmony = re.sub(pattern, repl, harmony)

    harmony = harmony.replace('/o', 'ø')
    return harmony


# ======================================================================
# BEAT / OFFSET HANDLERS (FIXED)
# ======================================================================

def duration_to_beat(
    duration: int,
    divisions_per_quarter: int,
    beat_type: int
) -> Tuple[int, Optional[Fraction]]:
    """
    Convert a MusicXML duration (in divisions) to (beat_number, subdivision).

    MusicXML:
      - <divisions> = divisions per quarter note
    Time signature:
      - beat_type = denominator (4=quarter, 8=eighth, 2=half, etc.)

    We interpret "bN" as beat N in the notated meter where the beat unit is beat_type.
    """
    # divisions per beat-unit = divisions_per_quarter * (quarter / beat_unit)
    # quarter = 4/beat_type of a beat unit
    divs_per_beat_unit = Fraction(divisions_per_quarter) * Fraction(4, beat_type)
    if divs_per_beat_unit <= 0:
        divs_per_beat_unit = Fraction(1)

    beat_offset = Fraction(duration, 1) / divs_per_beat_unit  # 0 means beat 1
    beat_number = int(beat_offset) + 1
    remainder = beat_offset - int(beat_offset)
    return beat_number, (remainder if remainder != 0 else None)


def format_beat(beat: int, subdivision: Optional[Fraction]) -> str:
    """
    Format beat markers. For subdivisions, emit a decimal that music21 romanText
    generally tolerates (e.g., b1.5).
    """
    if subdivision is None:
        return f"b{beat}"

    val = Fraction(beat, 1) + subdivision
    as_float = float(val)

    # Avoid "b2.0"
    if as_float.is_integer():
        return f"b{int(as_float)}"

    # Keep a reasonable number of decimals
    s = f"{as_float:.6f}".rstrip('0').rstrip('.')
    return f"b{s}"


def beat_to_duration(beat_str: str, divisions_per_quarter: int, beat_type: int) -> int:
    """
    Convert beat marker bX to a MusicXML duration (divisions) offset from measure start.
    """
    beat_value = float(beat_str.lstrip('b'))
    beat_offset = beat_value - 1.0
    divs_per_beat_unit = Fraction(divisions_per_quarter) * Fraction(4, beat_type)
    dur = Fraction(beat_offset) * divs_per_beat_unit
    return int(dur)


# ======================================================================
# PARSE ROMAN-TEXT (.txt) → structured dict
# ======================================================================

def parse_text_analysis(filepath: str) -> Dict[str, Any]:
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    metadata: Dict[str, str] = {}
    measures: List[Dict[str, Any]] = []
    comments: List[str] = []
    current_key: Optional[str] = None

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if '[TODO' in line:
            continue

        # Metadata line
        if ':' in line and not line.startswith('m'):
            k, v = line.split(':', 1)
            k = k.strip()
            v = v.strip()
            if k in ['Composer', 'Piece', 'Analyst', 'Proofreader', 'Tempo', 'Time Signature']:
                metadata[k] = v
            elif k in ['Form', 'Pedal', 'Note']:
                comments.append(line)
            continue

        # Measure line
        if line.startswith('m'):
            if '=' in line:  # equivalence
                comments.append(line)
                continue

            parts = line.split()
            if not parts:
                continue

            measure_num = parts[0][1:]
            if 'var' in measure_num:
                comments.append(line)
                continue

            harmonies: List[Tuple[str, Optional[str], str]] = []
            i = 1
            current_beat: Optional[str] = None

            while i < len(parts):
                part = parts[i]

                if part.startswith('b'):
                    current_beat = part
                    i += 1
                    continue
                if part.endswith(':'):
                    current_key = part[:-1]
                    i += 1
                    continue

                harmony = part
                if current_beat is None:
                    current_beat = 'b1'
                harmonies.append((current_beat, current_key, harmony))
                current_beat = None
                i += 1

            measures.append({
                "number": measure_num,
                "harmonies": harmonies
            })

    return {
        "metadata": metadata,
        "measures": measures,
        "comments": comments
    }


# ======================================================================
# PARSE MUSICXML → RomanText lines (FIXED meter + beat math)
# ======================================================================

def _get_ns_prefix(root: ET.Element) -> str:
    if '}' in root.tag:
        ns = root.tag.split('}')[0].strip('{')
        return '{' + ns + '}'
    return ''


def _find_rna_part_id(root: ET.Element, ns_prefix: str) -> str:
    for score_part in root.findall(f'.//{ns_prefix}score-part'):
        pid = score_part.get('id')
        pname = score_part.find(f'{ns_prefix}part-name')
        if pid == 'RNA':
            return pid
        if pname is not None and pname.text:
            t = pname.text.strip()
            if 'Roman' in t or 'RNA' in t or 'Numeral' in t:
                return pid
    raise ValueError("No RNA part found.")


def _get_part_by_id(root: ET.Element, ns_prefix: str, pid: str) -> ET.Element:
    for p in root.findall(f'.//{ns_prefix}part'):
        if p.get('id') == pid:
            return p
    raise ValueError("RNA part missing.")


def parse_musicxml(filepath: str) -> Tuple[dict, List[str]]:
    """Return (metadata, analysis_lines) extracted from MusicXML."""
    tree = ET.parse(filepath)
    root = tree.getroot()
    ns_prefix = _get_ns_prefix(root)

    rna_part_id = _find_rna_part_id(root, ns_prefix)
    rna_part = _get_part_by_id(root, ns_prefix, rna_part_id)

    # Default divisions (per quarter note) + time sig
    divisions_per_quarter = 10080
    beats = 4
    beat_type = 4

    # If there's a global divisions/time somewhere, use as initial defaults
    divisions_elem = rna_part.find(f'.//{ns_prefix}divisions')
    if divisions_elem is not None and divisions_elem.text and divisions_elem.text.strip().isdigit():
        divisions_per_quarter = int(divisions_elem.text.strip())

    ts_elem = rna_part.find(f'.//{ns_prefix}time')
    if ts_elem is not None:
        b = ts_elem.find(f'{ns_prefix}beats')
        bt = ts_elem.find(f'{ns_prefix}beat-type')
        if b is not None and bt is not None and b.text and bt.text:
            beats = int(b.text.strip())
            beat_type = int(bt.text.strip())

    metadata = {"Time Signature": f"{beats}/{beat_type}"}

    analysis_lines: List[str] = []
    current_key: Optional[str] = None

    for measure in rna_part.findall(f'{ns_prefix}measure'):
        number = measure.get('number')

        # Per-measure updates: divisions and time changes live in <attributes>
        attr = measure.find(f'{ns_prefix}attributes')
        if attr is not None:
            div_el = attr.find(f'{ns_prefix}divisions')
            if div_el is not None and div_el.text and div_el.text.strip().isdigit():
                divisions_per_quarter = int(div_el.text.strip())

            t = attr.find(f'{ns_prefix}time')
            if t is not None:
                b = t.find(f'{ns_prefix}beats')
                bt = t.find(f'{ns_prefix}beat-type')
                if b is not None and bt is not None and b.text and bt.text:
                    beats = int(b.text.strip())
                    beat_type = int(bt.text.strip())
                    # update global metadata only for the header; we keep the latest seen
                    metadata["Time Signature"] = f"{beats}/{beat_type}"

        current_position = 0
        harmonies: List[Tuple[int, str]] = []

        # Collect harmonies with their current_position offsets
        for elem in list(measure):
            tag = elem.tag.replace(ns_prefix, '')

            if tag == 'forward':
                dur = elem.find(f'{ns_prefix}duration')
                if dur is not None and dur.text:
                    current_position += int(dur.text.strip())

            elif tag == 'harmony':
                fx = elem.find(f'{ns_prefix}function')
                if fx is not None and fx.text:
                    harmonies.append((current_position, fx.text.strip()))

            elif tag == 'note':
                dur = elem.find(f'{ns_prefix}duration')
                if dur is not None and dur.text:
                    current_position += int(dur.text.strip())

        if not harmonies:
            continue

        out: List[str] = [f"m{number}"]
        measure_key: Optional[str] = None

        for i, (pos, harm) in enumerate(harmonies):
            # Convert offset to beat marker (correctly)
            beat, sub = duration_to_beat(pos, divisions_per_quarter, beat_type)

            # Safety valve: if beat exceeds measure beats, skip this harmony
            # (prevents generating b3 in a 2/4 bar, which breaks music21 parsing)
            if beat > beats:
                # You can alternatively clamp, but skipping is less misleading
                continue

            # Key changes
            if ':' in harm:
                key, h2 = harm.split(':', 1)
                is_new = (key != measure_key)

                if i == 0:
                    if pos > 0:
                        out.append(format_beat(beat, sub))
                    out.append(f"{key}:")
                elif is_new:
                    out.append(format_beat(beat, sub))
                    out.append(f"{key}:")
                else:
                    out.append(format_beat(beat, sub))

                measure_key = key
                harm = h2
            else:
                if i == 0:
                    if pos > 0:
                        out.append(format_beat(beat, sub))
                    if current_key and current_key != measure_key:
                        out.append(f"{current_key}:")
                        measure_key = current_key
                else:
                    out.append(format_beat(beat, sub))

            harm = transform_figured_bass(harm)
            out.append(harm)

        # If we skipped everything due to safety valve, skip the measure
        if len(out) > 1:
            analysis_lines.append(" ".join(out))

    return metadata, analysis_lines


# ======================================================================
# GENERATE TEXT OUTPUT
# ======================================================================

def generate_output(metadata: dict, analysis_lines: List[str]) -> str:
    output: List[str] = []
    output.append("Composer: [TODO]")
    output.append("Piece: [TODO]")
    output.append("Analyst: [TODO]")
    output.append("Proofreader: [TODO]")
    output.append("Note: [TODO]")
    output.append("")
    output.append("Tempo: [TODO]")
    output.append(f"Time Signature: {metadata.get('Time Signature', '4/4')}")
    output.append("")
    output.append("Form: [TODO]")
    output.append("[TODO: Add pedal markings if applicable]")
    output.append("")

    output.extend(analysis_lines)

    output.append("")
    output.append("[TODO Review and add as needed]")
    return "\n".join(output)


# ======================================================================
# TEXT → MUSICXML (not implemented here)
# ======================================================================

def generate_musicxml(data: dict) -> str:
    """
    Placeholder: You didn't include this in the snippet you sent.
    If you need txt -> MusicXML, tell me what schema you want and I’ll wire it up.
    """
    raise NotImplementedError("Text → MusicXML generation is not implemented in this script.")


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: convert_musicxml_to_analysis.py <input> [output]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    is_xml = input_file.endswith(".xml") or input_file.endswith(".musicxml")
    is_txt = input_file.endswith(".txt")

    if not (is_xml or is_txt):
        print("Error: expected .xml/.musicxml or .txt")
        sys.exit(1)

    try:
        if is_xml:
            print(f"Converting MusicXML → text: {input_file}")
            metadata, analysis_lines = parse_musicxml(input_file)
            output = generate_output(metadata, analysis_lines)

            if output_file:
                Path(output_file).write_text(output, encoding="utf-8")
                print(f"Written: {output_file}")
            else:
                print(output)

        else:
            print(f"Converting text → MusicXML: {input_file}")
            data = parse_text_analysis(input_file)
            xml_str = generate_musicxml(data)

            if output_file:
                Path(output_file).write_text(xml_str, encoding="utf-8")
                print(f"Written: {output_file}")
            else:
                print(xml_str)

    except Exception as e:
        print("\nERROR:", e, file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
