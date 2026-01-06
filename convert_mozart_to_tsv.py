#!/usr/bin/env python3
"""
Convert Mozart MusicXML + analysis TXT to ChordGNN-compatible TSV.

Goals:
- Column names compatible with your "real" TSVs
- Types compatible with chordgnn.utils.chord_representations.time_divided_tsv_to_note_array
- Roman numeral -> chord tones computed from actual RN + local key (not always C major)
- s_notes contains multiple notes per time slice (quantized onset bins)

Dependencies:
- pandas
- partitura
- music21  (used to realize RomanNumeral chords)
"""

import os
from pathlib import Path
import argparse
import pandas as pd
import partitura as pt

# music21 is used to realize roman numeral chords into pitches/pcset/etc.
# If music21 isn't available, we fall back gracefully (but you probably have it).
try:
    from music21 import roman as m21roman
    from music21 import key as m21key
    from music21 import pitch as m21pitch
except Exception:
    m21roman = None
    m21key = None
    m21pitch = None


REAL_TSV_COLUMNS = [
    "j_offset",
    "s_duration",
    "s_measure",
    "s_notes",
    "s_intervals",
    "s_isOnset",
    "a_measure",
    "a_duration",
    "a_annotationNumber",
    "a_romanNumeral",
    "a_isOnset",
    "a_pitchNames",
    "a_bass",
    "a_root",
    "a_inversion",
    "a_quality",
    "a_pcset",
    "a_localKey",
    "a_tonicizedKey",
    "a_degree1",
    "a_degree2",
    "measureMisalignment",
    "qualityScoreNotes",
    "qualityNonChordTones",
    "qualityMissingChordTones",
    "qualitySquaredSum",
    "incongruentBass",
]


def parse_analysis_txt(txt_file: str):
    """Parse a Mozart analysis .txt file into structured events."""
    with open(txt_file, "r") as f:
        lines = f.readlines()

    metadata = {}
    analysis = []
    current_key = None

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("[TODO"):
            continue

        # Metadata
        if ":" in line and not line.startswith("m"):
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip()
            if key in ["Composer", "Piece", "Time Signature", "Tempo"]:
                metadata[key] = val
            continue

        # Skip non-analysis categories
        if any(line.startswith(x) for x in ["Form:", "Pedal:", "Note:"]):
            continue

        # Measure analysis
        if line.startswith("m"):
            # Skip equivalences like m3-4 = m1-2
            if "=" in line:
                continue

            parts = line.split()
            if not parts:
                continue

            measure_str = parts[0][1:]  # remove leading 'm'

            # Skip variant measures like m12var1
            if "var" in measure_str:
                continue

            try:
                measure_num = int(measure_str)
            except ValueError:
                continue

            i = 1
            current_beat = 1.0
            while i < len(parts):
                part = parts[i]

                # Beat marker like b2 or b2.5
                if part.startswith("b"):
                    try:
                        current_beat = float(part[1:])
                    except ValueError:
                        pass
                    i += 1
                    continue

                # Key change like C: or f:
                if part.endswith(":"):
                    current_key = part[:-1]
                    i += 1
                    continue

                # Roman numeral
                roman = part
                analysis.append(
                    {
                        "measure": measure_num,
                        "beat": float(current_beat),
                        "key": current_key,
                        "roman": roman,
                    }
                )
                i += 1

    # Ensure sorted for "last annotation at/before time"
    analysis.sort(key=lambda x: (x["measure"], x["beat"]))
    return metadata, analysis


def parse_time_signature(ts: str):
    """
    Parse '3/4' etc. Returns (numerator, denominator) or defaults to (4,4).
    """
    if not ts:
        return 4, 4
    ts = ts.strip()
    try:
        num_s, den_s = ts.split("/", 1)
        return int(num_s), int(den_s)
    except Exception:
        return 4, 4


def midi_to_m21_name_with_octave(midi_pitch: int) -> str:
    """Use music21 pitch naming if available (gives '-' for flats)."""
    if m21pitch is None:
        # fallback simple sharp-based naming
        note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        octave = (midi_pitch // 12) - 1
        note = note_names[midi_pitch % 12]
        return f"{note}{octave}"
    return m21pitch.Pitch(midi_pitch).nameWithOctave


def key_token_to_music21_key(key_tok: str):
    """
    Convert analysis key tokens like 'C', 'f', 'A-' into music21 Key.
    - Lowercase -> minor
    - Uppercase -> major
    - Keep '-' for flats (music21 understands A- etc.)
    """
    if m21key is None:
        return None

    if not key_tok:
        return m21key.Key("C")

    k = key_tok.strip()
    if not k:
        return m21key.Key("C")

    mode = "minor" if k[0].islower() else "major"
    tonic = k[0].upper() + k[1:]  # preserve accidentals like '-' or '#'
    try:
        return m21key.Key(tonic, mode)
    except Exception:
        return m21key.Key("C")


def analysis_lookup(analysis_events, meas: int, beat_in_meas: float):
    """
    Return the last analysis event with (measure, beat) <= (meas, beat_in_meas).
    If none, return the first event.
    """
    if not analysis_events:
        return None

    # Linear scan is OK for Mozart sizes; if you want speed, binary search later.
    chosen = None
    for ev in analysis_events:
        if (ev["measure"], ev["beat"]) <= (meas, beat_in_meas):
            chosen = ev
        else:
            break
    return chosen if chosen is not None else analysis_events[0]


def realize_roman_numeral(roman_str: str, local_key_tok: str):
    """
    Compute chord attributes from roman numeral and key using music21.
    Returns a dict matching needed a_* fields.
    Falls back safely if music21 can't parse.
    """
    # Defaults (safe + parser-compatible)
    out = {
        "a_romanNumeral": roman_str if roman_str else "I",
        "a_pitchNames": "('C', 'E', 'G')",
        "a_bass": "C",
        "a_root": "C",
        "a_inversion": 0.0,
        "a_quality": "unknown",
        "a_pcset": "(0, 4, 7)",
        "a_localKey": (local_key_tok if local_key_tok else "C"),
        "a_tonicizedKey": (local_key_tok if local_key_tok else "C"),
        "a_degree1": 1,
        "a_degree2": None,
    }

    if m21roman is None or m21key is None:
        return out

    kobj = key_token_to_music21_key(local_key_tok)
    try:
        rn = m21roman.RomanNumeral(roman_str, kobj)

        # Pitch names as tuple literal string, e.g. ('E-', 'G', 'B-', 'D-')
        pitch_names = tuple(p.name for p in rn.pitches)  # no octaves in your real row
        out["a_pitchNames"] = repr(pitch_names)

        # Bass/root
        out["a_bass"] = rn.bass().name
        out["a_root"] = rn.root().name

        # Inversion (0,1,2,3...) as float (real TSV uses 0.0)
        out["a_inversion"] = float(rn.inversion())

        # Quality text (music21 gives short names; make it readable)
        # Your real file had e.g. "dominant seventh chord"
        q = getattr(rn, "commonName", None) or getattr(rn, "quality", None) or "unknown"
        out["a_quality"] = str(q)

        # pcset as tuple of pitch classes (0-11), literal string
        pcs = tuple(sorted({p.pitchClass for p in rn.pitches}))
        out["a_pcset"] = repr(pcs)

        # Local key token: keep your analysis token (like A-), but normalize if empty
        out["a_localKey"] = local_key_tok if local_key_tok else kobj.tonic.name

        # Degree1: scale degree (int)
        try:
            out["a_degree1"] = int(rn.scaleDegree)
        except Exception:
            out["a_degree1"] = 1

        # Degree2: keep None for now (secondary functions are messy across notations)
        out["a_degree2"] = None

        # Tonicized key: if secondary, could infer; otherwise local key
        out["a_tonicizedKey"] = out["a_localKey"]

        return out

    except Exception:
        return out


def xml_txt_to_tsv(xml_file: str, txt_file: str, output_tsv: str, grid: float = 0.5):
    """
    grid: quantization in beats (e.g., 0.5 -> eighth-note grid if beat=quarter)
    """
    print(f"Processing: {os.path.basename(xml_file)}")

    try:
        metadata, analysis = parse_analysis_txt(txt_file)
    except Exception as e:
        print(f"  ERROR parsing {txt_file}: {e}")
        return False

    if not analysis:
        print(f"  WARNING: No analysis found in {txt_file}")
        return False

    # Time signature for measure/beat approximation if needed
    ts_num, ts_den = parse_time_signature(metadata.get("Time Signature", "4/4"))
    # We treat "beat" units as quarter-note beats; if denom != 4, we still use numerator as beats/measure.
    beats_per_measure = float(ts_num)

    # Load score
    try:
        score = pt.load_musicxml(xml_file)
    except Exception as e:
        print(f"  ERROR loading {xml_file}: {e}")
        return False

    # Select part & note array
    try:
        part = score[0] if isinstance(score, list) else score
        note_array = part.note_array()
    except Exception as e:
        print(f"  ERROR extracting notes: {e}")
        return False

    # Determine division scale if onset_beat missing
    dtype_names = set(note_array.dtype.names)
    divs_per_quarter = None
    if "onset_beat" not in dtype_names:
        # Partitura often stores division units in onset_div; estimate qdiv from score/part
        # We'll use a common fallback if not available.
        divs_per_quarter = getattr(part, "divisions", None)
        if not divs_per_quarter:
            divs_per_quarter = 480.0  # safe-ish default

    # Build onset bins: group notes by quantized onset time
    bins = {}  # q_onset -> list of midi pitches
    dur_by_bin = {}  # q_onset -> representative duration (min)
    for n in note_array:
        if "onset_beat" in dtype_names:
            onset = float(n["onset_beat"])
            dur = float(n["duration_beat"]) if "duration_beat" in dtype_names else 1.0
        else:
            onset = float(n["onset_div"]) / float(divs_per_quarter)
            dur = float(n["duration_div"]) / float(divs_per_quarter) if "duration_div" in dtype_names else 1.0

        q_onset = round(onset / grid) * grid
        midi = int(n["pitch"])

        bins.setdefault(q_onset, []).append(midi)
        dur_by_bin[q_onset] = min(dur_by_bin.get(q_onset, dur), dur)

    if not bins:
        print("  WARNING: No note data rows created")
        return False

    # Create TSV rows
    rows = []
    sorted_onsets = sorted(bins.keys())

    ann_counter = 0
    for q_onset in sorted_onsets:
        midi_pitches = sorted(bins[q_onset])
        s_notes_list = [midi_to_m21_name_with_octave(m) for m in midi_pitches]
        s_is_onset = [True] * len(s_notes_list)

        # Approximate measure/beat from q_onset and time signature:
        # measure starts at 1, beat starts at 1.0
        # If q_onset is in quarter-beat units: measure = floor(q_onset / beats_per_measure) + 1
        meas = int(q_onset // beats_per_measure) + 1
        beat_in_meas = (q_onset % beats_per_measure) + 1.0

        ev = analysis_lookup(analysis, meas, beat_in_meas)
        local_key_tok = ev["key"] if (ev and ev.get("key")) else "C"
        roman_str = ev["roman"] if (ev and ev.get("roman")) else "I"

        chord_info = realize_roman_numeral(roman_str, local_key_tok)

        # Make sure literals are emitted as strings exactly like your real TSV style
        # (lists/tuples as literal strings)
        row = {
            "j_offset": float(q_onset),
            "s_duration": float(max(grid, dur_by_bin.get(q_onset, grid))),
            "s_measure": float(meas),
            "s_notes": repr(s_notes_list),          # e.g. ['E-2', 'G2']
            "s_intervals": "[]",                    # keep empty; can be computed later
            "s_isOnset": repr(s_is_onset),          # e.g. [True, True]

            "a_measure": float(meas),
            "a_duration": float(max(grid, dur_by_bin.get(q_onset, grid))),
            "a_annotationNumber": float(ann_counter),
            "a_romanNumeral": chord_info["a_romanNumeral"],
            "a_isOnset": True,

            "a_pitchNames": chord_info["a_pitchNames"],  # tuple literal string
            "a_bass": chord_info["a_bass"],
            "a_root": chord_info["a_root"],
            "a_inversion": float(chord_info["a_inversion"]),
            "a_quality": chord_info["a_quality"],
            "a_pcset": chord_info["a_pcset"],

            "a_localKey": chord_info["a_localKey"],
            "a_tonicizedKey": chord_info["a_tonicizedKey"],
            "a_degree1": int(chord_info["a_degree1"]),
            "a_degree2": chord_info["a_degree2"],        # None (NOT 'None')

            "measureMisalignment": False,

            # Score fields should be numeric like your real file (not lists/strings)
            "qualityScoreNotes": 0.0,
            "qualityNonChordTones": 0.0,
            "qualityMissingChordTones": 0.0,
            "qualitySquaredSum": 0.0,

            "incongruentBass": 0.0,
        }

        rows.append(row)
        ann_counter += 1

    df = pd.DataFrame(rows, columns=REAL_TSV_COLUMNS)

    # Enforce exact column order and write
    Path(output_tsv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_tsv, sep="\t", index=False)
    print(f"  ✓ Created: {output_tsv} ({len(df)} rows)")
    return True


def convert_all(data_dir: str, output_dir: str, grid: float):
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(list(data_path.glob("*.txt")))
    print(f"Found {len(txt_files)} TXT files\n")

    converted = 0
    failed = []

    for txt_file in txt_files:
        base = txt_file.stem
        xml_file = data_path / f"{base}.xml"
        if not xml_file.exists():
            print(f"WARNING: No XML file for {txt_file.name}")
            continue

        out_tsv = out_path / f"{base}.tsv"
        ok = xml_txt_to_tsv(str(xml_file), str(txt_file), str(out_tsv), grid=grid)
        if ok:
            converted += 1
        else:
            failed.append(base)

    print(f"\n{'='*60}")
    print("Conversion complete!")
    print(f"  ✓ Successfully converted: {converted} files")
    if failed:
        print(f"  ✗ Failed: {len(failed)} files")
        print(f"    {', '.join(failed[:5])}" + ("..." if len(failed) > 5 else ""))
    print(f"  Output directory: {out_path}")
    print(f"{'='*60}\n")

    return converted


def main():
    ap = argparse.ArgumentParser(description="Convert Mozart XML+TXT to ChordGNN TSV")
    ap.add_argument("--data_dir", default="./data/mozart", help="Directory containing Mozart XML and TXT files")
    ap.add_argument("--output_dir", default="./mozart_tsv", help="Output directory for TSV files")
    ap.add_argument("--grid", type=float, default=0.5, help="Quantization grid in beats (default 0.5)")
    args = ap.parse_args()

    print(f"Converting Mozart files from: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Grid: {args.grid}\n")

    convert_all(args.data_dir, args.output_dir, grid=args.grid)


if __name__ == "__main__":
    main()
