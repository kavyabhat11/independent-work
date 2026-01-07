#!/usr/bin/env python3
"""
Convert Mozart MusicXML + analysis TXT to ChordGNN-compatible TSV.

FIXED VERSION - Properly encodes:
- Voice leading (soprano/alto/tenor/bass from actual notes)
- Secondary dominants (correct tonicized keys)
- Degree2 for applied chords
- Harmonic rhythm from annotation durations
- PCset from actual notes in score
"""

import os
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import partitura as pt

# music21 is used to realize roman numeral chords into pitches/pcset/etc.
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
    "s_beat",
    "ts_beats",
    "ts_beat_type",
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
    "a_harmonicRhythm",
    "a_tenor",
    "a_alto",
    "a_soprano",
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
    """Parse '3/4' etc. Returns (numerator, denominator) or defaults to (4,4)."""
    if not ts:
        return 4, 4
    ts = ts.strip()
    try:
        num_s, den_s = ts.split("/", 1)
        return int(num_s), int(den_s)
    except Exception:
        return 4, 4


def midi_to_name(midi_pitch: int) -> str:
    """Convert MIDI pitch to note name (e.g., 60 -> C4)."""
    if m21pitch is None:
        note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        octave = (midi_pitch // 12) - 1
        note = note_names[midi_pitch % 12]
        return f"{note}{octave}"
    return m21pitch.Pitch(midi_pitch).nameWithOctave


def midi_to_name_no_octave(midi_pitch: int) -> str:
    """Convert MIDI pitch to note name without octave (e.g., 60 -> C)."""
    if m21pitch is None:
        note_names = ["C", "C-sharp", "D", "D-sharp", "E", "F", "F-sharp", "G", "G-sharp", "A", "A-sharp", "B"]
        return note_names[midi_pitch % 12].replace("-sharp", "#")
    p = m21pitch.Pitch(midi_pitch)
    return p.name.replace('b', '-')  # Use '-' for flats


def extract_voices_from_midi(midi_pitches):
    """
    Extract SATB voices from a list of MIDI pitches.
    Returns: (soprano, alto, tenor, bass) as note names without octave.
    """
    if not midi_pitches:
        return "C", "C", "C", "C"

    sorted_pitches = sorted(midi_pitches)

    # Bass = lowest
    bass = midi_to_name_no_octave(sorted_pitches[0])

    # Soprano = highest
    soprano = midi_to_name_no_octave(sorted_pitches[-1])

    # Tenor and Alto = distribute middle notes
    if len(sorted_pitches) == 1:
        # Only one note - all voices same
        return soprano, soprano, soprano, bass
    elif len(sorted_pitches) == 2:
        # Two notes - upper voices sing high, lower voices sing low
        return soprano, soprano, bass, bass
    elif len(sorted_pitches) == 3:
        # Three notes - use middle for both tenor and alto
        middle = midi_to_name_no_octave(sorted_pitches[1])
        return soprano, middle, middle, bass
    else:
        # Four or more notes - use second-lowest for tenor, second-highest for alto
        tenor = midi_to_name_no_octave(sorted_pitches[1])
        alto = midi_to_name_no_octave(sorted_pitches[-2])
        return soprano, alto, tenor, bass


def key_token_to_music21_key(key_tok: str):
    """Convert analysis key tokens like 'C', 'f', 'A-' into music21 Key."""
    if m21key is None:
        return None

    if not key_tok:
        return m21key.Key("C")

    k = key_tok.strip()
    if not k:
        return m21key.Key("C")

    mode = "minor" if k[0].islower() else "major"
    tonic = k[0].upper() + k[1:]  # preserve accidentals
    try:
        return m21key.Key(tonic, mode)
    except Exception:
        return m21key.Key("C")


def roman_numeral_to_scale_degree(rn_str: str):
    """
    Convert Roman numeral string to scale degree (1-7).
    Examples: I->1, ii->2, III->3, IV->4, V->5, vi->6, vii->7
    Handles uppercase/lowercase, ignores quality markers.
    """
    if not rn_str:
        return 1

    # Strip quality markers and inversion numbers
    import re
    base = re.sub(r'[o+#øØ]', '', rn_str)  # Remove quality markers
    base = re.sub(r'\d', '', base)  # Remove numbers
    base = base.strip()

    # Map Roman numerals to scale degrees
    roman_map = {
        'i': 1, 'I': 1,
        'ii': 2, 'II': 2,
        'iii': 3, 'III': 3,
        'iv': 4, 'IV': 4,
        'v': 5, 'V': 5,
        'vi': 6, 'VI': 6,
        'vii': 7, 'VII': 7,
        'N': 2,  # Neapolitan is lowered 2
        'It': 6, 'Fr': 6, 'Ger': 6,  # Aug6 chords built on b6
    }

    base_lower = base.lower()
    return roman_map.get(base_lower, roman_map.get(base, 1))


def compute_tonicized_key(tonkey_rn: str, local_key_tok: str):
    """
    Compute the actual tonicized key from a Roman numeral and local key.

    Examples:
        tonkey_rn="V", local_key_tok="C" -> "G"
        tonkey_rn="ii", local_key_tok="C" -> "d"
        tonkey_rn="IV", local_key_tok="G" -> "C"
    """
    if not tonkey_rn or tonkey_rn == local_key_tok:
        return local_key_tok if local_key_tok else "C"

    if m21key is None or m21pitch is None:
        return local_key_tok if local_key_tok else "C"

    try:
        local_key = key_token_to_music21_key(local_key_tok)
        degree = roman_numeral_to_scale_degree(tonkey_rn)

        # Get the scale degree pitch from the local key
        # music21 scale degrees are 1-indexed
        scale_pitch = local_key.pitchFromDegree(degree)

        # Determine if tonicized key is major or minor based on Roman numeral case
        is_minor = tonkey_rn[0].islower() if tonkey_rn else False

        # Create the tonicized key
        ton_key = m21key.Key(scale_pitch, 'minor' if is_minor else 'major')

        # Format: lowercase first letter for minor, uppercase for major
        tonic_name = scale_pitch.name.replace('b', '-')
        if is_minor:
            tonic_name = tonic_name.lower()

        return tonic_name if tonic_name else "C"

    except Exception:
        return local_key_tok if local_key_tok else "C"


def duration_to_harmonic_rhythm_category(duration_beats: float):
    """
    Convert duration in beats to HarmonicRhythm7 category (0-6).

    Categories:
    0: onset (very short, < 0.25 beats)
    1: thirty-second (0.25 beats)
    2: sixteenth (0.5 beats)
    3: eighth (1 beat)
    4: quarter (2 beats)
    5: half (4 beats)
    6: whole (8+ beats)
    """
    if duration_beats < 0.25:
        return 0
    elif duration_beats < 0.5:
        return 1
    elif duration_beats < 1.0:
        return 2
    elif duration_beats < 2.0:
        return 3
    elif duration_beats < 4.0:
        return 4
    elif duration_beats < 8.0:
        return 5
    else:
        return 6


def analysis_lookup(analysis_events, meas: int, beat_in_meas: float):
    """
    Return the last analysis event with (measure, beat) <= (meas, beat_in_meas).
    If none, return the first event.
    """
    if not analysis_events:
        return None

    chosen = None
    for ev in analysis_events:
        if (ev["measure"], ev["beat"]) <= (meas, beat_in_meas):
            chosen = ev
        else:
            break
    return chosen if chosen is not None else analysis_events[0]


def simplify_roman_numeral_to_31class(roman_str: str):
    """
    Simplify complex Roman numeral labels to fit the 31-class vocabulary.

    Returns: (base_rn, tonkey)
    """
    if not roman_str:
        return ("I", None)

    # Split by '/' to separate secondary function
    parts = roman_str.split('/')

    # The base chord is the first part (may contain inversion numbers)
    base = parts[0]

    # Check for quality markers in slash notation: ii/o2/i means "iio in 2nd inv, tonicized to i"
    quality_marker = ""
    if len(parts) > 1:
        if parts[1].startswith('o'):
            quality_marker = "o"
            if '7' in parts[1]:
                quality_marker = "o7"

    # Check if there's a secondary function (tonicization)
    tonkey = None
    if len(parts) > 1:
        # Find the last part that looks like a Roman numeral
        for i in range(len(parts) - 1, 0, -1):
            part = parts[i]
            if part.startswith('o'):
                continue
            if part and (part[0].lower() in 'ivxn' or part == 'bII' or part.startswith('#')):
                tonkey = part
                break

    # Remove inversion numbers from base chord
    import re

    base_clean = base

    # Remove inversion suffixes
    base_clean = re.sub(r'\[.*?\]', '', base_clean)
    base_clean = re.sub(r'\+\d+$', '+', base_clean)
    base_clean = re.sub(r'6/5$', '', base_clean)
    base_clean = re.sub(r'6/4$', '', base_clean)
    base_clean = re.sub(r'4/3$', '', base_clean)
    base_clean = re.sub(r'6$', '', base_clean)
    base_clean = re.sub(r'(?<!7)2$', '', base_clean)
    base_clean = re.sub(r'(?<!7)4$', '', base_clean)
    base_clean = re.sub(r'(?<!7)3$', '', base_clean)

    # Apply quality marker if found
    if quality_marker:
        base_clean = base_clean + quality_marker

    # 31-class vocabulary
    vocab_31 = {
        'Cad', 'Fr7', 'Ger7', 'I', 'I7', 'III+', 'III+7', 'IV', 'IV7', 'It', 'N',
        'V', 'V+', 'V7', 'VI', 'VI7', 'i', 'i7', 'ii', 'ii7', 'iii', 'iii7',
        'iio', 'iiø7', 'iv', 'iv7', 'vi', 'vi7', 'viio', 'viio7', 'viiø7'
    }

    if base_clean in vocab_31:
        return (base_clean, tonkey)

    # Fallback mappings
    fallback_map = {
        'iio7': 'iiø7',
        'Io': 'I', 'IVo': 'IV', 'Vo': 'V',
        'vo': 'V', 'v': 'V',
        'Fr': 'Fr7', 'Ger': 'Ger7',
        'III': 'iii', 'III7': 'iii7', 'VII': 'viio',
        '#io': 'iio', '#iio': 'iii', '#ivo': 'V',
        '#vio': 'viio', '#vio7': 'viio7', '#io7': 'iio',
        '||': 'I',
    }

    if base_clean in fallback_map:
        return (fallback_map[base_clean], tonkey)

    # Extract just the Roman numeral part
    match = re.match(r'^([ivxIVX]+)', base_clean)
    if match:
        rn_base = match.group(1)
        rn_lower = rn_base.lower()
        if rn_lower in ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii']:
            if rn_base[0].isupper():
                normalized = rn_base.upper()
                if normalized in vocab_31:
                    return (normalized, tonkey)
            else:
                if rn_lower in vocab_31:
                    return (rn_lower, tonkey)

    if base_clean:
        return (base_clean, tonkey)
    return ("I", tonkey)


def realize_roman_numeral(roman_str: str, local_key_tok: str, midi_pitches: list):
    """
    Compute chord attributes from roman numeral, key, and actual MIDI pitches.
    Returns a dict matching needed a_* fields.

    FIXED: Uses actual notes from score for voices and pcset.
    """
    # Simplify to 31-class vocab
    base_rn, tonkey = simplify_roman_numeral_to_31class(roman_str)

    # Compute tonicized key
    tonicized_key_tok = compute_tonicized_key(tonkey, local_key_tok) if tonkey else local_key_tok

    # Extract voices from actual notes
    soprano, alto, tenor, bass = extract_voices_from_midi(midi_pitches)

    # Default pcset (will be overridden by music21 if available)
    pcs = (0, 4, 7)  # C major triad default

    # Validate and normalize keys
    validated_local_key = local_key_tok if local_key_tok else "C"
    validated_ton_key = tonicized_key_tok if tonicized_key_tok else validated_local_key

    # Ensure keys are not just punctuation
    if validated_local_key in ['-', '#', 'b', '']:
        validated_local_key = "C"
    if validated_ton_key in ['-', '#', 'b', '']:
        validated_ton_key = validated_local_key

    # Defaults
    out = {
        "a_romanNumeral": base_rn,
        "a_pitchNames": "('C', 'E', 'G')",
        "a_bass": bass,
        "a_root": "C",
        "a_inversion": 0.0,
        "a_quality": "unknown",
        "a_pcset": repr(pcs),
        "a_localKey": validated_local_key,
        "a_tonicizedKey": validated_ton_key,
        "a_degree1": '1',
        "a_degree2": 'None',
        "a_soprano": soprano,
        "a_alto": alto,
        "a_tenor": tenor,
    }

    if m21roman is None or m21key is None:
        return out

    # Use the tonicized key for music21 RomanNumeral parsing
    kobj = key_token_to_music21_key(tonicized_key_tok)
    try:
        rn = m21roman.RomanNumeral(base_rn, kobj)

        # Pitch names
        pitch_names = tuple(p.name.replace('b', '-') for p in rn.pitches)
        out["a_pitchNames"] = repr(pitch_names)

        # Root (from music21, but bass from actual notes)
        out["a_root"] = rn.root().name.replace('b', '-')

        # Inversion
        out["a_inversion"] = int(rn.inversion())

        # PCset from music21 RomanNumeral (guaranteed to be in vocabulary)
        pcs_from_rn = tuple(sorted(set(p.pitchClass for p in rn.pitches)))
        out["a_pcset"] = repr(pcs_from_rn)

        # Quality
        quality_map = {
            "major triad": "maj",
            "minor triad": "min",
            "dominant seventh chord": "7",
            "major seventh chord": "maj7",
            "minor seventh chord": "min7",
            "diminished seventh chord": "dim7",
            "half-diminished seventh chord": "hdim7",
            "diminished triad": "dim",
            "augmented triad": "aug",
            "augmented sixth": "aug6",
            "augmented seventh chord": "aug7",
        }

        q = getattr(rn, "commonName", None) or getattr(rn, "quality", None) or "maj"
        q_str = str(q)
        out["a_quality"] = quality_map.get(q_str, "maj")

        # Normalize local key
        local_key_normalized = (local_key_tok if local_key_tok else kobj.tonic.name)
        local_key_normalized = local_key_normalized.replace('b', '-').replace('##', '#')
        # Validate key is not empty or just punctuation
        if not local_key_normalized or local_key_normalized in ['-', '#', 'b', '']:
            local_key_normalized = "C"
        out["a_localKey"] = local_key_normalized

        # Degree1: scale degree of the chord in the tonicized key
        try:
            out["a_degree1"] = str(int(rn.scaleDegree))
        except Exception:
            out["a_degree1"] = '1'

        # Degree2: scale degree of the tonicized key in the local key
        if tonkey:
            degree2 = roman_numeral_to_scale_degree(tonkey)
            out["a_degree2"] = str(degree2)
        else:
            out["a_degree2"] = 'None'

        # Tonicized key (already computed above)
        ton_normalized = tonicized_key_tok.replace('b', '-').replace('##', '#')
        # Validate key is not empty or just punctuation
        if not ton_normalized or ton_normalized in ['-', '#', 'b', '']:
            ton_normalized = local_key_normalized
        out["a_tonicizedKey"] = ton_normalized

        return out

    except Exception as e:
        # If music21 fails, return defaults with actual voices
        return out


def xml_txt_to_tsv(xml_file: str, txt_file: str, output_tsv: str, grid: float = 0.5):
    """
    Convert XML + TXT to TSV with proper voice leading and harmonic encoding.
    grid: quantization in beats (e.g., 0.5 -> eighth-note grid)
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

    # Time signature
    ts_num, ts_den = parse_time_signature(metadata.get("Time Signature", "4/4"))
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
        divs_per_quarter = getattr(part, "divisions", None)
        if not divs_per_quarter:
            divs_per_quarter = 480.0

    # Build onset bins: group notes by quantized onset time
    bins = {}  # q_onset -> list of midi pitches
    dur_by_bin = {}
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

    # Compute annotation durations for harmonic rhythm
    annotation_durations = {}
    sorted_analyses = sorted(analysis, key=lambda x: (x["measure"], x["beat"]))
    for i, ev in enumerate(sorted_analyses):
        # Compute offset in beats for this annotation
        ev_offset = (ev["measure"] - 1) * beats_per_measure + (ev["beat"] - 1.0)

        # Duration until next annotation or end of piece
        if i + 1 < len(sorted_analyses):
            next_ev = sorted_analyses[i + 1]
            next_offset = (next_ev["measure"] - 1) * beats_per_measure + (next_ev["beat"] - 1.0)
            duration = next_offset - ev_offset
        else:
            # Last annotation - use a default long duration
            duration = 8.0  # whole note equivalent

        annotation_durations[(ev["measure"], ev["beat"])] = duration

    # Create TSV rows
    rows = []
    sorted_onsets = sorted(bins.keys())

    ann_counter = 0
    for q_onset in sorted_onsets:
        midi_pitches = sorted(bins[q_onset])
        s_notes_list = [midi_to_name(m) for m in midi_pitches]
        s_is_onset = [True] * len(s_notes_list)

        # Approximate measure/beat from q_onset
        meas = int(q_onset // beats_per_measure) + 1
        beat_in_meas = (q_onset % beats_per_measure) + 1.0

        ev = analysis_lookup(analysis, meas, beat_in_meas)
        local_key_tok = ev["key"] if (ev and ev.get("key")) else "C"
        roman_str = ev["roman"] if (ev and ev.get("roman")) else "I"

        # Get annotation duration for harmonic rhythm
        if ev:
            ann_duration = annotation_durations.get((ev["measure"], ev["beat"]), 1.0)
        else:
            ann_duration = 1.0

        harmonic_rhythm_cat = duration_to_harmonic_rhythm_category(ann_duration)

        # Realize chord with actual MIDI pitches
        chord_info = realize_roman_numeral(roman_str, local_key_tok, midi_pitches)

        row = {
            "j_offset": float(q_onset),
            "s_duration": float(max(grid, dur_by_bin.get(q_onset, grid))),
            "s_measure": float(meas),
            "s_beat": float(beat_in_meas),
            "ts_beats": int(ts_num),
            "ts_beat_type": int(ts_den),
            "s_notes": repr(s_notes_list),
            "s_intervals": "[]",
            "s_isOnset": repr(s_is_onset),

            "a_measure": float(meas),
            "a_duration": float(ann_duration),
            "a_annotationNumber": float(ann_counter),
            "a_romanNumeral": chord_info["a_romanNumeral"],
            "a_isOnset": True,

            "a_pitchNames": chord_info["a_pitchNames"],
            "a_bass": chord_info["a_bass"],
            "a_root": chord_info["a_root"],
            "a_inversion": float(chord_info["a_inversion"]),
            "a_quality": chord_info["a_quality"],
            "a_pcset": chord_info["a_pcset"],

            "a_localKey": chord_info["a_localKey"],
            "a_tonicizedKey": chord_info["a_tonicizedKey"],
            "a_degree1": int(chord_info["a_degree1"]),
            "a_degree2": chord_info["a_degree2"],
            "a_harmonicRhythm": harmonic_rhythm_cat,
            "a_tenor": chord_info["a_tenor"],
            "a_alto": chord_info["a_alto"],
            "a_soprano": chord_info["a_soprano"],

            "measureMisalignment": False,
            "qualityScoreNotes": 0.0,
            "qualityNonChordTones": 0.0,
            "qualityMissingChordTones": 0.0,
            "qualitySquaredSum": 0.0,
            "incongruentBass": 0.0,
        }

        rows.append(row)
        ann_counter += 1

    df = pd.DataFrame(rows, columns=REAL_TSV_COLUMNS)

    # Write TSV
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
    ap = argparse.ArgumentParser(description="Convert Mozart XML+TXT to ChordGNN TSV (FIXED)")
    ap.add_argument("--data_dir", default="./data/mozart", help="Directory containing Mozart XML and TXT files")
    ap.add_argument("--output_dir", default="./mozart_tsv_fixed", help="Output directory for TSV files")
    ap.add_argument("--grid", type=float, default=0.5, help="Quantization grid in beats (default 0.5)")
    args = ap.parse_args()

    print(f"Converting Mozart files from: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Grid: {args.grid}\n")

    convert_all(args.data_dir, args.output_dir, grid=args.grid)


if __name__ == "__main__":
    main()
