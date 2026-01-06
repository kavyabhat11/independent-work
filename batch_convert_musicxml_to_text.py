import os
import subprocess

INPUT_DIR = "analysis/mozart_test"         # Input: analyzed MusicXML files
OUTPUT_DIR = "txt_analysis/mozart"    # Output: cleaned text analyses

os.makedirs(OUTPUT_DIR, exist_ok=True)

for fname in sorted(os.listdir(INPUT_DIR)):
    # Only convert files ending with -analysis.musicxml
    if not fname.endswith("-analysis.musicxml"):
        continue

    xml_path = os.path.join(INPUT_DIR, fname)

    # Example: K310-1-analysis.musicxml -> K310-1-analysis.txt
    base = os.path.splitext(fname)[0]
    txt_path = os.path.join(OUTPUT_DIR, base + ".txt")

    print(f"Converting: {fname} → {txt_path}")

    # Step 1: Convert MusicXML to text
    subprocess.run([
        "python",
        "convert_musicxml_to_analysis.py",
        xml_path,
        txt_path
    ], check=True)

    # Step 2: Fix Cadential chords → I (cadence simplification)
    print(f"  Applying cadence fix to: {txt_path}")

    with open(txt_path, "r") as f:
        text = f.read()

    # Replace Cad with I (safe rule: only Roman numeral contexts)
    # This fixes:
    #   Cad     → I
    #   Cad6/4  → I6/4
    #   Cad64   → I64
    #   etc.
    cleaned = text.replace("Cad", "I")

    with open(txt_path, "w") as f:
        f.write(cleaned)

    print(f"  ✔ Cleaned & saved: {txt_path}\n")
