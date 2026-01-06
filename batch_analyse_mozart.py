#!/usr/bin/env python3
import os
import glob
import shutil
import subprocess

# -------- CONFIG --------
TSV_TEST_DIR = "mozart_dataset/test"      # test TSVs (defines test set)
SOURCE_DIR = "data/mozart"                # contains .xml + .txt
TEST_DIR = "data/mozart_test"             # new isolated test dir

ANALYSE_SCRIPT = "analyse_score.py"
OUTPUT_DIR = "analysis/mozart_test"

USE_CKPT = "kb9520-princeton-university/chord_rec/mozart-finetuned-model:v1"
# ------------------------

def base(path):
    return os.path.splitext(os.path.basename(path))[0]

def main():
    os.makedirs(TEST_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1) Collect test basenames from TSVs
    tsvs = sorted(glob.glob(os.path.join(TSV_TEST_DIR, "*.tsv")))
    if not tsvs:
        raise SystemExit(f"No TSVs found in {TSV_TEST_DIR}")

    test_bases = sorted({base(p) for p in tsvs})

    print(f"[1/4] Found {len(test_bases)} test pieces")
    print("      Examples:", test_bases[:10])

    # 2) Clear test dir to avoid stale files
    for f in glob.glob(os.path.join(TEST_DIR, "*")):
        os.remove(f)

    # 3) Copy XML + TXT into test dir
    print(f"\n[2/4] Copying XML + TXT into {TEST_DIR}")

    missing = {"xml": [], "txt": []}

    for b in test_bases:
        src_xml = os.path.join(SOURCE_DIR, f"{b}.xml")
        src_txt = os.path.join(SOURCE_DIR, f"{b}.txt")

        dst_xml = os.path.join(TEST_DIR, f"{b}.xml")
        dst_txt = os.path.join(TEST_DIR, f"{b}.txt")

        if os.path.exists(src_xml):
            shutil.copy2(src_xml, dst_xml)
        else:
            missing["xml"].append(b)

        if os.path.exists(src_txt):
            shutil.copy2(src_txt, dst_txt)
        else:
            missing["txt"].append(b)

    print(f"      Copied {len(test_bases) - len(missing['xml'])} XMLs")
    print(f"      Copied {len(test_bases) - len(missing['txt'])} TXTs")

    if missing["xml"]:
        print(f"      ⚠ Missing XML for {len(missing['xml'])} pieces")
        print("        Examples:", missing["xml"][:10])
    if missing["txt"]:
        print(f"      ⚠ Missing TXT for {len(missing['txt'])} pieces")
        print("        Examples:", missing["txt"][:10])

    # 4) Run analyzer ONLY on test set using finetuned checkpoint
    xml_files = sorted(
        f for f in os.listdir(TEST_DIR)
        if f.endswith(".xml") and "-analysis" not in f
    )

    print(f"\n[3/4] Running analysis on {len(xml_files)} test pieces")

    for fname in xml_files:
        b = os.path.splitext(fname)[0]
        in_path = os.path.join(TEST_DIR, fname)

        cmd = [
            "python", ANALYSE_SCRIPT,
            "--score_path", in_path,
            "--use_ckpt", USE_CKPT,
        ]
        print("\nCMD:", " ".join(cmd))
        subprocess.run(cmd, check=True)

        out_file = f"{b}-analysis.musicxml"
        out_path = os.path.join(TEST_DIR, out_file)

        if not os.path.exists(out_path):
            print(f"  ❌ Expected output missing: {out_file}")
            continue

        shutil.move(out_path, os.path.join(OUTPUT_DIR, out_file))
        print(f"  → Moved {out_file} to {OUTPUT_DIR}")

    # 5) Summary
    produced = len(glob.glob(os.path.join(OUTPUT_DIR, "*-analysis.musicxml")))

    print(f"\n[4/4] DONE")
    print(f"      Test pieces: {len(test_bases)}")
    print(f"      Analyses produced: {produced}")
    print(f"      Output dir: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
