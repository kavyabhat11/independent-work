#!/usr/bin/env python3
"""
Split Mozart/Haydn/Beethoven TSV files into train/val/test sets.

CRITICAL: Groups by WORK (not by file) to avoid splitting movements across sets.
All files from the same work (e.g., K545-1, K545-2, K545-3) stay together.
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

def extract_work_id(filename):
    """
    Extract work ID from filename to group related movements/sections.

    Examples:
        K545-1.tsv → K545
        K545-2.tsv → K545
        sym100a-EXPO.tsv → sym100a
        sym100a-DEV.tsv → sym100a
        op31n2-1.tsv → op31n2

    Strategy: Take everything before the last dash + number/section pattern
    """
    stem = Path(filename).stem  # Remove .tsv extension

    # Handle common patterns:
    # - K545-1 → K545
    # - sym100a-EXPO → sym100a
    # - op31n2-1 → op31n2

    # Split by last dash
    if '-' in stem:
        parts = stem.rsplit('-', 1)
        work_id = parts[0]
        return work_id
    else:
        # No dash - entire name is the work ID
        return stem


def split_by_work(tsv_dir, output_dir, n_train=44, n_val=30, seed=42):
    """
    Split TSV files by WORK (not by file) to keep movements together.

    Args:
        tsv_dir: Directory containing TSV files
        output_dir: Output directory for split dataset
        n_train: Number of training WORKS (not files)
        n_val: Number of validation WORKS
        seed: Random seed for reproducibility
    """
    random.seed(seed)

    tsv_path = Path(tsv_dir)
    tsv_files = list(tsv_path.glob("*.tsv"))

    print(f"Found {len(tsv_files)} TSV files in {tsv_dir}")

    # Group files by work
    works = defaultdict(list)
    for tsv_file in tsv_files:
        work_id = extract_work_id(tsv_file.name)
        works[work_id].append(tsv_file)

    print(f"\nGrouped into {len(works)} works:")
    # Show examples of grouping
    for work_id, files in sorted(works.items())[:5]:
        file_names = [f.name for f in files]
        print(f"  {work_id}: {len(files)} files - {', '.join(file_names)}")
    if len(works) > 5:
        print(f"  ... and {len(works) - 5} more works")

    # Check if we have enough works
    total_works = len(works)
    if total_works < (n_train + n_val + 1):
        print(f"\n⚠ WARNING: Only {total_works} works available!")
        print(f"Requested: {n_train} train + {n_val} val = {n_train + n_val} minimum")
        print("Adjusting splits...")
        n_train = min(n_train, total_works - 10)
        n_val = min(n_val, total_works - n_train - 5)

    # Shuffle works (not individual files!)
    work_ids = list(works.keys())
    random.shuffle(work_ids)

    # Split by work
    train_work_ids = work_ids[:n_train]
    val_work_ids = work_ids[n_train:n_train + n_val]
    test_work_ids = work_ids[n_train + n_val:]

    # Collect all files for each split
    train_files = []
    for work_id in train_work_ids:
        train_files.extend(works[work_id])

    val_files = []
    for work_id in val_work_ids:
        val_files.extend(works[work_id])

    test_files = []
    for work_id in test_work_ids:
        test_files.extend(works[work_id])

    print(f"\n{'='*70}")
    print("DATASET SPLIT (by work, not by file)")
    print(f"{'='*70}")
    print(f"Training:   {len(train_work_ids):3d} works → {len(train_files):3d} files")
    print(f"Validation: {len(val_work_ids):3d} works → {len(val_files):3d} files")
    print(f"Test:       {len(test_work_ids):3d} works → {len(test_files):3d} files")
    print(f"Total:      {len(works):3d} works → {len(tsv_files):3d} files")

    # Verify no overlap
    train_set = set(train_work_ids)
    val_set = set(val_work_ids)
    test_set = set(test_work_ids)

    assert len(train_set & val_set) == 0, "Training and validation overlap!"
    assert len(train_set & test_set) == 0, "Training and test overlap!"
    assert len(val_set & test_set) == 0, "Validation and test overlap!"
    print("\n✓ No work appears in multiple splits")

    # Create output directories
    output_path = Path(output_dir)
    train_dir = output_path / "training"
    val_dir = output_path / "validation"
    test_dir = output_path / "test"

    for dir in [train_dir, val_dir, test_dir]:
        dir.mkdir(parents=True, exist_ok=True)

    # Copy files
    print(f"\nCopying files to {output_dir}...")
    for f in train_files:
        shutil.copy(f, train_dir / f.name)
    for f in val_files:
        shutil.copy(f, val_dir / f.name)
    for f in test_files:
        shutil.copy(f, test_dir / f.name)

    # Show sample works in each split
    print(f"\nSample training works: {', '.join(sorted(train_work_ids)[:5])}")
    print(f"Sample validation works: {', '.join(sorted(val_work_ids)[:5])}")
    print(f"Sample test works: {', '.join(sorted(test_work_ids)[:5])}")

    print(f"\n{'='*70}")
    print("✓ SPLIT COMPLETE")
    print(f"{'='*70}")
    print(f"Output directories:")
    print(f"  Training:   {train_dir}")
    print(f"  Validation: {val_dir}")
    print(f"  Test:       {test_dir}")

    return output_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Split Mozart/Haydn/Beethoven dataset BY WORK (keeps movements together)"
    )
    parser.add_argument("--tsv_dir", default="./mozart_tsv_fixed",
                        help="Directory with TSV files")
    parser.add_argument("--output_dir", default="./mozart_dataset_fixed",
                        help="Output directory")
    parser.add_argument("--n_train", type=int, default=44,
                        help="Number of training WORKS (not files)")
    parser.add_argument("--n_val", type=int, default=30,
                        help="Number of validation WORKS (not files)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    split_by_work(args.tsv_dir, args.output_dir,
                  args.n_train, args.n_val, args.seed)
