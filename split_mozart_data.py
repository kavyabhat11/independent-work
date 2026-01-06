#!/usr/bin/env python3
"""
Split Mozart TSV files into train/val/test sets.
"""

import os
import shutil
import random
from pathlib import Path

def split_mozart_dataset(tsv_dir, output_dir, n_train=30, n_val=5, seed=42):
    """
    Split Mozart TSV files into train/val/test directories.

    Args:
        tsv_dir: Directory containing Mozart TSV files
        output_dir: Output directory for split dataset
        n_train: Number of training files
        n_val: Number of validation files
        seed: Random seed for reproducibility
    """
    random.seed(seed)

    tsv_path = Path(tsv_dir)
    tsv_files = list(tsv_path.glob("*.tsv"))

    print(f"Found {len(tsv_files)} TSV files in {tsv_dir}")

    if len(tsv_files) < (n_train + n_val + 1):
        print(f"\nWARNING: Only {len(tsv_files)} files available!")
        print(f"Requested: {n_train} train + {n_val} val = {n_train + n_val} minimum")
        print("Adjusting splits...")
        total = len(tsv_files)
        n_train = min(n_train, total - 6)
        n_val = min(n_val, total - n_train - 1)

    # Shuffle and split
    random.shuffle(tsv_files)
    train_files = tsv_files[:n_train]
    val_files = tsv_files[n_train:n_train + n_val]
    test_files = tsv_files[n_train + n_val:]

    print(f"\nDataset split:")
    print(f"  Training:   {len(train_files)} files")
    print(f"  Validation: {len(val_files)} files")
    print(f"  Test:       {len(test_files)} files")

    # Create output directories
    output_path = Path(output_dir)
    train_dir = output_path / "training"
    val_dir = output_path / "validation"
    test_dir = output_path / "test"

    for dir in [train_dir, val_dir, test_dir]:
        dir.mkdir(parents=True, exist_ok=True)

    # Copy files
    print("\nCopying files...")
    for f in train_files:
        shutil.copy(f, train_dir / f.name)
    for f in val_files:
        shutil.copy(f, val_dir / f.name)
    for f in test_files:
        shutil.copy(f, test_dir / f.name)

    print(f"\n✓ Dataset created at: {output_dir}")
    print(f"  - Training:   {train_dir}")
    print(f"  - Validation: {val_dir}")
    print(f"  - Test:       {test_dir}")

    return output_dir

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Split Mozart dataset")
    parser.add_argument("--tsv_dir", default="./mozart_tsv",
                        help="Directory with TSV files")
    parser.add_argument("--output_dir", default="./mozart_dataset",
                        help="Output directory")
    parser.add_argument("--n_train", type=int, default=30,
                        help="Number of training files")
    parser.add_argument("--n_val", type=int, default=5,
                        help="Number of validation files")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    split_mozart_dataset(args.tsv_dir, args.output_dir,
                        args.n_train, args.n_val, args.seed)
