#!/usr/bin/env python3
"""
Check what vocabulary size the pretrained model uses.
This tells us which dataset version to use for finetuning.
"""

import os
import wandb

# The pretrained model artifact
ARTIFACT_NAME = "kb9520-princeton-university/chord_rec/model-kvd0jic5:v0"

print("Downloading and inspecting pretrained model...")
print(f"Artifact: {ARTIFACT_NAME}\n")

# Download the artifact
api = wandb.Api()
artifact = api.artifact(ARTIFACT_NAME, type="model")
artifact_dir = artifact.download(root="./artifacts")

print(f"Downloaded to: {artifact_dir}")

# Find the checkpoint file
import glob
ckpt_files = glob.glob(os.path.join(artifact_dir, "*.ckpt"))
if not ckpt_files:
    print("ERROR: No .ckpt file found in artifact!")
    exit(1)

ckpt_path = ckpt_files[0]
print(f"Checkpoint: {ckpt_path}\n")

# Load and inspect
import torch
ckpt = torch.load(ckpt_path, map_location="cpu")
hparams = ckpt.get("hyper_parameters", {})

print("="*70)
print("PRETRAINED MODEL HYPERPARAMETERS")
print("="*70)

# Print all tasks
tasks = hparams.get("tasks", {})
if tasks:
    print("\nTask vocabulary sizes:")
    for task, size in tasks.items():
        marker = " ← KEY!" if task == "romanNumeral" else ""
        print(f"  {task:15s}: {size}{marker}")
else:
    print("  ERROR: No 'tasks' found in hyperparameters!")

# Print other important params
print(f"\nOther important parameters:")
print(f"  in_feats:      {hparams.get('in_feats', 'NOT FOUND')}")
print(f"  n_hidden:      {hparams.get('n_hidden', 'NOT FOUND')}")
print(f"  n_layers:      {hparams.get('n_layers', 'NOT FOUND')}")
print(f"  lr:            {hparams.get('lr', 'NOT FOUND')}")
print(f"  weight_decay:  {hparams.get('weight_decay', 'NOT FOUND')}")

# Determine which dataset to use
print("\n" + "="*70)
print("FINETUNING RECOMMENDATIONS")
print("="*70)

if "romanNumeral" in tasks:
    rn_size = tasks["romanNumeral"]

    if rn_size == 31:
        print("\n✓ Pretrained model uses romanNumeral: 31")
        print("  → Use version != 'v1.0.0' (e.g., 'v2.0.0')")
        print("  → This loads Augmented2022ChordGraphDataset")
    elif rn_size == 76:
        print("\n✓ Pretrained model uses romanNumeral: 76")
        print("  → Use version='v1.0.0'")
        print("  → This loads AugmentedNetChordGraphDataset")
    else:
        print(f"\n⚠ Unknown romanNumeral size: {rn_size}")
        print("  → You may need to use pretrained_tasks directly")

    print(f"\nIn finetune_from_checkpoint.py, change:")
    print(f"  Line 54: DATA_VERSION = ??? (set appropriately)")
    print(f"  Line 126: tasks = ckpt['hyper_parameters']['tasks']  # Use pretrained!")
else:
    print("\n⚠ No romanNumeral task found - unexpected!")

print("\n" + "="*70)
