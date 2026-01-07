#!/usr/bin/env python3
"""
Test pretrained ChordGNN checkpoint on Mozart validation data.
Get baseline accuracy BEFORE any finetuning.
"""

import os
import torch
import chordgnn as st
from pytorch_lightning import Trainer

# Same config as finetuning script
WANDB_ARTIFACT = "melkisedeath/chord_rec/model-kvd0jic5:v0"
ARTIFACT_ROOT = "./artifacts"
MOZART_ROOT = os.environ.get("MOZART_ROOT", "./mozart_dataset")
CACHE_ROOT = "/scratch/network/kb9520/chordgnn_data"
BATCH_SIZE = 1
NUM_WORKERS = 8

print("="*70)
print("TESTING PRETRAINED CHECKPOINT ON MOZART DATA")
print("="*70)
print()

# Step 1: Download checkpoint
print("Step 1: Downloading pretrained checkpoint...")
import wandb
api = wandb.Api()
artifact = api.artifact(WANDB_ARTIFACT, type="model")
artifact_dir = artifact.download(root=ARTIFACT_ROOT)
PRETRAINED_CKPT = os.path.join(artifact_dir, "model.ckpt")
print(f"✓ Checkpoint: {PRETRAINED_CKPT}\n")

# Step 2: Load checkpoint to inspect
print("Step 2: Inspecting checkpoint...")
ckpt = torch.load(PRETRAINED_CKPT, map_location="cpu")
state_dict = ckpt.get("state_dict", {})
pretrained_hparams = ckpt.get("hyper_parameters", {})

# HARDCODED: We know from finetuning script it's 31 classes
# Auto-detection was picking up hidden layers (256) instead of vocab size (31)
rn_vocab_size = 31
DATA_VERSION = "v2.0.0"
print(f"✓ Using DATA_VERSION={DATA_VERSION} (31-class RomanNumeral vocab)\n")

# Step 3: Prepare Mozart validation data
print("Step 3: Preparing Mozart validation data...")
import glob
import shutil

dataset_dir = os.path.join(CACHE_ROOT,
                           "AugmentedNetLatestChordDataset" if DATA_VERSION == "v2.0.0" else "AugmentedNetChordDataset",
                           "dataset")

# Clean existing cache
if os.path.exists(dataset_dir):
    print(f"Cleaning existing dataset cache: {dataset_dir}")
    shutil.rmtree(dataset_dir)

os.makedirs(dataset_dir, exist_ok=True)
os.makedirs(os.path.join(dataset_dir, "validation"), exist_ok=True)

# Copy ONLY validation files
for tsv in glob.glob(f"{MOZART_ROOT}/validation/*.tsv"):
    shutil.copy(tsv, os.path.join(dataset_dir, "validation"))

val_ct = len(glob.glob(f"{dataset_dir}/validation/*.tsv"))
print(f"✓ Copied {val_ct} validation files\n")

# Step 4: Create dataset
print("Step 4: Creating dataset...")
if DATA_VERSION == "v1.0.0":
    dataset = st.data.datasets.chord.AugmentedNetChordGraphDataset(
        raw_dir=CACHE_ROOT,
        force_reload=True,
        nprocs=max(1, NUM_WORKERS),
        include_synth=False,
        num_tasks=11,
        collection="all",
    )
else:
    dataset = st.data.datasets.chord.Augmented2022ChordGraphDataset(
        raw_dir=CACHE_ROOT,
        force_reload=True,
        nprocs=NUM_WORKERS,
        include_synth=False,
        num_tasks=11,
        collection="all",
    )

print(f"✓ Dataset created with {len(dataset.graphs)} graphs\n")

# Step 5: Load model from checkpoint
print("Step 5: Loading pretrained model...")
from pytorch_lightning import LightningModule

# Get tasks and architecture from pretrained checkpoint
if DATA_VERSION == "v1.0.0":
    from chordgnn.utils.chord_representations import available_representations
else:
    from chordgnn.utils.chord_representations_latest import available_representations

pretrained_tasks = available_representations

# Get architecture params
pretrained_n_hidden = int(pretrained_hparams.get("n_hidden", 512))
pretrained_n_layers = int(pretrained_hparams.get("n_layers", 6))
pretrained_dropout = float(pretrained_hparams.get("dropout", 0.5))
pretrained_use_nade = bool(pretrained_hparams.get("use_nade", False))
pretrained_use_jk = bool(pretrained_hparams.get("use_jk", False))
pretrained_use_rotograd = bool(pretrained_hparams.get("use_rotograd", False))

# Get in_feats from dataset
b0 = next(iter(torch.utils.data.DataLoader(dataset, batch_size=1)))
in_feats = int(b0[0].shape[-1])

print(f"Architecture: n_hidden={pretrained_n_hidden}, n_layers={pretrained_n_layers}")

# Build model
checkpoint_keys = list(state_dict.keys())
has_frozen_model = any(k.startswith("frozen_model.") for k in checkpoint_keys)

if has_frozen_model:
    print("Using PostChordPrediction architecture")
    model = st.models.PostChordPrediction(
        tasks=pretrained_tasks,
        in_feats=in_feats,
        n_hidden=pretrained_n_hidden,
        n_layers=pretrained_n_layers,
        dropout=pretrained_dropout,
        use_nade=pretrained_use_nade,
        use_jk=pretrained_use_jk,
        use_rotograd=pretrained_use_rotograd,
        lr=1e-3,
        weight_decay=1e-4,
        device="cpu",
    )
else:
    print("Using ChordPrediction architecture")
    model = st.models.ChordPrediction(
        tasks=pretrained_tasks,
        in_feats=in_feats,
        n_hidden=pretrained_n_hidden,
        n_layers=pretrained_n_layers,
        dropout=pretrained_dropout,
        use_nade=pretrained_use_nade,
        use_jk=pretrained_use_jk,
        use_rotograd=pretrained_use_rotograd,
        lr=1e-3,
        weight_decay=1e-4,
    )

# Load weights
cleaned_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("train_loss.") or k.startswith("val_loss.") or k.startswith("test_loss."):
        continue
    if not has_frozen_model and k.startswith("module."):
        cleaned_state_dict[k[7:]] = v
    else:
        cleaned_state_dict[k] = v

model.load_state_dict(cleaned_state_dict, strict=False)
print("✓ Model loaded\n")

# Step 6: Run evaluation
print("="*70)
print("RUNNING EVALUATION ON VALIDATION SET")
print("="*70)
print()

# Put model in eval mode
model.eval()
model = model.cuda() if torch.cuda.is_available() else model

# Create simple dataloader
from torch.utils.data import DataLoader
val_loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)

# Track metrics
from collections import defaultdict
correct_by_task = defaultdict(int)
total_by_task = defaultdict(int)

with torch.no_grad():
    for batch_idx, batch in enumerate(val_loader):
        batch_inputs, edges, edge_type, batch_label, onset_div, name = batch

        # Move to device
        device = model.device if hasattr(model, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch_inputs = batch_inputs.squeeze(0).float().to(device)
        edges = edges.squeeze(0).to(device)
        edge_type = edge_type.squeeze(0).to(device)
        onset_div = onset_div.squeeze().to(device)

        # Prepare labels
        batch_labels = batch_label.squeeze(0)
        batch_label_dict = {task: batch_labels[:, i].squeeze().long().to(device)
                           for i, task in enumerate(available_representations.keys())}

        # Forward pass
        from chordgnn.utils.hgraph import add_reverse_edges_from_edge_index
        edges, edge_type = add_reverse_edges_from_edge_index(edges, edge_type)

        if has_frozen_model:
            preds = model.frozen_model(batch_inputs, edges, edge_type, onset_div)
        else:
            preds = model(batch_inputs, edges, edge_type, onset_div)

        # Compute accuracy for each task
        for task_name, pred in preds.items():
            if task_name in batch_label_dict:
                target = batch_label_dict[task_name]
                pred_classes = pred.argmax(dim=-1)
                correct = (pred_classes == target).sum().item()
                total = target.numel()

                correct_by_task[task_name] += correct
                total_by_task[task_name] += total

        if (batch_idx + 1) % 10 == 0:
            print(f"Processed {batch_idx + 1}/{len(val_loader)} batches...")

print("\n" + "="*70)
print("BASELINE RESULTS (Pretrained on Mozart Validation)")
print("="*70)

for task in sorted(correct_by_task.keys()):
    if total_by_task[task] > 0:
        acc = 100.0 * correct_by_task[task] / total_by_task[task]
        print(f"{task:20s}: {acc:6.2f}% ({correct_by_task[task]}/{total_by_task[task]})")

# Highlight Roman numeral
if 'romanNumeral' in correct_by_task:
    rn_acc = 100.0 * correct_by_task['romanNumeral'] / total_by_task['romanNumeral']
    print(f"\n{'='*70}")
    print(f"ROMAN NUMERAL BASELINE: {rn_acc:.2f}%")
    print(f"{'='*70}")

print("\n✓ Baseline evaluation complete!")
print("This is the accuracy to BEAT with finetuning.")
