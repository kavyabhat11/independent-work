#!/usr/bin/env python3
"""
Finetune pretrained ChordGNN on Mozart data.
Uses existing checkpoint and continues training.
"""

import os
import shutil
import glob

print("\n" + "="*70)
print("Mozart Finetuning from Pretrained Checkpoint")
print("="*70 + "\n")

# Step 1: Setup Mozart data in cache
print("Setting up Mozart dataset...")

cache_root = "/scratch/network/kb9520/chordgnn_data"
cache = os.path.join(cache_root, "AugmentedNetChordDataset", "dataset")

#cache = os.path.expanduser("~/.chordgnn/AugmentedNetChordDataset/dataset")
if os.path.exists(cache):
    shutil.rmtree(cache)

os.makedirs(os.path.join(cache, "training"), exist_ok=True)
os.makedirs(os.path.join(cache, "validation"), exist_ok=True)
os.makedirs(os.path.join(cache, "test"), exist_ok=True)

# Copy Mozart TSVs
mozart = "/Users/kavyabhat/mozart_dataset"
for tsv in glob.glob(f"{mozart}/training/*.tsv"):
    shutil.copy(tsv, os.path.join(cache, "training"))
for tsv in glob.glob(f"{mozart}/validation/*.tsv"):
    shutil.copy(tsv, os.path.join(cache, "validation"))
for tsv in glob.glob(f"{mozart}/test/*.tsv"):
    shutil.copy(tsv, os.path.join(cache, "test"))

train_ct = len(glob.glob(f"{cache}/training/*.tsv"))
val_ct = len(glob.glob(f"{cache}/validation/*.tsv"))
test_ct = len(glob.glob(f"{cache}/test/*.tsv"))
print(f"✓ Data ready: {train_ct} train, {val_ct} val, {test_ct} test\n")

# Step 2: Load and finetune
import chordgnn as st
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

torch.manual_seed(0)

print("Loading dataset...")
datamodule = st.data.AugmentedGraphDatamodule(
    num_workers=4, include_synth=False, num_tasks=11,
    collection="all", batch_size=4, version="v1.0.0")

datamodule.setup()

print(f"✓ Training: {len(datamodule.dataset_train)} samples (with augmentation)")
print(f"✓ Test: {len(datamodule.dataset_test)} samples\n")

# Load FROM CHECKPOINT
ckpt_path = "./artifacts/model-kvd0jic5:v0/model.ckpt"
print(f"Loading pretrained model from: {ckpt_path}")

model = st.models.chord.ChordPrediction.load_from_checkpoint(
    ckpt_path,
    features=datamodule.features,
    tasks=datamodule.tasks,
    lr=0.0005,  # Lower LR for finetuning
    weight_decay=0.0035
)

print("✓ Pretrained model loaded!\n")

checkpoint = ModelCheckpoint(
    save_top_k=1, monitor="val_loss", mode="min",
    filename="mozart-finetune-{epoch:02d}-{val_loss:.3f}")
early_stop = EarlyStopping(monitor="val_loss", patience=10, mode="min")

trainer = Trainer(
    max_epochs=30,  # Fewer epochs for finetuning
    accelerator="auto",
    devices=[0] if torch.cuda.is_available() else None,
    callbacks=[checkpoint, early_stop],
    reload_dataloaders_every_n_epochs=5
)

print("="*70)
print("Starting finetuning on Mozart data...")
print("="*70 + "\n")

trainer.fit(model, datamodule)

print("\n" + "="*70)
print("Testing...")
print("="*70 + "\n")

trainer.test(model, datamodule, ckpt_path=checkpoint.best_model_path)

print("\n" + "="*70)
print("✓ COMPLETE!")
print(f"Finetuned model: {checkpoint.best_model_path}")
print("="*70 + "\n")
