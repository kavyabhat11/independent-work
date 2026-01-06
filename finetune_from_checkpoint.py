#!/usr/bin/env python3
"""
Finetune pretrained ChordGNN on Mozart data.

This version:
- Rebuilds Mozart cache
- Loads datamodule
- Loads checkpoint
- Instantiates ChordPrediction using REQUIRED args pulled from the checkpoint when possible
- Loads ONLY encoder weights (remaps frozen_model.encoder.* -> current model encoder prefix)
- Trains + tests
"""

import os
import shutil
import glob

print("\n" + "=" * 70)
print("Mozart Finetuning from Pretrained Checkpoint (Encoder-only load)")
print("=" * 70 + "\n")

# -------------------------
# Step 1: Setup Mozart data
# -------------------------
print("Setting up Mozart dataset...")

cache_root = "/scratch/network/kb9520/chordgnn_data"
cache = os.path.join(cache_root, "AugmentedNetChordDataset", "dataset")

if os.path.exists(cache):
    shutil.rmtree(cache)

os.makedirs(os.path.join(cache, "training"), exist_ok=True)
os.makedirs(os.path.join(cache, "validation"), exist_ok=True)
os.makedirs(os.path.join(cache, "test"), exist_ok=True)

mozart = "./mozart_dataset"
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

# -------------------------
# Step 2: Load datamodule
# -------------------------
import torch
import chordgnn as st
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

torch.manual_seed(0)

print("Loading dataset...")
datamodule = st.data.AugmentedGraphDatamodule(
    num_workers=8,
    include_synth=False,
    num_tasks=11,
    collection="all",
    batch_size=4,
    version="v1.0.0",
)
datamodule.setup()

print(f"✓ Training: {len(datamodule.dataset_train)} samples (with augmentation)")
print(f"✓ Val: {len(datamodule.dataset_val)} samples")
print(f"✓ Test: {len(datamodule.dataset_test)} samples\n")

# -------------------------
# Step 3: Instantiate model (required args) + load encoder weights
# -------------------------
ckpt_path = "./artifacts/model-kvd0jic5:v0/model.ckpt"
print(f"Loading pretrained encoder from: {ckpt_path}")

ckpt = torch.load(ckpt_path, map_location="cpu")

# Try to recover hparams used to build the pretrained model.
# Lightning often stores these under one of these keys:
hparams = {}
for key in ("hyper_parameters", "hparams", "pytorch-lightning_version"):
    if key in ckpt and isinstance(ckpt[key], dict):
        hparams = ckpt[key]
        break

# Fallback: some checkpoints store them at ckpt["hyper_parameters"]
if not hparams and "hyper_parameters" in ckpt and isinstance(ckpt["hyper_parameters"], dict):
    hparams = ckpt["hyper_parameters"]

# Determine required args
# - tasks: MUST be your current datamodule.tasks (11 tasks), otherwise you’ll reintroduce mismatch
tasks = datamodule.tasks

# - in_feats: best is from datamodule if exposed; otherwise from checkpoint hparams
if hasattr(datamodule, "in_feats"):
    in_feats = int(datamodule.in_feats)
elif "in_feats" in hparams:
    in_feats = int(hparams["in_feats"])
else:
    # last resort: infer from features object if it has a dim
    if hasattr(datamodule, "features") and hasattr(datamodule.features, "in_feats"):
        in_feats = int(datamodule.features.in_feats)
    else:
        raise RuntimeError("Could not determine in_feats from datamodule or checkpoint. Print datamodule.features for clues.")

# - n_hidden and n_layers: take from checkpoint if available, else choose safe defaults
n_hidden = int(hparams.get("n_hidden", 256))
n_layers = int(hparams.get("n_layers", 2))

print("Model init params:")
print("  in_feats =", in_feats)
print("  n_hidden =", n_hidden)
print("  n_layers =", n_layers)
print("  num_tasks =", len(tasks))

# Create model (your class REQUIRES these args)
model = st.models.chord.ChordPrediction(
    in_feats=in_feats,
    n_hidden=n_hidden,
    tasks=tasks,
    n_layers=n_layers,
    lr=0.0005,
    weight_decay=0.0035,
)

state_dict = ckpt["state_dict"]

# Auto-detect encoder prefix in current model
model_keys = list(model.state_dict().keys())

def find_encoder_prefix(keys):
    needles = [
        "encoder.spelling_embedding.weight",
        "encoder.pitch_embedding.weight",
        "encoder.embedding.weight",
        "encoder.encoder.layers.0",
    ]
    for needle in needles:
        for k in keys:
            if needle in k:
                i = k.find("encoder.")
                return k[:i]
    return None

target_prefix = find_encoder_prefix(model_keys)
if target_prefix is None:
    print("\n[DEBUG] Could not auto-detect encoder prefix. First 80 model keys:\n")
    for k in model_keys[:80]:
        print(k)
    raise RuntimeError("Could not find encoder.* keys in current model.state_dict().")

encoder_state = {}

def add_if_encoder(src_key, tensor):
    j = src_key.find("encoder.")
    if j == -1:
        return
    rest = src_key[j + len("encoder."):]
    dst_key = f"{target_prefix}encoder.{rest}"
    encoder_state[dst_key] = tensor

for k, v in state_dict.items():
    if k.startswith("train_loss."):
        continue
    if "encoder." in k and (k.startswith("frozen_model.") or k.startswith("module.") or k.startswith("encoder.")):
        add_if_encoder(k, v)

missing, unexpected = model.load_state_dict(encoder_state, strict=False)

print("✓ Pretrained encoder loaded (heads reinitialized for current tasks).")
print(f"  Detected target encoder prefix: {repr(target_prefix)}")
print(f"  Loaded encoder tensors: {len(encoder_state)}")
print(f"  Missing keys (expected; heads/loss/etc): {len(missing)}")
print(f"  Unexpected keys: {len(unexpected)}\n")

# -------------------------
# Step 4: Train
# -------------------------
checkpoint = ModelCheckpoint(
    save_top_k=1,
    monitor="val_loss",
    mode="min",
    filename="mozart-finetune-{epoch:02d}-{val_loss:.3f}",
)
early_stop = EarlyStopping(monitor="val_loss", patience=10, mode="min")

trainer = Trainer(
    max_epochs=30,
    accelerator="auto",
    devices=[0] if torch.cuda.is_available() else None,
    callbacks=[checkpoint, early_stop],
    reload_dataloaders_every_n_epochs=5,
)

print("=" * 70)
print("Starting finetuning on Mozart data...")
print("=" * 70 + "\n")

trainer.fit(model, datamodule)

# -------------------------
# Step 5: Test best checkpoint
# -------------------------
print("\n" + "=" * 70)
print("Testing...")
print("=" * 70 + "\n")

trainer.test(model, datamodule, ckpt_path=checkpoint.best_model_path)

print("\n" + "=" * 70)
print("✓ COMPLETE!")
print(f"Finetuned model: {checkpoint.best_model_path}")
print("=" * 70 + "\n")
