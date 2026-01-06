#!/usr/bin/env python3
"""
Finetune pretrained ChordGNN on Mozart data (encoder-only load) + W&B logging.

What this script does:
1) Rebuilds Mozart TSV cache into the dataset cache directory ChordGNN expects.
2) Loads AugmentedGraphDatamodule (AugmentedNetChordDataset).
3) Builds ChordPrediction with REQUIRED args (in_feats, n_hidden, tasks, n_layers).
4) Loads ONLY pretrained encoder weights from the checkpoint (avoids head/loss mismatch).
5) Sanitizes ANY NaN/Inf in batch floating tensors for TRAIN/VAL/TEST (prevents loss=nan).
6) Trains up to 40 epochs with EarlyStopping on val_loss + ModelCheckpoint(best val_loss).
7) Tests best checkpoint.
8) Logs to W&B project: chord_rec and logs best ckpt as a W&B artifact.

Run:
  python finetune_from_checkpoint.py
"""

import os
import glob
import shutil

import torch
import chordgnn as st

import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import Callback


print("\n" + "=" * 70)
print("Mozart Finetuning (Encoder-only init + NaN sanitize + EarlyStopping + W&B)")
print("=" * 70 + "\n")

torch.manual_seed(0)

# -------------------------
# Config
# -------------------------
PRETRAINED_CKPT = "./artifacts/model-kvd0jic5:v0/model.ckpt"
MOZART_ROOT = "./mozart_dataset"

CACHE_ROOT = "/scratch/network/kb9520/chordgnn_data"
CACHE = os.path.join(CACHE_ROOT, "AugmentedNetChordDataset", "dataset")

LR = 5e-4
WEIGHT_DECAY = 3.5e-3
MAX_EPOCHS = 40
BATCH_SIZE = 4
NUM_WORKERS = 8
NUM_TASKS = 11
DATA_VERSION = "v1.0.0"

WANDB_PROJECT = "chord_rec"
WANDB_RUN_NAME = "mozart-finetune-earlystop"


# -------------------------
# Step 1: Setup Mozart data
# -------------------------
print("Setting up Mozart dataset cache...")

if os.path.exists(CACHE):
    shutil.rmtree(CACHE)

os.makedirs(os.path.join(CACHE, "training"), exist_ok=True)
os.makedirs(os.path.join(CACHE, "validation"), exist_ok=True)
os.makedirs(os.path.join(CACHE, "test"), exist_ok=True)

for tsv in glob.glob(f"{MOZART_ROOT}/training/*.tsv"):
    shutil.copy(tsv, os.path.join(CACHE, "training"))
for tsv in glob.glob(f"{MOZART_ROOT}/validation/*.tsv"):
    shutil.copy(tsv, os.path.join(CACHE, "validation"))
for tsv in glob.glob(f"{MOZART_ROOT}/test/*.tsv"):
    shutil.copy(tsv, os.path.join(CACHE, "test"))

train_ct = len(glob.glob(f"{CACHE}/training/*.tsv"))
val_ct = len(glob.glob(f"{CACHE}/validation/*.tsv"))
test_ct = len(glob.glob(f"{CACHE}/test/*.tsv"))
print(f"✓ Data ready: {train_ct} train, {val_ct} val, {test_ct} test\n")


# -------------------------
# Step 2: Load datamodule
# -------------------------
print("Loading dataset...")
datamodule = st.data.AugmentedGraphDatamodule(
    num_workers=NUM_WORKERS,
    include_synth=False,
    num_tasks=NUM_TASKS,
    collection="all",
    batch_size=BATCH_SIZE,
    version=DATA_VERSION,
)
datamodule.setup()

print(f"✓ Training: {len(datamodule.dataset_train)} samples (with augmentation)")
print(f"✓ Val: {len(datamodule.dataset_val)} samples")
print(f"✓ Test: {len(datamodule.dataset_test)} samples\n")


# -------------------------
# Step 3: W&B logger
# -------------------------
wandb_logger = WandbLogger(
    project=WANDB_PROJECT,
    name=WANDB_RUN_NAME,
    log_model=True,
)


# -------------------------
# Step 4: Build model + load encoder weights
# -------------------------
print(f"Loading pretrained encoder from: {PRETRAINED_CKPT}")

ckpt = torch.load(PRETRAINED_CKPT, map_location="cpu")
state_dict = ckpt["state_dict"]

hparams = {}
if "hyper_parameters" in ckpt and isinstance(ckpt["hyper_parameters"], dict):
    hparams = ckpt["hyper_parameters"]

tasks = datamodule.tasks

# Determine in_feats robustly
if hasattr(datamodule, "in_feats"):
    in_feats = int(datamodule.in_feats)
elif "in_feats" in hparams:
    in_feats = int(hparams["in_feats"])
elif hasattr(datamodule, "features") and hasattr(datamodule.features, "in_feats"):
    in_feats = int(datamodule.features.in_feats)
else:
    b0 = next(iter(datamodule.train_dataloader()))
    in_feats = int(b0[0].shape[-1])

n_hidden = int(hparams.get("n_hidden", 256))
n_layers = int(hparams.get("n_layers", 2))

print("Model init params:")
print("  in_feats =", in_feats)
print("  n_hidden =", n_hidden)
print("  n_layers =", n_layers)
print("  num_tasks =", len(tasks))
print("  lr =", LR)
print("  weight_decay =", WEIGHT_DECAY)
print()

model = st.models.chord.ChordPrediction(
    in_feats=in_feats,
    n_hidden=n_hidden,
    tasks=tasks,
    n_layers=n_layers,
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)

# Detect encoder prefix in current model
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
    print("\n[DEBUG] Could not auto-detect encoder prefix. First 100 model keys:\n")
    for k in model_keys[:100]:
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

wandb_logger.experiment.config.update({
    "dataset": "Mozart",
    "finetune": True,
    "pretrained_ckpt": PRETRAINED_CKPT,
    "max_epochs": MAX_EPOCHS,
    "batch_size": BATCH_SIZE,
    "num_workers": NUM_WORKERS,
    "num_tasks": len(tasks),
    "in_feats": in_feats,
    "n_hidden": n_hidden,
    "n_layers": n_layers,
    "lr": LR,
    "weight_decay": WEIGHT_DECAY,
    "precision": 32,
    "grad_clip": 1.0,
    "earlystop_patience": 5,
    "earlystop_min_delta": 1e-4,
    "nan_sanitize": "all_floats_train_val_test",
})


# -------------------------
# Step 5: NaN/Inf sanitize callback (TRAIN + VAL + TEST)
# -------------------------
class SanitizeBatchNaNs(Callback):
    """
    Replace any NaN/Inf in floating tensors inside the batch with 0.0.
    Runs for TRAIN/VAL/TEST so train_loss/val_loss won't become NaN.
    """
    def __init__(self, verbose_first_k=5):
        super().__init__()
        self.verbose_first_k = verbose_first_k
        self._printed = 0

    def _scan_and_fix(self, obj, path="batch"):
        if torch.is_tensor(obj):
            if obj.dtype.is_floating_point:
                bad = torch.isnan(obj) | torch.isinf(obj)
                if bad.any():
                    if self._printed < self.verbose_first_k:
                        n = bad.sum().item()
                        print(f"[SanitizeBatchNaNs] Fixed {n} NaN/Inf in {path} (shape={tuple(obj.shape)})")
                        self._printed += 1
                    obj[bad] = 0.0
            return

        if isinstance(obj, dict):
            for k, v in obj.items():
                self._scan_and_fix(v, f"{path}.{k}")
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                self._scan_and_fix(v, f"{path}[{i}]")

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._scan_and_fix(batch)

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        self._scan_and_fix(batch)

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        self._scan_and_fix(batch)


# -------------------------
# Step 6: Train with EarlyStopping + checkpointing
# -------------------------
checkpoint = ModelCheckpoint(
    save_top_k=1,
    monitor="val_loss",
    mode="min",
    filename="mozart-finetune-{epoch:02d}-{val_loss:.3f}",
)

early_stop = EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=5,     # stop after 5 epochs with no meaningful improvement
    min_delta=1e-4, # require at least this much improvement
    verbose=True,
)

trainer = Trainer(
    max_epochs=MAX_EPOCHS,
    accelerator="auto",
    devices=[0] if torch.cuda.is_available() else None,
    callbacks=[checkpoint, early_stop, SanitizeBatchNaNs(verbose_first_k=5)],
    reload_dataloaders_every_n_epochs=5,
    gradient_clip_val=1.0,
    precision=32,
    logger=wandb_logger,
)

print("=" * 70)
print("Starting finetuning on Mozart data...")
print("=" * 70 + "\n")

trainer.fit(model, datamodule)

print("\nBEST CKPT:", checkpoint.best_model_path)
print("BEST SCORE:", checkpoint.best_model_score)

# Log the best ckpt as a W&B artifact so analyze_score.py can load it by artifact name/version
if checkpoint.best_model_path and os.path.exists(checkpoint.best_model_path):
    artifact = wandb.Artifact(
        name="mozart-finetuned-model",
        type="model",
        description="ChordGNN finetuned on Mozart data (best val checkpoint)"
    )
    artifact.add_file(checkpoint.best_model_path)
    wandb_logger.experiment.log_artifact(artifact)
    print("✓ Logged W&B artifact: mozart-finetuned-model (see W&B for version)\n")
else:
    print("WARNING: No best_model_path found; skipping artifact logging.\n")


# -------------------------
# Step 7: Test best checkpoint
# -------------------------
print("\n" + "=" * 70)
print("Testing...")
print("=" * 70 + "\n")

trainer.test(model, datamodule, ckpt_path=checkpoint.best_model_path)

print("\n" + "=" * 70)
print("✓ COMPLETE!")
print(f"Finetuned model: {checkpoint.best_model_path}")
print("=" * 70 + "\n")

wandb.finish()
