#!/usr/bin/env python3
"""
Finetune pretrained ChordGNN on Mozart data (FIXED VERSION)

CRITICAL FIXES:
1. Loads pretrained checkpoint FIRST to detect vocab size
2. Uses pretrained tasks (not datamodule tasks) to ensure architecture matches
3. Sets DATA_VERSION automatically based on pretrained romanNumeral vocab
4. Uses much lower learning rate (1e-5) to prevent catastrophic forgetting
5. Matches all architecture params (n_layers, n_hidden) from pretrained
6. Increased early stopping patience for small datasets

What this script does:
1) Loads pretrained checkpoint to inspect vocab and architecture
2) Sets DATA_VERSION to match pretrained vocab (31 vs 76 classes)
3) Rebuilds Mozart TSV cache into the dataset cache directory
4) Loads datamodule with CORRECT version (but ignores its task definitions)
5) Builds model using PRETRAINED tasks (ensures exact architecture match)
6) Loads pretrained weights (encoder + heads all load successfully now)
7) Sanitizes NaN/Inf in batches
8) Trains with lower LR, EarlyStopping, ModelCheckpoint
9) Tests and logs to W&B

Run:
  python finetune_from_checkpoint_FIXED.py
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
print("Mozart Finetuning (FIXED: Vocab-Matched + Lower LR + W&B)")
print("=" * 70 + "\n")

torch.manual_seed(0)

# -------------------------
# Config
# -------------------------
WANDB_ARTIFACT = "melkisedeath/chord_rec/model-kvd0jic5:v0"
ARTIFACT_ROOT = "./artifacts"
MOZART_ROOT = "./mozart_dataset"  # Should contain training/validation/test subdirs (from split_mozart_data.py)

CACHE_ROOT = "/scratch/network/kb9520/chordgnn_data"
CACHE = os.path.join(CACHE_ROOT, "AugmentedNetChordDataset", "dataset")

# FIXED: Much lower learning rate for finetuning (was 5e-4, caused catastrophic forgetting)
LR = 1e-5  # 50x smaller than before
WEIGHT_DECAY = 1e-4  # Also reduced (was 3.5e-3)

MAX_EPOCHS = 40
BATCH_SIZE = 4
NUM_WORKERS = 8
NUM_TASKS = 11

WANDB_PROJECT = "chord_rec"
WANDB_RUN_NAME = "mozart-finetune-FIXED"


# -------------------------
# Step 1: Download pretrained checkpoint if needed
# -------------------------
print("=" * 70)
print("STEP 1: Loading pretrained checkpoint")
print("=" * 70 + "\n")

# Construct expected local path
artifact_basename = os.path.basename(WANDB_ARTIFACT)  # "model-kvd0jic5:v0"
local_artifact_dir = os.path.join(ARTIFACT_ROOT, artifact_basename)
expected_ckpt_path = os.path.join(local_artifact_dir, "model.ckpt")

# Check if checkpoint exists locally
if os.path.exists(expected_ckpt_path):
    print(f"✓ Found local checkpoint: {expected_ckpt_path}")
    PRETRAINED_CKPT = expected_ckpt_path
else:
    print(f"Checkpoint not found locally at: {expected_ckpt_path}")
    print(f"Downloading from W&B: {WANDB_ARTIFACT}")

    api = wandb.Api()
    artifact = api.artifact(WANDB_ARTIFACT, type="model")
    downloaded_dir = artifact.download(root=ARTIFACT_ROOT)

    # Find the .ckpt file in downloaded directory
    ckpt_files = glob.glob(os.path.join(downloaded_dir, "*.ckpt"))
    if not ckpt_files:
        raise FileNotFoundError(f"No .ckpt file found in downloaded artifact: {downloaded_dir}")

    PRETRAINED_CKPT = ckpt_files[0]
    print(f"✓ Downloaded to: {PRETRAINED_CKPT}\n")

# -------------------------
# Step 2: Inspect pretrained checkpoint to detect vocab
# -------------------------
print("=" * 70)
print("STEP 2: Inspecting pretrained checkpoint to detect vocab")
print("=" * 70 + "\n")

print(f"Loading checkpoint: {PRETRAINED_CKPT}")
ckpt = torch.load(PRETRAINED_CKPT, map_location="cpu")
state_dict = ckpt["state_dict"]

pretrained_hparams = {}
if "hyper_parameters" in ckpt and isinstance(ckpt["hyper_parameters"], dict):
    pretrained_hparams = ckpt["hyper_parameters"]

pretrained_tasks = pretrained_hparams.get("tasks", None)
if pretrained_tasks is None:
    raise RuntimeError("Pretrained checkpoint missing 'hyper_parameters.tasks'")

print("\nPretrained model task vocabulary sizes:")
for task, size in pretrained_tasks.items():
    marker = " ← CRITICAL!" if task == "romanNumeral" else ""
    print(f"  {task:15s}: {size}{marker}")

if "romanNumeral" not in pretrained_tasks:
    raise RuntimeError("Pretrained checkpoint missing 'romanNumeral' task!")

rn_vocab_size = pretrained_tasks["romanNumeral"]
print(f"\n✓ Pretrained uses romanNumeral: {rn_vocab_size} classes")

# Auto-select DATA_VERSION to match pretrained vocab
if rn_vocab_size == 31:
    DATA_VERSION = "v2.0.0"  # or any string != "v1.0.0"
    print(f"  → Setting DATA_VERSION = '{DATA_VERSION}' (Augmented2022ChordGraphDataset)")
elif rn_vocab_size == 76:
    DATA_VERSION = "v1.0.0"
    print(f"  → Setting DATA_VERSION = '{DATA_VERSION}' (AugmentedNetChordGraphDataset)")
else:
    raise RuntimeError(f"Unexpected romanNumeral vocab size: {rn_vocab_size}")

# Extract other pretrained architecture params
pretrained_n_hidden = int(pretrained_hparams.get("n_hidden", 256))
pretrained_n_layers = int(pretrained_hparams.get("n_layers", 1))  # default 1, not 2!
pretrained_in_feats = int(pretrained_hparams.get("in_feats", 83))
pretrained_dropout = float(pretrained_hparams.get("dropout", 0.5))
pretrained_use_nade = bool(pretrained_hparams.get("use_nade", False))
pretrained_use_jk = bool(pretrained_hparams.get("use_jk", False))
pretrained_use_rotograd = bool(pretrained_hparams.get("use_rotograd", False))

print(f"\nPretrained architecture:")
print(f"  in_feats:     {pretrained_in_feats}")
print(f"  n_hidden:     {pretrained_n_hidden}")
print(f"  n_layers:     {pretrained_n_layers}")
print(f"  dropout:      {pretrained_dropout}")
print(f"  use_nade:     {pretrained_use_nade}")
print(f"  use_jk:       {pretrained_use_jk}")
print(f"  use_rotograd: {pretrained_use_rotograd}")
print()


# -------------------------
# Step 3: Setup Mozart data with custom raw_dir
# -------------------------
print("=" * 70)
print("STEP 3: Setting up Mozart dataset cache")
print("=" * 70 + "\n")

# CRITICAL FIX: Use CACHE_ROOT as raw_dir so dataset doesn't download full dataset
# Copy Mozart TSVs to the location that matches DATA_VERSION
# For v1.0.0 → AugmentedNetChordDataset
# For v2.0.0 → AugmentedNetLatestChordDataset
if DATA_VERSION == "v1.0.0":
    dataset_dir = os.path.join(CACHE_ROOT, "AugmentedNetChordDataset", "dataset")
else:
    dataset_dir = os.path.join(CACHE_ROOT, "AugmentedNetLatestChordDataset", "dataset")

if os.path.exists(dataset_dir):
    print(f"Cleaning existing dataset cache: {dataset_dir}")
    shutil.rmtree(dataset_dir)

os.makedirs(os.path.join(dataset_dir, "training"), exist_ok=True)
os.makedirs(os.path.join(dataset_dir, "validation"), exist_ok=True)
os.makedirs(os.path.join(dataset_dir, "test"), exist_ok=True)

print(f"Copying Mozart TSVs to: {dataset_dir}")
# Expect MOZART_ROOT to have training/validation/test subdirectories
# Use split_mozart_data.py to create this structure first
for tsv in glob.glob(f"{MOZART_ROOT}/training/*.tsv"):
    shutil.copy(tsv, os.path.join(dataset_dir, "training"))
for tsv in glob.glob(f"{MOZART_ROOT}/validation/*.tsv"):
    shutil.copy(tsv, os.path.join(dataset_dir, "validation"))
for tsv in glob.glob(f"{MOZART_ROOT}/test/*.tsv"):
    shutil.copy(tsv, os.path.join(dataset_dir, "test"))

train_ct = len(glob.glob(f"{dataset_dir}/training/*.tsv"))
val_ct = len(glob.glob(f"{dataset_dir}/validation/*.tsv"))
test_ct = len(glob.glob(f"{dataset_dir}/test/*.tsv"))
print(f"✓ Data ready: {train_ct} train, {val_ct} val, {test_ct} test\n")


# -------------------------
# Step 4: Load datamodule with custom raw_dir (to prevent full dataset download)
# -------------------------
print("=" * 70)
print("STEP 4: Creating custom Mozart datamodule")
print("=" * 70 + "\n")

print(f"Creating dataset with version='{DATA_VERSION}' and raw_dir='{CACHE_ROOT}'...")

# CRITICAL FIX: Create dataset directly with raw_dir to avoid downloading full dataset
if DATA_VERSION == "v1.0.0":
    dataset = st.data.datasets.chord.AugmentedNetChordGraphDataset(
        raw_dir=CACHE_ROOT,  # ← This prevents downloading the full dataset!
        force_reload=True,   # ← Force reprocessing of Mozart TSVs
        nprocs=max(1, NUM_WORKERS),
        include_synth=False,
        num_tasks=NUM_TASKS,
        collection="all",
    )
else:
    dataset = st.data.datasets.chord.Augmented2022ChordGraphDataset(
        raw_dir=CACHE_ROOT,  # ← This prevents downloading the full dataset!
        force_reload=True,   # ← Force reprocessing of Mozart TSVs
        nprocs=NUM_WORKERS,
        include_synth=False,
        num_tasks=NUM_TASKS,
        collection="all",
    )

# Create a minimal datamodule wrapper
from pytorch_lightning import LightningDataModule

class MozartDatamodule(LightningDataModule):
    def __init__(self, dataset, batch_size, num_workers):
        super().__init__()  # CRITICAL: Call parent __init__
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tasks = dataset.tasks
        self.features = dataset.features
        self.in_feats = dataset.features.in_feats if hasattr(dataset.features, 'in_feats') else None

    def setup(self, stage=None):
        # Split dataset into train/val/test based on filenames
        all_graphs = [(i, g) for i, g in enumerate(self.dataset.graphs)]

        # Determine split based on graph names
        train_idx = []
        val_idx = []
        test_idx = []

        print(f"\nSplitting {len(all_graphs)} graphs into train/val/test...")

        for i, g in all_graphs:
            name = g.name.lower()  # Case-insensitive matching

            # Debug: print first few names to see the pattern
            if i < 3:
                print(f"  Sample graph name: {g.name}")

            # Files in training/, validation/, test/ subdirs
            if 'training' in name:
                train_idx.append(i)
            elif 'validation' in name:
                val_idx.append(i)
            elif 'test' in name:
                test_idx.append(i)

        print(f"  → Found {len(train_idx)} training, {len(val_idx)} validation, {len(test_idx)} test samples")

        # Fallback: if split failed, use all data for training
        if len(train_idx) == 0 and len(val_idx) == 0 and len(test_idx) == 0:
            print("  ⚠ WARNING: No samples matched train/val/test split!")
            print("  → Using 80/10/10 split as fallback")
            n = len(all_graphs)
            train_idx = list(range(0, int(0.8 * n)))
            val_idx = list(range(int(0.8 * n), int(0.9 * n)))
            test_idx = list(range(int(0.9 * n), n))

        from torch.utils.data import Subset

        self.dataset_train = Subset(self.dataset, train_idx)
        self.dataset_val = Subset(self.dataset, val_idx)
        self.dataset_test = Subset(self.dataset, test_idx)

    def train_dataloader(self):
        from torch.utils.data import DataLoader
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None
        )

    def val_dataloader(self):
        from torch.utils.data import DataLoader
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None
        )

    def test_dataloader(self):
        from torch.utils.data import DataLoader
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None
        )

datamodule = MozartDatamodule(dataset, BATCH_SIZE, NUM_WORKERS)
datamodule.setup()

print(f"✓ Training: {len(datamodule.dataset_train)} samples (with augmentation)")
print(f"✓ Val: {len(datamodule.dataset_val)} samples")
print(f"✓ Test: {len(datamodule.dataset_test)} samples")

# Verify datamodule tasks match pretrained
datamodule_tasks = datamodule.tasks
print(f"\nDatamodule task vocab sizes:")
for task, size in datamodule_tasks.items():
    pretrained_size = pretrained_tasks.get(task, "MISSING")
    match = "✓" if size == pretrained_size else "✗ MISMATCH!"
    if pretrained_size == "MISSING":
        print(f"  {task:15s}: {size:3} (pretrained: {pretrained_size}) {match}")
    else:
        print(f"  {task:15s}: {size:3} (pretrained: {pretrained_size:3}) {match}")

# Check for critical mismatches
mismatches = []
for task, size in datamodule_tasks.items():
    if task in pretrained_tasks and size != pretrained_tasks[task]:
        mismatches.append(f"{task}: datamodule={size}, pretrained={pretrained_tasks[task]}")

if mismatches:
    print("\n⚠ WARNING: Task vocab mismatches detected:")
    for mm in mismatches:
        print(f"  • {mm}")
    print("\n  → We will use PRETRAINED tasks to build model (ignoring datamodule tasks)")
else:
    print("\n✓ All task vocabs match! Safe to use either.")

print()


# -------------------------
# Step 5: W&B logger
# -------------------------
wandb_logger = WandbLogger(
    project=WANDB_PROJECT,
    name=WANDB_RUN_NAME,
    log_model=True,
)


# -------------------------
# Step 6: Build model using PRETRAINED tasks
# -------------------------
print("=" * 70)
print("STEP 6: Building model with pretrained architecture")
print("=" * 70 + "\n")

# CRITICAL FIX: Use pretrained_tasks, NOT datamodule.tasks
tasks = pretrained_tasks

# Determine in_feats robustly (try pretrained first, then datamodule)
if "in_feats" in pretrained_hparams:
    in_feats = int(pretrained_hparams["in_feats"])
elif hasattr(datamodule, "in_feats"):
    in_feats = int(datamodule.in_feats)
elif hasattr(datamodule, "features") and hasattr(datamodule.features, "in_feats"):
    in_feats = int(datamodule.features.in_feats)
else:
    b0 = next(iter(datamodule.train_dataloader()))
    in_feats = int(b0[0].shape[-1])

# FIXED: Use pretrained params, not defaults
n_hidden = pretrained_n_hidden
n_layers = pretrained_n_layers
dropout = pretrained_dropout
use_nade = pretrained_use_nade
use_jk = pretrained_use_jk
use_rotograd = pretrained_use_rotograd

print("Building model with:")
print(f"  in_feats:       {in_feats}")
print(f"  n_hidden:       {n_hidden}")
print(f"  n_layers:       {n_layers}")
print(f"  dropout:        {dropout}")
print(f"  num_tasks:      {len(tasks)}")
print(f"  lr:             {LR} (50x smaller than before!)")
print(f"  weight_decay:   {WEIGHT_DECAY}")
print(f"  use_nade:       {use_nade}")
print(f"  use_jk:         {use_jk}")
print(f"  use_rotograd:   {use_rotograd}")

# CRITICAL FIX: Detect if checkpoint uses PostChordPrediction or ChordPrediction
checkpoint_keys = list(state_dict.keys())
has_frozen_model = any(k.startswith("frozen_model.") for k in checkpoint_keys)

print(f"\nCheckpoint architecture detection:")
if has_frozen_model:
    print(f"  ✓ Detected PostChordPrediction (frozen_model.* keys)")
    print(f"  → Building PostChordPrediction to match")

    # Build a frozen_model (encoder)
    from chordgnn.models.chord import ChordPredictionModel
    frozen_model = ChordPredictionModel(in_feats=in_feats)

    # Build PostChordPrediction
    model = st.models.chord.PostChordPrediction(
        in_feats=in_feats,
        n_hidden=n_hidden,
        tasks=tasks,  # CRITICAL: Use pretrained tasks!
        n_layers=n_layers,
        dropout=dropout,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        use_nade=use_nade,
        use_jk=use_jk,
        use_rotograd=use_rotograd,
        frozen_model=frozen_model,
        device="cpu",  # Will be moved to GPU by Trainer if available
    )
else:
    print(f"  ✓ Detected ChordPrediction (encoder.* or module.* keys)")
    print(f"  → Building ChordPrediction to match")

    model = st.models.chord.ChordPrediction(
        in_feats=in_feats,
        n_hidden=n_hidden,
        tasks=tasks,  # CRITICAL: Use pretrained tasks!
        n_layers=n_layers,
        dropout=dropout,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        use_nade=use_nade,
        use_jk=use_jk,
        use_rotograd=use_rotograd,
        device="cpu",  # Will be moved to GPU by Trainer if available
    )

print()


# -------------------------
# Step 7: Load pretrained weights
# -------------------------
print("=" * 70)
print("STEP 7: Loading pretrained weights")
print("=" * 70 + "\n")

print("Attempting to load all pretrained weights...")

# Clean state dict - remove loss parameters
cleaned_state_dict = {}
for k, v in state_dict.items():
    # Skip loss parameters
    if k.startswith("train_loss.") or k.startswith("val_loss.") or k.startswith("test_loss."):
        continue

    # For ChordPrediction (not PostChordPrediction), strip 'module.' prefix if present
    if not has_frozen_model and k.startswith("module."):
        cleaned_state_dict[k[7:]] = v
    else:
        # For PostChordPrediction, keep keys as-is (frozen_model.*)
        cleaned_state_dict[k] = v

# Load with strict=False to allow missing keys (like optimizer state)
missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)

print(f"\n✓ Pretrained weights loaded")
print(f"  Loaded tensors:     {len(cleaned_state_dict)}")
print(f"  Missing keys:       {len(missing)}")
print(f"  Unexpected keys:    {len(unexpected)}")

# Verify weights actually loaded correctly
if len(missing) > len(cleaned_state_dict) * 0.5:
    print("\n⚠ WARNING: More than 50% of model keys are missing!")
    print("  This suggests the checkpoint architecture doesn't match the model.")
    print("  First 10 missing keys:")
    for k in missing[:10]:
        print(f"    {k}")
    print("\n  First 10 checkpoint keys:")
    for k in list(cleaned_state_dict.keys())[:10]:
        print(f"    {k}")
    raise RuntimeError("Weight loading failed - architecture mismatch!")

# Check if encoder loaded
encoder_keys_in_ckpt = [k for k in cleaned_state_dict.keys() if "encoder" in k]
encoder_keys_in_model = [k for k in model.state_dict().keys() if "encoder" in k]
encoder_loaded = len([k for k in encoder_keys_in_model if k not in missing]) > 0

if encoder_loaded:
    print(f"\n✓ Encoder loaded successfully ({len(encoder_keys_in_ckpt)} encoder keys)")
else:
    print(f"\n✗ ERROR: Encoder did NOT load!")
    raise RuntimeError("Encoder failed to load - critical error!")

# Check if important task heads loaded
important_heads = ["romanNumeral", "localkey", "tonkey", "pcset", "bass"]
loaded_heads = []
missing_heads = []

for head in important_heads:
    # Check if any model keys for this head are NOT in missing list
    head_keys_in_model = [k for k in model.state_dict().keys() if head in k]
    head_keys_loaded = [k for k in head_keys_in_model if k not in missing]

    if head_keys_loaded:
        loaded_heads.append(head)
    else:
        missing_heads.append(head)

if loaded_heads:
    print(f"✓ Task heads loaded: {', '.join(loaded_heads)}")
if missing_heads:
    print(f"⚠ Task heads NOT loaded (random init): {', '.join(missing_heads)}")
    print(f"  → This is OK if you're adding new tasks, but check if unexpected!")

print()


# -------------------------
# Step 8: Log config to W&B
# -------------------------
wandb_logger.experiment.config.update({
    "dataset": "Mozart",
    "finetune": True,
    "pretrained_ckpt": PRETRAINED_CKPT,
    "pretrained_vocab_size": rn_vocab_size,
    "data_version": DATA_VERSION,
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
    "earlystop_patience": 10,
    "earlystop_min_delta": 1e-4,
    "nan_sanitize": "all_floats_train_val_test",
})


# -------------------------
# Step 9: NaN/Inf sanitize callback
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
# Step 10: Train with EarlyStopping + checkpointing
# -------------------------
checkpoint = ModelCheckpoint(
    save_top_k=1,
    monitor="val_loss",
    mode="min",
    filename="mozart-finetune-FIXED-{epoch:02d}-{val_loss:.3f}",
)

# FIXED: Increased patience from 5 to 10 for small dataset
early_stop = EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=10,    # Increased from 5
    min_delta=1e-4,
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
print("STEP 11: Starting finetuning on Mozart data")
print("=" * 70 + "\n")

trainer.fit(model, datamodule)

print("\nBEST CKPT:", checkpoint.best_model_path)
print("BEST SCORE:", checkpoint.best_model_score)

# Log the best ckpt as a W&B artifact
if checkpoint.best_model_path and os.path.exists(checkpoint.best_model_path):
    artifact = wandb.Artifact(
        name="mozart-finetuned-model-FIXED",
        type="model",
        description="ChordGNN finetuned on Mozart (FIXED: vocab matched, lower LR)"
    )
    artifact.add_file(checkpoint.best_model_path)
    wandb_logger.experiment.log_artifact(artifact)
    print("✓ Logged W&B artifact: mozart-finetuned-model-FIXED\n")
else:
    print("WARNING: No best_model_path found; skipping artifact logging.\n")


# -------------------------
# Step 12: Test best checkpoint
# -------------------------
print("\n" + "=" * 70)
print("STEP 12: Testing best checkpoint")
print("=" * 70 + "\n")

trainer.test(model, datamodule, ckpt_path=checkpoint.best_model_path)

print("\n" + "=" * 70)
print("✓ COMPLETE!")
print(f"Finetuned model: {checkpoint.best_model_path}")
print("=" * 70 + "\n")

wandb.finish()
