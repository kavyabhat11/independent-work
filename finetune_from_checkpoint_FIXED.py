#!/usr/bin/env python3
"""
Finetune pretrained ChordGNN on Mozart data (robust + checkpoint-truth)

This version fixes the two big silent-footgun issues:
1) ALWAYS loads the correct W&B artifact checkpoint (model.ckpt). No globbing.
2) Uses the checkpoint-truth 14 tasks + fixed label column order, not available_representations.
   (Your dataset returns labels like (T, 15, 1); we squeeze + take first 14 cols.)

It also:
- Detects DATA_VERSION from ckpt romanNumeral head (31 -> v2.0.0, 76 -> v1.0.0)
- Forces n_hidden=256 if ckpt heads imply 256 (yours do)
- Splits train/val/test by matching graph.name to MOZART_ROOT split filenames
- Freezes frozen_model by default, then unfreezes ONLY last 1 GCN layer + GRU
  (model has only 2 total GCN layers; unfreezing both = full fine-tuning)
  (set UNFREEZE_LAST_N_GCN_LAYERS=0 and UNFREEZE_GRU=False to freeze all encoder layers)

Run:
  export MOZART_ROOT=/path/to/mozart_dataset   # contains training/validation/test/*.tsv
  wandb login                                  # if needed
  python finetune_mozart_ckpttruth.py
"""

import os
import glob
import shutil
import math
from collections import defaultdict

import torch
import chordgnn as st

import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback


# =============================================================================
# CONFIG
# =============================================================================
WANDB_ARTIFACT = "melkisedeath/chord_rec/model-kvd0jic5:v0"
ARTIFACT_ROOT = "./artifacts"

MOZART_ROOT = os.environ.get("MOZART_ROOT", "./mozart_dataset")
CACHE_ROOT = os.environ.get("CACHE_ROOT", "/scratch/network/kb9520/chordgnn_data")

LR = float(os.environ.get("LR", "5e-5"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "1e-4"))
MAX_EPOCHS = int(os.environ.get("MAX_EPOCHS", "80"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "8"))

# IMPORTANT: checkpoint has 14 heads
NUM_TASKS = 14

WANDB_PROJECT = "chord_rec"
WANDB_RUN_NAME = os.environ.get("WANDB_RUN_NAME", "mozart-finetune-CKPTTRUTH")

# Exact checkpoint head order you used in baseline
TASK_ORDER = [
    "localkey", "tonkey", "degree1", "degree2", "quality", "inversion",
    "root", "romanNumeral", "hrhythm", "pcset", "bass", "tenor", "alto", "soprano"
]

# How many GCN layers to unfreeze from the end (0 = fully frozen, 1 = last GCN layer only)
# Model has only 2 total GCN layers, so 2 would unfreeze everything
UNFREEZE_LAST_N_GCN_LAYERS = int(os.environ.get("UNFREEZE_LAST_N_GCN_LAYERS", "1"))
# Also unfreeze GRU and final projection layers
UNFREEZE_GRU = os.environ.get("UNFREEZE_GRU", "True").lower() == "true"

torch.manual_seed(0)


# =============================================================================
# PATCH: old torch can't handle nn.Linear(device=None,dtype=None)
# (only needed if your env is old; harmless otherwise)
# =============================================================================
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter

class PatchedLinear(nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Parameter

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)

        factory_kwargs = {}
        if device is not None:
            factory_kwargs["device"] = device
        if dtype is not None:
            factory_kwargs["dtype"] = dtype

        self.weight = Parameter(torch.empty((self.out_features, self.in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(self.out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

nn.Linear = PatchedLinear


# =============================================================================
# HELPERS
# =============================================================================
def banner(msg: str):
    print("\n" + "=" * 78)
    print(msg)
    print("=" * 78)

def download_ckpt_wandb() -> str:
    """
    Always end up with ./artifacts/model.ckpt that is the artifact's file.
    No globbing, no accidental picking up local finetune ckpts.
    """
    expected = os.path.join(ARTIFACT_ROOT, "model.ckpt")
    if os.path.exists(expected):
        print(f"✓ Found local checkpoint: {expected}")
        return expected

    print(f"Downloading from W&B: {WANDB_ARTIFACT}")
    api = wandb.Api()
    artifact = api.artifact(WANDB_ARTIFACT, type="model")
    downloaded_dir = artifact.download(root=ARTIFACT_ROOT)

    ckpt_path = os.path.join(downloaded_dir, "model.ckpt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Expected model.ckpt at {ckpt_path} (downloaded_dir={downloaded_dir})")

    # copy to canonical path for stability
    os.makedirs(ARTIFACT_ROOT, exist_ok=True)
    if ckpt_path != expected:
        shutil.copy(ckpt_path, expected)
        ckpt_path = expected

    print(f"✓ Downloaded to: {ckpt_path}")
    return ckpt_path

def infer_task_dims_from_ckpt(state_dict: dict) -> dict:
    """
    Uses the final layer weights: frozen_model.classifier.classifier.<task>.layers.1.weight
    shape = (out_dim, hidden_dim). We only need out_dim.
    """
    dims = {}
    for t in TASK_ORDER:
        key = f"frozen_model.classifier.classifier.{t}.layers.1.weight"
        if key in state_dict:
            dims[t] = int(state_dict[key].shape[0])
        else:
            # fallback: try without frozen_model prefix
            key2 = f"classifier.classifier.{t}.layers.1.weight"
            if key2 in state_dict:
                dims[t] = int(state_dict[key2].shape[0])

    missing = [t for t in TASK_ORDER if t not in dims]
    if missing:
        raise RuntimeError(f"Could not infer dims for tasks from ckpt: {missing}")
    return dims

def infer_head_hidden_from_ckpt(state_dict: dict) -> int:
    """
    Head hidden size from layers.0.weight: (hidden_dim, hidden_dim) usually (256,256)
    """
    key = "frozen_model.classifier.classifier.romanNumeral.layers.0.weight"
    if key in state_dict:
        return int(state_dict[key].shape[0])
    # fallback
    for k, v in state_dict.items():
        if "romanNumeral.layers.0.weight" in k and hasattr(v, "shape"):
            return int(v.shape[0])
    return 256

def strip_state_dict_for_loading(state_dict: dict, keep_frozen_model_prefix: bool) -> dict:
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith(("train_loss.", "val_loss.", "test_loss.")):
            continue
        if keep_frozen_model_prefix:
            cleaned[k] = v
        else:
            # strip "frozen_model." if you're loading into bare ChordPredictionModel
            nk = k
            if nk.startswith("frozen_model."):
                nk = nk[len("frozen_model."):]
            if nk.startswith("module."):
                nk = nk[len("module."):]
            cleaned[nk] = v
    return cleaned

def copy_mozart_splits_into_cache(data_version: str):
    """
    Writes Mozart TSVs into the dataset cache directory expected by chordgnn datasets.
    """
    if data_version == "v1.0.0":
        dataset_dir = os.path.join(CACHE_ROOT, "AugmentedNetChordDataset", "dataset")
    else:
        dataset_dir = os.path.join(CACHE_ROOT, "AugmentedNetLatestChordDataset", "dataset")

    if os.path.exists(dataset_dir):
        print(f"Cleaning existing dataset cache: {dataset_dir}")
        shutil.rmtree(dataset_dir)

    os.makedirs(os.path.join(dataset_dir, "training"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "validation"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "test"), exist_ok=True)

    for split in ("training", "validation", "test"):
        for tsv in glob.glob(os.path.join(MOZART_ROOT, split, "*.tsv")):
            shutil.copy(tsv, os.path.join(dataset_dir, split))

    train_ct = len(glob.glob(os.path.join(dataset_dir, "training", "*.tsv")))
    val_ct = len(glob.glob(os.path.join(dataset_dir, "validation", "*.tsv")))
    test_ct = len(glob.glob(os.path.join(dataset_dir, "test", "*.tsv")))
    print(f"✓ Data ready: {train_ct} train, {val_ct} val, {test_ct} test")
    return dataset_dir, train_ct, val_ct, test_ct


# =============================================================================
# DATA MODULE
# =============================================================================
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Subset

class MozartDatamodule(LightningDataModule):
    def __init__(self, dataset, dataset_dir, num_workers: int):
        super().__init__()
        self.dataset = dataset
        self.dataset_dir = dataset_dir
        self.num_workers = num_workers

    def collate_fn(self, batch):
        """
        Returns a single-graph batch compatible with chordgnn Lightning models:
          (x, edges, edge_type, labels_dict, onset_div, lengths)
        """
        from chordgnn.utils.hgraph import add_reverse_edges_from_edge_index

        x, edges, edge_type, labels, onset_div, name = batch[0]

        # squeeze batch dims
        x = x.squeeze(0).float()
        edges = edges.squeeze(0)
        edge_type = edge_type.squeeze(0)
        onset_div = onset_div.squeeze()

        # labels: often (T, 15, 1) or (T, 15); we want (T, 14) for TASK_ORDER
        labels = labels.squeeze(0) if labels.ndim == 4 else labels  # just in case
        if labels.ndim == 3 and labels.shape[-1] == 1:
            labels = labels.squeeze(-1)  # (T, num_cols)
        if labels.ndim != 2 or labels.shape[1] < len(TASK_ORDER):
            raise RuntimeError(f"Unexpected labels shape in collate_fn: {tuple(labels.shape)} name={name}")

        label_mat = labels[:, :len(TASK_ORDER)]
        labels_dict = {task: label_mat[:, i].long() for i, task in enumerate(TASK_ORDER)}

        # add reverse edges
        edges, edge_type = add_reverse_edges_from_edge_index(edges, edge_type)

        lengths = torch.tensor([label_mat.shape[0]]).long()
        return x, edges, edge_type, labels_dict, onset_div, lengths

    def setup(self, stage=None):
        # build file whitelists from the split folders we created
        train_files = set(os.path.splitext(os.path.basename(f))[0]
                          for f in glob.glob(os.path.join(self.dataset_dir, "training", "*.tsv")))
        val_files = set(os.path.splitext(os.path.basename(f))[0]
                        for f in glob.glob(os.path.join(self.dataset_dir, "validation", "*.tsv")))
        test_files = set(os.path.splitext(os.path.basename(f))[0]
                         for f in glob.glob(os.path.join(self.dataset_dir, "test", "*.tsv")))

        train_idx, val_idx, test_idx = [], [], []

        # graph.name sometimes has augmentation suffix "-1", "-2" etc.
        for i, g in enumerate(self.dataset.graphs):
            base = g.name
            parts = base.rsplit("-", 1)
            if len(parts) == 2 and parts[1].isdigit():
                maybe = parts[0]
                if maybe in train_files or maybe in val_files or maybe in test_files:
                    base = maybe

            if base in train_files:
                train_idx.append(i)
            elif base in val_files:
                val_idx.append(i)
            elif base in test_files:
                test_idx.append(i)

        if not train_idx or not val_idx or not test_idx:
            print("Split matching failed.")
            print("Example graph names:", [g.name for g in self.dataset.graphs[:10]])
            print("Example train files:", list(train_files)[:10])
            raise RuntimeError("Failed to match graphs to train/val/test.")

        self.dataset_train = Subset(self.dataset, train_idx)
        self.dataset_val = Subset(self.dataset, val_idx)
        self.dataset_test = Subset(self.dataset, test_idx)

        print(f"✓ Matched graphs: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=1, shuffle=True,
                          num_workers=self.num_workers, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=1, shuffle=False,
                          num_workers=self.num_workers, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=1, shuffle=False,
                          num_workers=self.num_workers, collate_fn=self.collate_fn)


# =============================================================================
# CALLBACK: sanitize NaNs
# =============================================================================
class SanitizeBatchNaNs(Callback):
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
                        print(f"[SanitizeBatchNaNs] Fixed {int(bad.sum().item())} NaN/Inf in {path} shape={tuple(obj.shape)}")
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


# =============================================================================
# MAIN
# =============================================================================
def main():
    banner("Mozart Finetuning (checkpoint-truth)")

    # ---- Step 1: ckpt
    banner("STEP 1: Load pretrained checkpoint (correct)")
    pretrained_ckpt = download_ckpt_wandb()
    PRETRAINED_CKPT = pretrained_ckpt
    # ---- Step 2: inspect ckpt
    banner("STEP 2: Inspect checkpoint (task dims + version)")
    ckpt = torch.load(pretrained_ckpt, map_location="cpu")
    import hashlib

    def sha256_file(path, chunk=1<<20):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                b = f.read(chunk)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()

    print("\n[CHECKPOINT VERIFY]")
    print("PRETRAINED_CKPT:", PRETRAINED_CKPT)
    print("exists:", os.path.exists(PRETRAINED_CKPT), "size_bytes:", os.path.getsize(PRETRAINED_CKPT))
    print("sha256:", sha256_file(PRETRAINED_CKPT)[:16], "(prefix)")
    print("ckpt keys:", list(ckpt.keys())[:8])

    sd = ckpt.get("state_dict", {})
    print("state_dict tensors:", len(sd))

    # confirm it's the model artifact, not a lightning finetune ckpt you created
    has_frozen = any(k.startswith("frozen_model.") for k in sd.keys())
    has_artifact_heads = any("classifier.classifier.romanNumeral.layers.1.weight" in k for k in sd.keys())
    print("has frozen_model.*:", has_frozen)
    print("has romanNumeral head key:", has_artifact_heads)

    # print RN head shape (should be (31,256))
    rn_key = "frozen_model.classifier.classifier.romanNumeral.layers.1.weight"
    if rn_key in sd:
        print("RN head weight shape:", tuple(sd[rn_key].shape))
    else:
        # fallback scan
        for k,v in sd.items():
            if "romanNumeral.layers.1.weight" in k:
                print("RN head weight shape (found):", k, tuple(v.shape))
                break

    state_dict = ckpt.get("state_dict", {})
    if not state_dict:
        raise RuntimeError("Checkpoint missing state_dict")

    task_dims = infer_task_dims_from_ckpt(state_dict)
    rn_vocab = task_dims["romanNumeral"]
    head_hidden = infer_head_hidden_from_ckpt(state_dict)

    if rn_vocab == 31:
        data_version = "v2.0.0"
    elif rn_vocab == 76:
        data_version = "v1.0.0"
    else:
        raise RuntimeError(f"Unexpected romanNumeral vocab in ckpt: {rn_vocab}")

    print("✓ romanNumeral vocab:", rn_vocab)
    print("✓ inferred head hidden:", head_hidden)
    print("✓ DATA_VERSION:", data_version)
    print("✓ task dims:", task_dims)

    # ---- Step 3: cache mozart splits
    banner("STEP 3: Write Mozart TSV splits into dataset cache")
    dataset_dir, train_ct, val_ct, test_ct = copy_mozart_splits_into_cache(data_version)

    # ---- Step 4: build dataset
    banner("STEP 4: Build chordgnn dataset (raw_dir=CACHE_ROOT)")
    if data_version == "v1.0.0":
        dataset = st.data.datasets.chord.AugmentedNetChordGraphDataset(
            raw_dir=CACHE_ROOT,
            force_reload=True,
            nprocs=max(1, NUM_WORKERS),
            include_synth=False,
            num_tasks=NUM_TASKS,
            collection="all",
        )
    else:
        dataset = st.data.datasets.chord.Augmented2022ChordGraphDataset(
            raw_dir=CACHE_ROOT,
            force_reload=True,
            nprocs=NUM_WORKERS,
            include_synth=False,
            num_tasks=NUM_TASKS,
            collection="all",
        )
    print("✓ dataset graphs:", len(dataset.graphs))

    # ---- Step 5: datamodule
    banner("STEP 5: Datamodule (split by filename)")
    datamodule = MozartDatamodule(dataset=dataset, dataset_dir=dataset_dir, num_workers=NUM_WORKERS)
    datamodule.setup()

    # ---- Step 6: logger
    wandb_logger = WandbLogger(project=WANDB_PROJECT, name=WANDB_RUN_NAME, log_model=True)

    # ---- Step 7: Build Lightning model (PostChordPrediction, because ckpt has frozen_model.*)
    banner("STEP 6: Build Lightning model to match ckpt (PostChordPrediction)")

    # in_feats from a batch (robust)
    b0 = next(iter(datamodule.train_dataloader()))
    in_feats = int(b0[0].shape[-1])
    print("✓ in_feats:", in_feats)

    n_hidden = head_hidden  # 256 for your ckpt
    n_layers = 1            # safe default; ckpt may have more but heads show 256 and your baseline used 1
    dropout = 0.44          # you saw this in baseline hparams sometimes; safe to keep
    use_nade = False
    use_jk = False
    use_rotograd = False

    # Build a frozen encoder model
    from chordgnn.models.chord import ChordPredictionModel
    frozen_model = ChordPredictionModel(in_feats=in_feats)

    model = st.models.chord.PostChordPrediction(
        in_feats=in_feats,
        n_hidden=n_hidden,
        tasks=task_dims,     # IMPORTANT: ints, checkpoint-truth
        n_layers=n_layers,
        dropout=dropout,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        use_nade=use_nade,
        use_jk=use_jk,
        use_rotograd=use_rotograd,
        frozen_model=frozen_model,
        device="cpu",
    )

    # ---- Step 8: load weights (keep frozen_model.* prefix)
    banner("STEP 7: Load pretrained weights into Lightning model")
    cleaned = strip_state_dict_for_loading(state_dict, keep_frozen_model_prefix=True)
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print("✓ loaded tensors:", len(cleaned))
    print("  missing:", len(missing))
    print("  unexpected:", len(unexpected))
    if missing:
        print("  missing sample:", missing[:15])
    if unexpected:
        print("  unexpected sample:", unexpected[:15])

    # ---- Step 9: freeze frozen_model by default, then selectively unfreeze
    banner("STEP 8: Freeze encoder (then unfreeze last layers)")

    # First, freeze everything
    for p in model.frozen_model.parameters():
        p.requires_grad = False
    print("✓ frozen_model fully frozen")

    unfrozen_parts = []

    # Unfreeze last N GCN layers if requested
    if UNFREEZE_LAST_N_GCN_LAYERS > 0:
        # ChordEncoder has: encoder (HGCN with self.layers ModuleList)
        if hasattr(model.frozen_model, 'encoder') and hasattr(model.frozen_model.encoder, 'encoder'):
            gcn = model.frozen_model.encoder.encoder  # HGCN instance
            if hasattr(gcn, 'layers'):
                n_total_layers = len(gcn.layers)
                n_to_unfreeze = min(UNFREEZE_LAST_N_GCN_LAYERS, n_total_layers)

                for layer in gcn.layers[-n_to_unfreeze:]:
                    for p in layer.parameters():
                        p.requires_grad = True

                unfrozen_parts.append(f"last {n_to_unfreeze}/{n_total_layers} GCN layers")
                print(f"✓ Unfroze last {n_to_unfreeze} GCN layers (out of {n_total_layers} total)")
            else:
                print("⚠ Could not find GCN layers to unfreeze")
        else:
            print("⚠ Could not find encoder to unfreeze GCN layers")

    # Unfreeze GRU layer only (not projection layers)
    if UNFREEZE_GRU:
        encoder = model.frozen_model.encoder

        # Unfreeze GRU only
        if hasattr(encoder, 'gru'):
            for p in encoder.gru.parameters():
                p.requires_grad = True
            unfrozen_parts.append("GRU")
            print(f"✓ Unfroze GRU layer")

    if not unfrozen_parts:
        print("✓ All encoder layers remain frozen (only training task heads)")
    else:
        print(f"✓ Unfrozen: {' + '.join(unfrozen_parts)}")

    # Print trainable parameter count
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Trainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")

    # ---- Step 10: callbacks + trainer
    banner("STEP 9: Train")
    ckpt_cb = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        filename="mozart-finetune-CKPTTRUTH-{epoch:02d}-{val_loss:.3f}",
    )
    es_cb = EarlyStopping(monitor="val_loss", mode="min", patience=10, min_delta=1e-4, verbose=True)

    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        devices=[0] if torch.cuda.is_available() else None,
        callbacks=[ckpt_cb, es_cb, SanitizeBatchNaNs(verbose_first_k=5)],
        gradient_clip_val=1.0,
        precision=32,
        logger=wandb_logger,
    )

    # log useful config
    wandb_logger.experiment.config.update({
        "mozart_root": MOZART_ROOT,
        "dataset_dir": dataset_dir,
        "train_ct_files": train_ct,
        "val_ct_files": val_ct,
        "test_ct_files": test_ct,
        "pretrained_ckpt": pretrained_ckpt,
        "data_version": data_version,
        "task_dims": task_dims,
        "task_order": TASK_ORDER,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "max_epochs": MAX_EPOCHS,
        "num_workers": NUM_WORKERS,
        "unfreeze_last_n_gcn_layers": UNFREEZE_LAST_N_GCN_LAYERS,
        "unfreeze_gru": UNFREEZE_GRU,
    })

    print("\n[PRE-FIT VALIDATE] (baseline under Lightning metrics)")
    trainer.validate(model, datamodule=datamodule, verbose=True)

    trainer.fit(model, datamodule)

    print("\nBEST CKPT:", ckpt_cb.best_model_path)
    print("BEST SCORE:", ckpt_cb.best_model_score)

    banner("STEP 10: Test best ckpt")
    trainer.test(model, datamodule, ckpt_path=ckpt_cb.best_model_path)

    wandb.finish()
    banner("DONE")


if __name__ == "__main__":
    main()
