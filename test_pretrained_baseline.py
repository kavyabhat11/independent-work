#!/usr/bin/env python3
"""
Test pretrained ChordGNN checkpoint on Mozart TEST data using PyTorch Lightning.

This script:
- Loads pretrained checkpoint: melkisedeath/chord_rec/model-kvd0jic5:v0
- Tests ONLY on ./mozart_dataset/test/*.tsv (NOT training or validation)
- Uses PyTorch Lightning Trainer.test() to compute all metrics
- Automatically computes RNalt, regular RomNum, and logs to wandb
- Does NOT retrain - only loads checkpoint and tests

Usage:
  export MOZART_ROOT=./mozart_dataset     # must contain test/*.tsv
  python test_pretrained_baseline.py
"""

import os
import glob
import shutil
import math
from collections import defaultdict
from typing import Optional

import torch
import chordgnn as st
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

# ----------------------------
# CONFIG
# ----------------------------
# If LOCAL_CKPT is set, use it directly instead of downloading from wandb
LOCAL_CKPT = os.environ.get("LOCAL_CKPT", "")
WANDB_ARTIFACT = os.environ.get("WANDB_ARTIFACT", "melkisedeath/chord_rec/model-kvd0jic5:v0")
ARTIFACT_ROOT = os.environ.get("ARTIFACT_ROOT", "./artifacts")

# Set to "post" for finetuned models (PostChordPrediction), "base" for pretrained (ChordPrediction)
MODEL_TYPE = os.environ.get("MODEL_TYPE", "base")

MOZART_ROOT = os.environ.get("MOZART_ROOT", "./mozart_dataset")
CACHE_ROOT = os.environ.get("CACHE_ROOT", "/scratch/network/kb9520/chordgnn_data")
DATA_VERSION = os.environ.get("DATA_VERSION", "v2.0.0")

NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "8"))


# ----------------------------
# PATCH: old torch can't handle nn.Linear(device=None,dtype=None)
# and your environment appears to have that issue.
# ----------------------------
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
        # Some call sites in your repo may pass weird types; fail loudly with context
        try:
            self.in_features = int(in_features)
        except Exception:
            raise TypeError("PatchedLinear: in_features is not int-like: {} ({})".format(in_features, type(in_features)))
        try:
            self.out_features = int(out_features)
        except Exception:
            raise TypeError("PatchedLinear: out_features is not int-like: {} ({})".format(out_features, type(out_features)))

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

# monkeypatch
nn.Linear = PatchedLinear


# ----------------------------
# HELPERS
# ----------------------------
def banner(msg: str):
    print("\n" + "=" * 78)
    print(msg)
    print("=" * 78)


def download_wandb_ckpt(artifact_path: str, root: str) -> str:
    import wandb
    a = wandb.Api().artifact(artifact_path, type="model")
    d = a.download(root=root)
    ckpt = os.path.join(d, "model.ckpt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError("Expected model.ckpt at {}, but not found.".format(ckpt))
    return ckpt


def collate_fn_test(batch):
    """
    Collate function for test data that converts labels to dict format expected by Lightning.
    Compatible with ChordPrediction.test_step().
    """
    from chordgnn.utils.hgraph import add_reverse_edges_from_edge_index

    # Dataset returns: (x, edges, edge_type, labels, onset_div, name)
    x, edges, edge_type, labels, onset_div, name = batch[0]

    # Squeeze batch dimensions
    x = x.squeeze(0).float()
    edges = edges.squeeze(0)
    edge_type = edge_type.squeeze(0)
    onset_div = onset_div.squeeze()

    # Task order matching checkpoint (14 tasks)
    TASK_ORDER = [
        "localkey", "tonkey", "degree1", "degree2", "quality", "inversion",
        "root", "romanNumeral", "hrhythm", "pcset", "bass", "tenor", "alto", "soprano"
    ]

    # Convert labels tensor to dict
    labels = labels.squeeze(0) if labels.ndim == 4 else labels
    if labels.ndim == 3 and labels.shape[-1] == 1:
        labels = labels.squeeze(-1)
    if labels.ndim != 2 or labels.shape[1] < 15:
        raise RuntimeError("Unexpected labels shape: {} for {}".format(tuple(labels.shape), name))

    # First 14 columns are the tasks
    label_mat = labels[:, :len(TASK_ORDER)]
    labels_dict = {task: label_mat[:, i].long() for i, task in enumerate(TASK_ORDER)}

    # 15th column is onset - add it if it exists
    if labels.shape[1] > 14:
        labels_dict["onset"] = labels[:, 14].long()

    # Add reverse edges
    edges, edge_type = add_reverse_edges_from_edge_index(edges, edge_type)

    # Return lengths (to match finetuning script collate_fn)
    lengths = torch.tensor([label_mat.shape[0]]).long()
    return x, edges, edge_type, labels_dict, onset_div, lengths


def copy_all_test_tsvs_into_cache(mozart_root: str, cache_root: str, data_version: str) -> int:
    """Copy TEST TSVs (not validation) into the cache directory."""
    if data_version == "v2.0.0":
        base = os.path.join(cache_root, "AugmentedNetLatestChordDataset", "dataset")
    else:
        base = os.path.join(cache_root, "AugmentedNetChordDataset", "dataset")

    test_dir = os.path.join(base, "test")

    if os.path.exists(base):
        print("Cleaning existing dataset cache at: {}".format(base))
        shutil.rmtree(base)
    os.makedirs(test_dir, exist_ok=True)

    candidates = sorted(glob.glob(os.path.join(mozart_root, "test", "*.tsv")))
    if not candidates:
        raise FileNotFoundError("No TSVs found under {}".format(os.path.join(mozart_root, "test", "*.tsv")))

    for f in candidates:
        shutil.copy(f, os.path.join(test_dir, os.path.basename(f)))

    print("✓ Copied {} test TSVs into cache:\n  {}".format(len(candidates), test_dir))
    return len(candidates)


# ----------------------------
# MAIN
# ----------------------------
def main():
    banner("PRETRAINED CHORDGNN BASELINE ON MOZART TEST DATA")

    print("MOZART_ROOT    = {}".format(MOZART_ROOT))
    print("CACHE_ROOT     = {}".format(CACHE_ROOT))
    print("DATA_VERSION   = {}".format(DATA_VERSION))
    print("WANDB_ARTIFACT = {}".format(WANDB_ARTIFACT))

    banner("Step 1: Get checkpoint path")
    if LOCAL_CKPT:
        ckpt_path = LOCAL_CKPT
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError("LOCAL_CKPT specified but not found: {}".format(ckpt_path))
        print("✓ Using local checkpoint: {}".format(ckpt_path))
    else:
        ckpt_path = download_wandb_ckpt(WANDB_ARTIFACT, ARTIFACT_ROOT)
        print("✓ Downloaded checkpoint: {}".format(ckpt_path))

    banner("Step 2: Copy TEST TSVs into cache")
    copy_all_test_tsvs_into_cache(MOZART_ROOT, CACHE_ROOT, DATA_VERSION)

    banner("Step 3: Build test dataset")
    if DATA_VERSION == "v1.0.0":
        DatasetCls = st.data.datasets.chord.AugmentedNetChordGraphDataset
    else:
        DatasetCls = st.data.datasets.chord.Augmented2022ChordGraphDataset

    # Try test split, fallback to all
    try:
        test_dataset = DatasetCls(
            raw_dir=CACHE_ROOT,
            force_reload=True,
            nprocs=max(1, NUM_WORKERS),
            include_synth=False,
            num_tasks=11,
            collection="test",
        )
        print("✓ Test dataset built with collection='test'")
    except Exception as e:
        print("collection='test' failed, trying collection='all': {}".format(e))
        test_dataset = DatasetCls(
            raw_dir=CACHE_ROOT,
            force_reload=True,
            nprocs=max(1, NUM_WORKERS),
            include_synth=False,
            num_tasks=11,
            collection="all",
        )
        print("✓ Test dataset built with collection='all'")

    print("✓ Test dataset length: {}".format(len(test_dataset)))

    banner("Step 4: Create test DataLoader")
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        collate_fn=collate_fn_test
    )

    banner("Step 5: Load model from checkpoint")
    # Choose model class based on MODEL_TYPE
    if MODEL_TYPE == "post":
        ModelClass = st.models.chord.PostChordPrediction
        print("Using PostChordPrediction (finetuned model)")
    else:
        ModelClass = st.models.chord.ChordPrediction
        print("Using ChordPrediction (pretrained model)")

    model = ModelClass.load_from_checkpoint(
        ckpt_path,
        strict=False,
        map_location="cpu"
    )
    print("✓ Model loaded from checkpoint")

    banner("Step 6: Run test with Lightning Trainer")
    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False,  # Disable logging for simple test
    )

    # This will automatically compute all metrics including RNalt
    trainer.test(model, dataloaders=test_loader)

    banner("Testing complete! Check output above for RNalt and other metrics.")


if __name__ == "__main__":
    main()
