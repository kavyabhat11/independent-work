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
import numpy as np
import math
from collections import defaultdict
from typing import Optional

import torch
import chordgnn as st
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

# Import cosine similarity resolution functions
COSINE_AVAILABLE = False
resolveRomanNumeralCosine = None
RN_LATEST = None
KEYS_LATEST = None
resolveRomanNumeralCosine_v1 = None
RN_V1 = None
KEYS_V1 = None

try:
    # Import cosine resolution function (only exists in v1 chord_representations.py)
    # This is what analyse_score.py uses
    from chordgnn.utils.chord_representations import (
        resolveRomanNumeralCosine,
        COMMON_ROMAN_NUMERALS as RN_V1,
        KEYS as KEYS_V1
    )

    # Import vocabulary from latest version as well
    from chordgnn.utils.chord_representations_latest import (
        COMMON_ROMAN_NUMERALS as RN_LATEST,
        KEYS as KEYS_LATEST
    )

    # Use v1 for both since resolveRomanNumeralCosine only exists there
    resolveRomanNumeralCosine_v1 = resolveRomanNumeralCosine

    COSINE_AVAILABLE = True
    print("✓ Cosine resolution functions loaded successfully")
except ImportError as e:
    print(f"⚠ Warning: Could not import cosine functions (ImportError): {e}")
    print("  This is expected if music21 or other dependencies are missing.")
    print("  Cosine resolution will be skipped, but direct RNalt evaluation will still work.")
    COSINE_AVAILABLE = False
except Exception as e:
    print(f"⚠ Warning: Could not import cosine functions (unexpected error): {e}")
    COSINE_AVAILABLE = False

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


def masked_accuracy(pred_logits: torch.Tensor, target: torch.Tensor):
    """Compute accuracy masking out invalid labels."""
    pred_classes = pred_logits.argmax(dim=-1)
    mask = target >= 0
    total = int(mask.sum().item())
    if total == 0:
        return 0, 0
    correct = int((pred_classes[mask] == target[mask]).sum().item())
    return correct, total


def align_target_to_pred_length(target: torch.Tensor, pred_len: int, onset_idx: Optional[torch.Tensor]):
    """Align targets to prediction length."""
    if target.ndim != 1:
        target = target.view(-1)

    if target.shape[0] == pred_len:
        return target

    if onset_idx is not None:
        onset_idx_flat = onset_idx.view(-1).long()
        if onset_idx_flat.shape[0] == pred_len:
            max_i = int(onset_idx_flat.max().item()) if onset_idx_flat.numel() > 0 else -1
            if target.shape[0] >= max_i + 1:
                return target[onset_idx_flat]

    if target.shape[0] > pred_len:
        return target[:pred_len]

    pad = torch.full((pred_len - target.shape[0],), -1, dtype=target.dtype, device=target.device)
    return torch.cat([target, pad], dim=0)


def acc_compute_time_step(acc_onset_level, onset_times):
    """
    Convert onset-level accuracy to time-step accuracy.
    Divides time into 0.125 (1/32 note) segments and replicates each onset's accuracy.
    Returns the mean accuracy weighted by duration.
    """
    import pandas as pd
    import copy

    if len(acc_onset_level) == 0 or len(onset_times) == 0:
        return 0.0

    # Debug: check onset time range
    onset_min, onset_max = onset_times.min(), onset_times.max()
    if onset_max - onset_min > 200:
        print("Warning: Very large onset time range: {:.2f} to {:.2f} (range={:.2f})".format(
            onset_min, onset_max, onset_max - onset_min))
        print("  This might indicate onset times are in wrong units. Normalizing...")
        onset_times = (onset_times - onset_min) / (onset_max - onset_min) * 100  # Normalize to 0-100

    df = pd.DataFrame({"onset": onset_times, "acc": acc_onset_level})
    dfout = copy.deepcopy(df)
    dfout["onset"] = dfout["onset"] - dfout["onset"].min()

    for i in range(1, len(df)):
        onset_diff = int((df["onset"][i] - df["onset"][i - 1]) / 0.125) - 1

        # Sanity check: cap at 1000 segments per gap (125 beats or ~2 minutes at 60 BPM)
        # This prevents infinite loops if onset times are corrupted
        if onset_diff < 0:
            onset_diff = 0
        elif onset_diff > 1000:
            print("Warning: Large gap between onsets ({} segments), capping at 1000".format(onset_diff))
            onset_diff = 1000

        row_data = {"onset": df.iloc[i - 1]["onset"], "acc": df.iloc[i - 1]["acc"]}
        for j in range(onset_diff):
            row_data["onset"] = row_data["onset"] + 0.125
            dfout = pd.concat([dfout, pd.DataFrame([row_data])], ignore_index=True)

    dfout.sort_values(by="onset", inplace=True)
    return dfout["acc"].to_numpy().mean()


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

    banner("Step 2: Load checkpoint to infer task configuration")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", {})
    hparams = ckpt.get("hyper_parameters", {}) or {}

    # Infer task dimensions from checkpoint heads
    tasks_from_ckpt = {}
    for key in state_dict.keys():
        # Look for classifier heads: <prefix>.classifier.<task>.layers.1.weight
        # Examples:
        #   frozen_model.classifier.classifier.romanNumeral.layers.1.weight
        #   classifier.classifier.romanNumeral.layers.1.weight
        #   classifier.romanNumeral.layers.1.weight
        if "classifier." in key and ".layers.1.weight" in key:
            parts = key.split(".")
            # Find the last "classifier" occurrence, task name is right after
            for i in range(len(parts) - 1, -1, -1):
                if parts[i] == "classifier" and i + 1 < len(parts):
                    task_name = parts[i + 1]
                    if task_name != "classifier" and task_name != "layers":
                        out_dim = state_dict[key].shape[0]
                        tasks_from_ckpt[task_name] = out_dim
                        break

    print("✓ Tasks inferred from checkpoint ({} tasks):".format(len(tasks_from_ckpt)))
    for task, dim in sorted(tasks_from_ckpt.items()):
        print("  {}: {}".format(task, dim))

    if len(tasks_from_ckpt) == 0:
        raise RuntimeError("Could not detect any tasks from checkpoint! Check key names.")

    # Dataset only supports num_tasks of 6, 11, or 14
    # Cap at 14 (the max)
    num_tasks_detected = len(tasks_from_ckpt)
    if num_tasks_detected <= 6:
        num_tasks_actual = 6
    elif num_tasks_detected <= 11:
        num_tasks_actual = 11
    else:
        num_tasks_actual = 14

    print("✓ Detected {} tasks from checkpoint, using num_tasks={}".format(num_tasks_detected, num_tasks_actual))

    # Get task order from the dataset class
    if DATA_VERSION == "v1.0.0":
        DatasetCls = st.data.datasets.chord.AugmentedNetChordGraphDataset
    else:
        DatasetCls = st.data.datasets.chord.Augmented2022ChordGraphDataset

    banner("Step 3: Copy TEST TSVs into cache")
    copy_all_test_tsvs_into_cache(MOZART_ROOT, CACHE_ROOT, DATA_VERSION)

    banner("Step 4: Build test dataset with correct num_tasks")
    # Try test split, fallback to all
    try:
        test_dataset = DatasetCls(
            raw_dir=CACHE_ROOT,
            force_reload=True,
            nprocs=max(1, NUM_WORKERS),
            include_synth=False,
            num_tasks=num_tasks_actual,
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
            num_tasks=num_tasks_actual,
            collection="all",
        )
        print("✓ Test dataset built with collection='all'")

    print("✓ Test dataset length: {}".format(len(test_dataset)))

    # Get the actual task order from the dataset
    if hasattr(test_dataset, 'tasks'):
        TASK_ORDER = list(test_dataset.tasks.keys())
    elif hasattr(test_dataset, 'task_names'):
        TASK_ORDER = test_dataset.task_names
    elif hasattr(test_dataset.dataset, 'tasks'):
        TASK_ORDER = list(test_dataset.dataset.tasks.keys())
    else:
        # Fallback to hardcoded order (risky!)
        print("WARNING: Could not detect dataset task order, using hardcoded order")
        TASK_ORDER = [
            "localkey", "tonkey", "degree1", "degree2", "quality", "inversion",
            "root", "romanNumeral", "hrhythm", "pcset", "bass", "tenor", "alto", "soprano"
        ]

    print("✓ Dataset task order:")
    for i, task in enumerate(TASK_ORDER):
        print("  [{}] {}".format(i, task))

    # Print vocabulary mappings for key tasks
    print("\n=== VOCABULARY MAPPINGS ===")
    if hasattr(test_dataset, 'tasks'):
        if 'romanNumeral' in test_dataset.tasks:
            print("romanNumeral vocab size: {}".format(test_dataset.tasks['romanNumeral']))
        if 'localkey' in test_dataset.tasks:
            print("localkey vocab size: {}".format(test_dataset.tasks['localkey']))
        if 'inversion' in test_dataset.tasks:
            print("inversion vocab size: {}".format(test_dataset.tasks['inversion']))

    # Try to get actual class lists
    try:
        if DATA_VERSION == "v2.0.0":
            from chordgnn.utils.chord_representations_latest import COMMON_ROMAN_NUMERALS, KEYS
            print("\nRomanNumeral classes (first 10): {}".format(list(COMMON_ROMAN_NUMERALS[:10])))
            print("LocalKey classes (first 10): {}".format(list(KEYS[:10])))
            print("Inversion classes: [0=root, 1=first, 2=second, 3=third]")
        else:
            from chordgnn.utils.chord_representations import COMMON_ROMAN_NUMERALS, KEYS
            print("\nRomanNumeral classes (first 10): {}".format(list(COMMON_ROMAN_NUMERALS[:10])))
            print("LocalKey classes (first 10): {}".format(list(KEYS[:10])))
            print("Inversion classes: [0=root, 1=first, 2=second, 3=third]")
    except Exception as e:
        print("Could not load class lists: {}".format(e))
    print("="*35 + "\n")

    banner("Step 5: Load model from checkpoint")

    print("\n=== CHECKPOINT VERIFICATION ===")
    print("Checkpoint path:        {}".format(ckpt_path))
    print("WANDB_ARTIFACT:         {}".format(WANDB_ARTIFACT))
    print("MODEL_TYPE env var:     {}".format(MODEL_TYPE))
    print("LOCAL_CKPT env var:     {}".format(LOCAL_CKPT if LOCAL_CKPT else "(not set)"))

    # Debug: show first 10 keys to detect model type
    print("\nFirst 10 checkpoint keys:")
    for k in list(state_dict.keys())[:10]:
        print("  {}".format(k))

    # Determine if this is a finetuned (PostChordPrediction) or base model
    is_finetuned = any(k.startswith("frozen_model.") for k in state_dict.keys())
    has_module_prefix = any(k.startswith("module.") for k in state_dict.keys())

    print("\nModel type detection:")
    print("  Has 'frozen_model.' prefix: {}".format(is_finetuned))
    print("  Has 'module.' prefix:       {}".format(has_module_prefix))
    print("  Detected model type:        {}".format("PostChordPrediction (finetuned)" if is_finetuned else "ChordPrediction (base)"))
    print("="*35 + "\n")

    if is_finetuned or MODEL_TYPE == "post":
        print("Detected finetuned model (PostChordPrediction)")
        from chordgnn.models.chord import ChordPredictionModel

        # Load the frozen encoder
        frozen_model = ChordPredictionModel(in_feats=352)  # Will be overridden by checkpoint

        # Load the full model
        ModelClass = st.models.chord.PostChordPrediction
        model = ModelClass.load_from_checkpoint(ckpt_path, strict=False, map_location="cpu")
        use_frozen = True
    else:
        print("Detected base pretrained model (ChordPrediction)")
        ModelClass = st.models.chord.ChordPrediction
        model = ModelClass.load_from_checkpoint(ckpt_path, strict=False, map_location="cpu")
        use_frozen = False

    print("✓ Model loaded from checkpoint")

    banner("Step 5: Manual evaluation loop")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    from chordgnn.utils.hgraph import add_reverse_edges_from_edge_index
    from chordgnn.models.chord import unique_onsets

    loader = DataLoader(test_dataset, batch_size=1, num_workers=0, shuffle=False)

    # TASK_ORDER is already set from dataset detection above
    correct_by_task = defaultdict(int)
    total_by_task = defaultdict(int)

    # RNalt: romanNumeral + localkey + inversion all correct
    # Track duration-weighted correctness (proper CSR)
    rnalt_correct_time = 0.0  # Sum of durations where correct
    rnalt_total_time = 0.0    # Sum of all durations
    rnalt_onset_correct = 0   # Onset-level for comparison
    rnalt_onset_total = 0

    # Val RomNum: degree1 + degree2 + quality + root + inversion + localkey all correct
    romnum_correct = 0
    romnum_total = 0

    # Per-component error tracking
    rn_errors = 0  # romanNumeral wrong
    lk_errors = 0  # localkey wrong
    inv_errors = 0  # inversion wrong
    total_comparisons = 0

    # Per-piece CSR tracking
    piece_csr_list = []

    # Cosine similarity-based RN resolution (like analyse_score.py)
    cosine_rn_correct = 0
    cosine_rn_total = 0
    cosine_rn_correct_time = 0.0
    cosine_rn_total_time = 0.0

    # One-time sanity check
    first_batch_checked = False

    # Load class lists for decoding
    try:
        if DATA_VERSION == "v2.0.0":
            from chordgnn.utils.chord_representations_latest import COMMON_ROMAN_NUMERALS, KEYS
        else:
            from chordgnn.utils.chord_representations import COMMON_ROMAN_NUMERALS, KEYS
        RN_CLASSES = list(COMMON_ROMAN_NUMERALS)
        LK_CLASSES = list(KEYS)
    except:
        RN_CLASSES = None
        LK_CLASSES = None

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            x, edges, edge_type, labels, onset_div, name = batch

            x = x.squeeze(0).float().to(device)
            edges = edges.squeeze(0).to(device)
            edge_type = edge_type.squeeze(0).to(device)
            onset_div = onset_div.squeeze().to(device)

            # Process labels
            labels = labels.squeeze(0)
            if labels.ndim == 3 and labels.shape[-1] == 1:
                labels = labels.squeeze(-1)
            labels = labels[:, :len(TASK_ORDER)]

            # Prepare model inputs
            onset_edges = edges[:, edge_type == 0]
            edges2, edge_type2 = add_reverse_edges_from_edge_index(edges, edge_type)
            onset_idx = unique_onsets(onset_div)

            # Forward pass
            if use_frozen:
                # For finetuned models: frozen_model then module
                x_encoded = model.frozen_model((x, edges2, edge_type2, onset_edges, onset_idx, None))
                out = model.module(x_encoded)
            else:
                # For base models: direct module call
                out = model.module((x, edges2, edge_type2, onset_edges, onset_idx, None))

            preds = out if isinstance(out, dict) else (out[0] if isinstance(out, (list, tuple)) else out)

            # One-time sanity check
            if not first_batch_checked:
                print("\n" + "="*70)
                print("SANITY CHECK - First Batch")
                print("="*70)
                print("Prediction keys:", sorted(preds.keys()))
                print("TASK_ORDER:     ", TASK_ORDER)
                print("Labels shape:   ", labels.shape)
                print("\nTask order alignment check:")
                for i, task in enumerate(TASK_ORDER):
                    in_preds = task in preds
                    pred_shape = str(preds[task].shape) if in_preds else "N/A"
                    label_unique = labels[:, i].unique().tolist() if i < labels.shape[1] else []
                    print(f"  [{i:2d}] {task:15s} in_preds={str(in_preds):5s} pred_shape={pred_shape:20s} label_range={str(label_unique[:5]):20s}")

                print("\nForward pass check:")
                print(f"  Model type: {type(model).__name__}")
                print(f"  use_frozen: {use_frozen}")
                print(f"  Output type: {type(out)}")
                print(f"  Preds type: {type(preds)}")

                print("\nLabel column check (first row, first 14 cols):")
                if labels.shape[0] > 0:
                    print("  ", labels[0, :14].tolist())

                # Show sample predictions vs ground truth
                if all(k in preds for k in ("romanNumeral", "localkey", "inversion")) and RN_CLASSES and LK_CLASSES:
                    print("\n" + "-"*70)
                    print("SAMPLE PREDICTIONS VS GROUND TRUTH (First 15 onsets)")
                    print("-"*70)
                    print(f"{'Onset':<6} {'Pred RN':<12} {'GT RN':<12} {'Pred LK':<8} {'GT LK':<8} {'Pred Inv':<4} {'GT Inv':<4} {'Match':<6}")
                    print("-"*70)

                    rn_pred = preds["romanNumeral"].argmax(dim=-1)
                    lk_pred = preds["localkey"].argmax(dim=-1)
                    inv_pred = preds["inversion"].argmax(dim=-1)

                    rn_idx = TASK_ORDER.index("romanNumeral")
                    lk_idx = TASK_ORDER.index("localkey")
                    inv_idx = TASK_ORDER.index("inversion")

                    rn_gt = align_target_to_pred_length(labels[:, rn_idx].long().to(device), rn_pred.shape[0], onset_idx)
                    lk_gt = align_target_to_pred_length(labels[:, lk_idx].long().to(device), lk_pred.shape[0], onset_idx)
                    inv_gt = align_target_to_pred_length(labels[:, inv_idx].long().to(device), inv_pred.shape[0], onset_idx)

                    max_samples = min(15, rn_pred.shape[0])
                    for i in range(max_samples):
                        if rn_gt[i] >= 0 and lk_gt[i] >= 0 and inv_gt[i] >= 0:
                            pred_rn_str = RN_CLASSES[rn_pred[i].item()] if rn_pred[i].item() < len(RN_CLASSES) else f"?{rn_pred[i].item()}"
                            gt_rn_str = RN_CLASSES[rn_gt[i].item()] if rn_gt[i].item() < len(RN_CLASSES) else f"?{rn_gt[i].item()}"
                            pred_lk_str = LK_CLASSES[lk_pred[i].item()] if lk_pred[i].item() < len(LK_CLASSES) else f"?{lk_pred[i].item()}"
                            gt_lk_str = LK_CLASSES[lk_gt[i].item()] if lk_gt[i].item() < len(LK_CLASSES) else f"?{lk_gt[i].item()}"
                            pred_inv = inv_pred[i].item()
                            gt_inv = inv_gt[i].item()

                            match = "✓" if (rn_pred[i] == rn_gt[i] and lk_pred[i] == lk_gt[i] and inv_pred[i] == inv_gt[i]) else "✗"
                            print(f"{i:<6} {pred_rn_str:<12} {gt_rn_str:<12} {pred_lk_str:<8} {gt_lk_str:<8} {pred_inv:<4} {gt_inv:<4} {match:<6}")
                    print("-"*70)

                first_batch_checked = True
                print("="*70 + "\n")

            # Compute per-task accuracy
            for t_i, tname in enumerate(TASK_ORDER):
                if tname not in preds:
                    continue
                pred_logits = preds[tname]
                target = labels[:, t_i].long().to(device)
                target_aligned = align_target_to_pred_length(target, pred_logits.shape[0], onset_idx)
                c, tot = masked_accuracy(pred_logits, target_aligned)
                correct_by_task[tname] += c
                total_by_task[tname] += tot

            # Compute RNalt (romanNumeral + localkey + inversion)
            # Duration-weighted CSR (correct method)
            if all(k in preds for k in ("romanNumeral", "localkey", "inversion")) and \
               all(k in TASK_ORDER for k in ("romanNumeral", "localkey", "inversion")):
                rn_pred = preds["romanNumeral"].argmax(dim=-1)
                lk_pred = preds["localkey"].argmax(dim=-1)
                inv_pred = preds["inversion"].argmax(dim=-1)

                rn_t = align_target_to_pred_length(labels[:, TASK_ORDER.index("romanNumeral")].long().to(device), rn_pred.shape[0], onset_idx)
                lk_t = align_target_to_pred_length(labels[:, TASK_ORDER.index("localkey")].long().to(device), lk_pred.shape[0], onset_idx)
                inv_t = align_target_to_pred_length(labels[:, TASK_ORDER.index("inversion")].long().to(device), inv_pred.shape[0], onset_idx)

                mask = (rn_t >= 0) & (lk_t >= 0) & (inv_t >= 0)
                # Per-onset correctness (1.0 if correct, 0.0 if wrong)
                rnalt_onset_acc = ((rn_pred == rn_t) & (lk_pred == lk_t) & (inv_pred == inv_t) & mask).float()

                # Duration-weighted CSR computation (per piece)
                if onset_idx is not None and len(onset_idx) > 0:
                    onset_idx_flat = onset_idx.view(-1).long()
                    times = onset_div[onset_idx_flat].float()  # divisions, monotonic within piece

                    # Compute durations in divisions
                    if times.numel() >= 2:
                        dur = torch.diff(times)  # duration = next_onset - this_onset
                        acc = rnalt_onset_acc[:dur.numel()]  # align to durations
                        mask2 = mask[:dur.numel()]  # same mask

                        # Only count valid labels
                        dur_valid = dur[mask2]
                        acc_valid = acc[mask2]

                        # Accumulate duration-weighted correctness
                        piece_correct_time = (acc_valid * dur_valid).sum().item()
                        piece_total_time = dur_valid.sum().item()

                        rnalt_correct_time += piece_correct_time
                        rnalt_total_time += piece_total_time

                        # Track per-piece CSR
                        if piece_total_time > 0:
                            piece_csr = 100.0 * piece_correct_time / piece_total_time
                            piece_csr_list.append((name[0] if isinstance(name, (list, tuple)) else str(name), piece_csr))

                # Also track onset-level for comparison
                rnalt_onset_correct += int((rnalt_onset_acc * mask.float()).sum().item())
                rnalt_onset_total += int(mask.sum().item())

                # Track per-component errors
                rn_wrong = ((rn_pred != rn_t) & mask).sum().item()
                lk_wrong = ((lk_pred != lk_t) & mask).sum().item()
                inv_wrong = ((inv_pred != inv_t) & mask).sum().item()
                valid_count = mask.sum().item()

                rn_errors += rn_wrong
                lk_errors += lk_wrong
                inv_errors += inv_wrong
                total_comparisons += valid_count

            # Compute Val RomNum (degree1+degree2+quality+root+inversion+localkey)
            if all(k in preds for k in ("degree1", "degree2", "quality", "root", "inversion", "localkey")) and \
               all(k in TASK_ORDER for k in ("degree1", "degree2", "quality", "root", "inversion", "localkey")):
                d1_pred = preds["degree1"].argmax(dim=-1)
                d2_pred = preds["degree2"].argmax(dim=-1)
                q_pred = preds["quality"].argmax(dim=-1)
                r_pred = preds["root"].argmax(dim=-1)
                i_pred = preds["inversion"].argmax(dim=-1)
                lk_pred = preds["localkey"].argmax(dim=-1)

                d1_t = align_target_to_pred_length(labels[:, TASK_ORDER.index("degree1")].long().to(device), d1_pred.shape[0], onset_idx)
                d2_t = align_target_to_pred_length(labels[:, TASK_ORDER.index("degree2")].long().to(device), d2_pred.shape[0], onset_idx)
                q_t = align_target_to_pred_length(labels[:, TASK_ORDER.index("quality")].long().to(device), q_pred.shape[0], onset_idx)
                r_t = align_target_to_pred_length(labels[:, TASK_ORDER.index("root")].long().to(device), r_pred.shape[0], onset_idx)
                i_t = align_target_to_pred_length(labels[:, TASK_ORDER.index("inversion")].long().to(device), i_pred.shape[0], onset_idx)
                lk_t = align_target_to_pred_length(labels[:, TASK_ORDER.index("localkey")].long().to(device), lk_pred.shape[0], onset_idx)

                mask = (d1_t >= 0) & (d2_t >= 0) & (q_t >= 0) & (r_t >= 0) & (i_t >= 0) & (lk_t >= 0)
                tot = int(mask.sum().item())
                if tot > 0:
                    degree_match = (d1_pred == d1_t) & (d2_pred == d2_t)
                    corr = int((degree_match & (q_pred == q_t) & (r_pred == r_t) & (i_pred == i_t) & (lk_pred == lk_t) & mask).sum().item())
                    romnum_correct += corr
                    romnum_total += tot

            # Compute cosine similarity-based Roman numeral resolution (like analyse_score.py)
            if COSINE_AVAILABLE and all(k in preds for k in ("bass", "tenor", "alto", "soprano", "pcset", "localkey", "tonkey", "romanNumeral")):
                # Get predictions
                bass_pred = preds["bass"].argmax(dim=-1).cpu()
                tenor_pred = preds["tenor"].argmax(dim=-1).cpu()
                alto_pred = preds["alto"].argmax(dim=-1).cpu()
                soprano_pred = preds["soprano"].argmax(dim=-1).cpu()
                pcset_pred = preds["pcset"].argmax(dim=-1).cpu()
                localkey_pred = preds["localkey"].argmax(dim=-1).cpu()
                tonkey_pred = preds["tonkey"].argmax(dim=-1).cpu()

                # Get ground truth
                rn_idx = TASK_ORDER.index("romanNumeral")
                lk_idx = TASK_ORDER.index("localkey")
                rn_gt = align_target_to_pred_length(labels[:, rn_idx].long().to(device), bass_pred.shape[0], onset_idx).cpu()
                lk_gt = align_target_to_pred_length(labels[:, lk_idx].long().to(device), bass_pred.shape[0], onset_idx).cpu()

                # Select appropriate class lists based on data version
                # Note: resolveRomanNumeralCosine only exists in v1, so we use it for both
                if DATA_VERSION == "v2.0.0":
                    RN_CLS = RN_LATEST
                    KEY_CLS = KEYS_LATEST
                else:
                    RN_CLS = RN_V1
                    KEY_CLS = KEYS_V1

                # Use v1 cosine function for both (it's the only one that exists)
                resolve_fn = resolveRomanNumeralCosine

                mask_cosine = rn_gt >= 0
                correct_cosine = torch.zeros(bass_pred.shape[0], dtype=torch.bool)

                for i in range(bass_pred.shape[0]):
                    if not mask_cosine[i]:
                        continue

                    try:
                        # Decode predictions
                        b = KEY_CLS[bass_pred[i].item()] if bass_pred[i] < len(KEY_CLS) else "C"
                        t = KEY_CLS[tenor_pred[i].item()] if tenor_pred[i] < len(KEY_CLS) else "C"
                        a = KEY_CLS[alto_pred[i].item()] if alto_pred[i] < len(KEY_CLS) else "C"
                        s = KEY_CLS[soprano_pred[i].item()] if soprano_pred[i] < len(KEY_CLS) else "C"
                        key = KEY_CLS[localkey_pred[i].item()] if localkey_pred[i] < len(KEY_CLS) else "C"
                        tonkey = KEY_CLS[tonkey_pred[i].item()] if tonkey_pred[i] < len(KEY_CLS) else "C"

                        # For pcset, we need the actual pitch class set, not just the index
                        # This is a simplification - in analyse_score.py they use actual pitch data
                        # Here we'll just use an empty pcset since we don't have the actual notes
                        pcs = []

                        # Get predicted RN using argmax for the numerator parameter
                        rn_argmax_idx = preds["romanNumeral"].argmax(dim=-1)[i].item()
                        numerator = RN_CLS[rn_argmax_idx] if rn_argmax_idx < len(RN_CLS) else "I"

                        # Resolve using cosine similarity
                        resolved = resolve_fn(b, t, a, s, pcs, key, numerator, tonkey)
                        resolved_rn = resolved[0]  # Returns (figure, inversion, ...)

                        # Get ground truth RN
                        gt_rn = RN_CLS[rn_gt[i].item()] if rn_gt[i] < len(RN_CLS) else "I"

                        # Compare
                        if resolved_rn == gt_rn:
                            correct_cosine[i] = True
                    except Exception as e:
                        # If resolution fails, mark as incorrect
                        pass

                # Track onset-level accuracy
                cosine_rn_correct += correct_cosine[mask_cosine].sum().item()
                cosine_rn_total += mask_cosine.sum().item()

                # Track duration-weighted accuracy
                if onset_idx is not None and len(onset_idx) > 0:
                    onset_idx_flat = onset_idx.view(-1).long()
                    times = onset_div[onset_idx_flat].float().cpu()

                    if times.numel() >= 2:
                        dur = torch.diff(times)
                        acc_cosine = correct_cosine[:dur.numel()].float()
                        mask2_cosine = mask_cosine[:dur.numel()]

                        dur_valid = dur[mask2_cosine]
                        acc_valid = acc_cosine[mask2_cosine]

                        cosine_rn_correct_time += (acc_valid * dur_valid).sum().item()
                        cosine_rn_total_time += dur_valid.sum().item()

            if (idx + 1) % 100 == 0:
                print("Processed {}/{} graphs...".format(idx + 1, len(loader)))

    banner("TEST RESULTS")

    print("\n=== EVALUATION SUMMARY ===")
    print(f"Total batches processed:  {idx + 1}")
    print(f"RNalt pieces with timing: {int(rnalt_total_time > 0)}")
    print(f"RNalt onset total:        {rnalt_onset_total}")
    print(f"Val RomNum total:         {romnum_total}")
    print("="*30 + "\n")

    print("Per-task accuracy:")
    for tname in TASK_ORDER:
        tot = total_by_task[tname]
        if tot == 0:
            print("{:14s}: n/a (total=0 - CHECK TASK ORDER!)".format(tname))
        else:
            acc = 100.0 * correct_by_task[tname] / tot
            print("{:14s}: {:7.2f}% ({}/{})".format(tname, acc, correct_by_task[tname], tot))

    print("\nComposite metrics:")
    if rnalt_total_time > 0:
        # Duration-weighted CSR (correct metric)
        csr = 100.0 * rnalt_correct_time / rnalt_total_time
        print("CSR (romanNumeral+localkey+inversion, duration-weighted): {:.2f}%".format(csr))
        print("  ↳ This is the correct time-weighted RNalt CSR metric")
        print("  ↳ Weights correctness by onset durations (divisions)")

        # Also show onset-level for comparison
        if rnalt_onset_total > 0:
            onset_rnalt = 100.0 * rnalt_onset_correct / rnalt_onset_total
            print("RNalt (onset-level, for comparison):                  {:.2f}%".format(onset_rnalt))
    else:
        print("CSR/RNalt: n/a (romanNumeral task not available)")

    if romnum_total > 0:
        romnum_acc = 100.0 * romnum_correct / romnum_total
        print("Val RomNum (degree1+degree2+quality+root+inversion+localkey): {:.2f}% ({}/{})".format(romnum_acc, romnum_correct, romnum_total))
    else:
        print("Val RomNum: n/a (not all required tasks available)")

    # Report cosine similarity-based RN resolution
    if cosine_rn_total_time > 0:
        print("\n" + "="*70)
        print("COSINE SIMILARITY-BASED RN RESOLUTION (like analyse_score.py)")
        print("="*70)
        print("This resolves Roman numerals from voices+pcset+key using cosine similarity")
        print("instead of directly using the romanNumeral task output.")
        print()
        cosine_csr = 100.0 * cosine_rn_correct_time / cosine_rn_total_time
        print("Cosine RN accuracy (duration-weighted): {:.2f}%".format(cosine_csr))
        if cosine_rn_total > 0:
            cosine_onset = 100.0 * cosine_rn_correct / cosine_rn_total
            print("Cosine RN accuracy (onset-level):      {:.2f}%".format(cosine_onset))
        print("\nComparison:")
        print("  Direct RNalt CSR (RN+LK+INV):         {:.2f}%".format(csr if rnalt_total_time > 0 else 0.0))
        print("  Cosine-resolved RN accuracy:          {:.2f}%".format(cosine_csr))
        print("  Δ (Cosine - Direct):                  {:+.2f}%".format(cosine_csr - (csr if rnalt_total_time > 0 else 0.0)))
    elif COSINE_AVAILABLE:
        print("\nCosine RN resolution: n/a (required tasks not available)")
    else:
        print("\nCosine RN resolution: n/a (cosine functions not imported)")

    # Per-component error breakdown
    if total_comparisons > 0:
        print("\n=== PER-COMPONENT ERROR BREAKDOWN ===")
        print("(Which component is wrong most often?)")
        print("  RomanNumeral errors: {:6d} ({:5.2f}%)".format(rn_errors, 100.0 * rn_errors / total_comparisons))
        print("  LocalKey errors:     {:6d} ({:5.2f}%)".format(lk_errors, 100.0 * lk_errors / total_comparisons))
        print("  Inversion errors:    {:6d} ({:5.2f}%)".format(inv_errors, 100.0 * inv_errors / total_comparisons))
        print("  Total comparisons:   {:6d}".format(total_comparisons))
        print("  Note: These overlap (one onset can have multiple errors)")

    # Per-piece CSR breakdown
    if piece_csr_list:
        print("\n=== PER-PIECE CSR BREAKDOWN ===")
        print("(Top 10 worst and best pieces)")
        sorted_pieces = sorted(piece_csr_list, key=lambda x: x[1])
        print("\nWorst 10 pieces:")
        for piece_name, csr in sorted_pieces[:10]:
            print("  {:30s} {:6.2f}%".format(piece_name, csr))
        print("\nBest 10 pieces:")
        for piece_name, csr in sorted_pieces[-10:]:
            print("  {:30s} {:6.2f}%".format(piece_name, csr))

    # Sanity check: warn if any tasks have suspiciously low totals
    print("\nSanity check - label totals per task:")
    avg_total = sum(total_by_task.values()) / len(total_by_task) if total_by_task else 0
    for tname in TASK_ORDER:
        tot = total_by_task[tname]
        if tot < avg_total * 0.5 and avg_total > 0:
            print("  ⚠️  {}: {} (much lower than avg {:.0f})".format(tname, tot, avg_total))
        else:
            print("  ✓  {}: {}".format(tname, tot))


if __name__ == "__main__":
    main()
