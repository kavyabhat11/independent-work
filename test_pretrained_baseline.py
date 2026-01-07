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
        # Look for classifier heads: frozen_model.classifier.classifier.<task>.layers.1.weight
        # or classifier.classifier.<task>.layers.1.weight
        if ".classifier." in key and ".layers.1.weight" in key:
            parts = key.split(".")
            if "classifier" in parts:
                idx = parts.index("classifier")
                if idx + 1 < len(parts):
                    task_name = parts[idx + 1]
                    out_dim = state_dict[key].shape[0]
                    tasks_from_ckpt[task_name] = out_dim

    print("✓ Tasks inferred from checkpoint:")
    for task, dim in tasks_from_ckpt.items():
        print("  {}: {}".format(task, dim))

    num_tasks_actual = len(tasks_from_ckpt)
    print("✓ num_tasks = {}".format(num_tasks_actual))

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

    banner("Step 5: Load model from checkpoint")
    # Determine if this is a finetuned (PostChordPrediction) or base model
    is_finetuned = any(k.startswith("frozen_model.") for k in state_dict.keys())

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
    rnalt_correct = 0
    rnalt_total = 0

    # Val RomNum: degree1 + degree2 + quality + root + inversion + localkey all correct
    romnum_correct = 0
    romnum_total = 0

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
                preds = model.module(x_encoded)
            else:
                # For base models: direct forward
                preds = model.module((x, edges2, edge_type2, onset_edges, onset_idx, None))

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
            if all(k in preds for k in ("romanNumeral", "localkey", "inversion")):
                rn_pred = preds["romanNumeral"].argmax(dim=-1)
                lk_pred = preds["localkey"].argmax(dim=-1)
                inv_pred = preds["inversion"].argmax(dim=-1)

                rn_t = align_target_to_pred_length(labels[:, TASK_ORDER.index("romanNumeral")].long().to(device), rn_pred.shape[0], onset_idx)
                lk_t = align_target_to_pred_length(labels[:, TASK_ORDER.index("localkey")].long().to(device), lk_pred.shape[0], onset_idx)
                inv_t = align_target_to_pred_length(labels[:, TASK_ORDER.index("inversion")].long().to(device), inv_pred.shape[0], onset_idx)

                mask = (rn_t >= 0) & (lk_t >= 0) & (inv_t >= 0)
                tot = int(mask.sum().item())
                if tot > 0:
                    corr = int(((rn_pred == rn_t) & (lk_pred == lk_t) & (inv_pred == inv_t) & mask).sum().item())
                    rnalt_correct += corr
                    rnalt_total += tot

            # Compute Val RomNum (degree1+degree2+quality+root+inversion+localkey)
            if all(k in preds for k in ("degree1", "degree2", "quality", "root", "inversion", "localkey")):
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

            if (idx + 1) % 100 == 0:
                print("Processed {}/{} graphs...".format(idx + 1, len(loader)))

    banner("TEST RESULTS")
    print("Per-task accuracy:")
    for tname in TASK_ORDER:
        tot = total_by_task[tname]
        if tot == 0:
            print("{:14s}: n/a (total=0 - CHECK TASK ORDER!)".format(tname))
        else:
            acc = 100.0 * correct_by_task[tname] / tot
            print("{:14s}: {:7.2f}% ({}/{})".format(tname, acc, correct_by_task[tname], tot))

    print("\nComposite metrics:")
    if rnalt_total > 0:
        rnalt_acc = 100.0 * rnalt_correct / rnalt_total
        print("RNalt (romanNumeral+localkey+inversion): {:.2f}% ({}/{})".format(rnalt_acc, rnalt_correct, rnalt_total))
    else:
        print("RNalt: n/a (total=0)")

    if romnum_total > 0:
        romnum_acc = 100.0 * romnum_correct / romnum_total
        print("Val RomNum (all 5 components):           {:.2f}% ({}/{})".format(romnum_acc, romnum_correct, romnum_total))
    else:
        print("Val RomNum: n/a (total=0)")

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
