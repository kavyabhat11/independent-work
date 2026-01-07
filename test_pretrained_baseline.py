#!/usr/bin/env python3
"""
DEBUG baseline eval on EXACTLY ONE Mozart validation TSV
using the PRETRAINED checkpoint:
  melkisedeath/chord_rec/model-kvd0jic5:v0

This version is "anchored to checkpoint truth":
- Uses task output dims inferred from the checkpoint heads (NOT available_representations)
- Forces n_hidden=256 to match checkpoint head weights
- Strips "frozen_model." prefix when loading weights
- Handles onset-level vs frame-level by aligning targets to pred length
- Masks invalid labels (<0)
- Runs ONE graph (one TSV) to make debugging easy

Usage:
  export MOZART_ROOT=/path/to/mozart_dataset              # must contain validation/*.tsv
  export DEBUG_TSV=/path/to/mozart_dataset/validation/X.tsv   # optional
  python debug_onefile_baseline.py

Notes:
- Keep NUM_WORKERS=0 while debugging.
- If wandb is not logged in: wandb login
"""

import os
import glob
import shutil
import math
from typing import Optional
from collections import defaultdict

import torch

# ----------------------------
# CONFIG
# ----------------------------
WANDB_ARTIFACT = os.environ.get("WANDB_ARTIFACT", "melkisedeath/chord_rec/model-kvd0jic5:v0")
ARTIFACT_ROOT = os.environ.get("ARTIFACT_ROOT", "./artifacts")

MOZART_ROOT = os.environ.get("MOZART_ROOT", "./mozart_dataset")
DEBUG_TSV = os.environ.get("DEBUG_TSV", "")  # optional exact TSV path

CACHE_ROOT = os.environ.get("CACHE_ROOT", "/scratch/network/kb9520/chordgnn_data")
DATA_VERSION = os.environ.get("DATA_VERSION", "v2.0.0")

NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "0"))  # keep 0 for debugging

# The ONLY correct task dims for this checkpoint (from your ckpt head weights)
TASK_DIMS = {
    "localkey": 38,
    "tonkey": 38,
    "degree1": 22,
    "degree2": 22,
    "quality": 11,
    "inversion": 4,
    "root": 35,
    "romanNumeral": 31,
    "hrhythm": 7,
    "pcset": 121,
    "bass": 35,
    "tenor": 35,
    "alto": 35,
    "soprano": 35,
}

# Must match checkpoint heads
FORCE_N_HIDDEN = 256


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


def strip_state_dict_prefixes(state_dict: dict) -> dict:
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith(("train_loss.", "val_loss.", "test_loss.")):
            continue
        nk = k
        if nk.startswith("frozen_model."):
            nk = nk[len("frozen_model."):]
        if nk.startswith("module."):
            nk = nk[len("module."):]
        cleaned[nk] = v
    return cleaned


def masked_accuracy(pred_logits: torch.Tensor, target: torch.Tensor):
    pred_classes = pred_logits.argmax(dim=-1)
    mask = target >= 0
    total = int(mask.sum().item())
    if total == 0:
        return 0, 0
    correct = int((pred_classes[mask] == target[mask]).sum().item())
    return correct, total


def align_target_to_pred_length(target: torch.Tensor, pred_len: int, onset_idx: Optional[torch.Tensor]):
    """
    If preds are onset-level and targets are frame-level, index targets by onset_idx.
    Otherwise truncate/pad to match.
    """
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


def copy_all_validation_tsvs_into_cache(mozart_root: str, cache_root: str, data_version: str) -> int:
    if data_version == "v2.0.0":
        base = os.path.join(cache_root, "AugmentedNetLatestChordDataset", "dataset")
    else:
        base = os.path.join(cache_root, "AugmentedNetChordDataset", "dataset")

    val_dir = os.path.join(base, "validation")

    if os.path.exists(base):
        print("Cleaning existing dataset cache at: {}".format(base))
        shutil.rmtree(base)
    os.makedirs(val_dir, exist_ok=True)

    candidates = sorted(glob.glob(os.path.join(mozart_root, "validation", "*.tsv")))
    if not candidates:
        raise FileNotFoundError("No TSVs found under {}".format(os.path.join(mozart_root, "validation", "*.tsv")))

    for f in candidates:
        shutil.copy(f, os.path.join(val_dir, os.path.basename(f)))

    print("✓ Copied {} validation TSVs into cache:\n  {}".format(len(candidates), val_dir))
    return len(candidates)


# ----------------------------
# MAIN
# ----------------------------
def main():
    banner("DEBUG: PRETRAINED CHORDGNN BASELINE ON ONE MOZART VALIDATION TSV")

    print("MOZART_ROOT    = {}".format(MOZART_ROOT))
    print("DEBUG_TSV      = {}".format(DEBUG_TSV if DEBUG_TSV else "(auto: first validation/*.tsv)"))
    print("CACHE_ROOT     = {}".format(CACHE_ROOT))
    print("DATA_VERSION   = {}".format(DATA_VERSION))
    print("WANDB_ARTIFACT = {}".format(WANDB_ARTIFACT))

    banner("Step 1: Download ckpt")
    ckpt_path = download_wandb_ckpt(WANDB_ARTIFACT, ARTIFACT_ROOT)
    print("✓ ckpt: {}".format(ckpt_path))

    banner("Step 2: Load ckpt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", {})
    hparams = ckpt.get("hyper_parameters", {}) or {}
    print("ckpt has {} state_dict keys".format(len(state_dict)))
    # print a couple useful hparams if present
    for k in ("n_layers", "dropout", "use_nade", "use_jk"):
        if k in hparams:
            print("hparam {} = {}".format(k, hparams[k]))

    banner("Step 3: Put ONE TSV into cache")
    copy_all_validation_tsvs_into_cache(MOZART_ROOT, CACHE_ROOT, DATA_VERSION)

    banner("Step 4: Build dataset (single graph)")
    import chordgnn as st

    if DATA_VERSION == "v1.0.0":
        DatasetCls = st.data.datasets.chord.AugmentedNetChordGraphDataset
    else:
        DatasetCls = st.data.datasets.chord.Augmented2022ChordGraphDataset

    # Try validation split, fallback to all
    try:
        dataset = DatasetCls(
            raw_dir=CACHE_ROOT,
            force_reload=True,
            nprocs=max(1, NUM_WORKERS),
            include_synth=False,
            num_tasks=11,
            collection="validation",
        )
        print("✓ dataset built with collection='validation'")
    except Exception as e:
        print("collection='validation' failed, falling back to collection='all'")
        print("  error: {}".format(e))
        dataset = DatasetCls(
            raw_dir=CACHE_ROOT,
            force_reload=True,
            nprocs=max(1, NUM_WORKERS),
            include_synth=False,
            num_tasks=11,
            collection="all",
        )
        print("✓ dataset built with collection='all'")

    print("Dataset length:", len(dataset))

    banner("Step 5: Build model (checkpoint-truth tasks) + load weights")
    tasks = TASK_DIMS.copy()
    task_names = list(tasks.keys())
    print("✓ Tasks ({}): {}".format(len(task_names), task_names))

    # infer in_feats
    b0 = next(iter(torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)))
    x0 = b0[0]
    in_feats = int(x0.shape[-1])
    print("✓ in_feats =", in_feats)

    # use ckpt-consistent architecture
    n_hidden = FORCE_N_HIDDEN
    n_layers = int(hparams.get("n_layers", 6))
    dropout = float(hparams.get("dropout", 0.5))
    use_nade = bool(hparams.get("use_nade", False))
    use_jk = bool(hparams.get("use_jk", False))

    print("✓ Forcing n_hidden={} (matches ckpt heads)".format(n_hidden))
    print("✓ Using n_layers={}, dropout={}, use_nade={}, use_jk={}".format(n_layers, dropout, use_nade, use_jk))

    model = st.models.ChordPredictionModel(
        in_feats=in_feats,
        n_hidden=n_hidden,
        tasks=tasks,              # IMPORTANT: ints, not types
        n_layers=n_layers,
        dropout=dropout,
        use_nade=use_nade,
        use_jk=use_jk,
    )

    cleaned = strip_state_dict_prefixes(state_dict)
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print("✓ Loaded weights: {} tensors".format(len(cleaned)))
    print("  missing={}, unexpected={}".format(len(missing), len(unexpected)))
    if unexpected:
        print("  unexpected sample:", unexpected[:10])
    if missing:
        print("  missing sample:", missing[:10])

    banner("Step 6: Evaluation over ALL validation graphs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    from torch.utils.data import DataLoader
    from chordgnn.utils.hgraph import add_reverse_edges_from_edge_index
    from chordgnn.models.chord import unique_onsets

    loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)

    dataset_task_order = [
        "localkey", "tonkey", "degree1", "degree2", "quality", "inversion",
        "root", "romanNumeral", "hrhythm", "pcset", "bass", "tenor", "alto", "soprano"
    ]
    N_TASKS = len(dataset_task_order)

    correct_by_task = defaultdict(int)
    total_by_task = defaultdict(int)

    # CSR-like: exact chord match root+quality+inversion
    csr_correct = 0
    csr_total = 0

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            x, edges, edge_type, labels, onset_div, name = batch

            x = x.squeeze(0).float().to(device)
            edges = edges.squeeze(0).to(device)
            edge_type = edge_type.squeeze(0).to(device)
            onset_div = onset_div.squeeze().to(device)

            # labels: (T, 15, 1) -> (T, 15) -> take first 14
            labels = labels.squeeze(0)
            if labels.ndim == 3 and labels.shape[-1] == 1:
                labels = labels.squeeze(-1)
            if labels.ndim != 2 or labels.shape[1] < N_TASKS:
                raise RuntimeError("Bad labels shape {} for {}".format(tuple(labels.shape), name))
            labels = labels[:, :N_TASKS]  # ignore extra 15th column

            onset_edges = edges[:, edge_type == 0]
            edges2, edge_type2 = add_reverse_edges_from_edge_index(edges, edge_type)
            onset_idx = unique_onsets(onset_div)

            preds = model((x, edges2, edge_type2, onset_edges, onset_idx, None))

            # per-task accuracy
            for t_i, tname in enumerate(dataset_task_order):
                if tname not in preds:
                    continue
                pred_logits = preds[tname]
                target = labels[:, t_i].long().to(device)
                target_aligned = align_target_to_pred_length(target, pred_logits.shape[0], onset_idx)
                c, tot = masked_accuracy(pred_logits, target_aligned)
                correct_by_task[tname] += c
                total_by_task[tname] += tot

            # CSR-like exact match
            if all(k in preds for k in ("root", "quality", "inversion")):
                r_pred = preds["root"].argmax(dim=-1)
                q_pred = preds["quality"].argmax(dim=-1)
                i_pred = preds["inversion"].argmax(dim=-1)

                r_t = align_target_to_pred_length(labels[:, dataset_task_order.index("root")].long().to(device), r_pred.shape[0], onset_idx)
                q_t = align_target_to_pred_length(labels[:, dataset_task_order.index("quality")].long().to(device), q_pred.shape[0], onset_idx)
                i_t = align_target_to_pred_length(labels[:, dataset_task_order.index("inversion")].long().to(device), i_pred.shape[0], onset_idx)

                mask = (r_t >= 0) & (q_t >= 0) & (i_t >= 0)
                tot = int(mask.sum().item())
                if tot > 0:
                    corr = int(((r_pred == r_t) & (q_pred == q_t) & (i_pred == i_t) & mask).sum().item())
                    csr_correct += corr
                    csr_total += tot

            if (idx + 1) % 10 == 0:
                print("Processed {}/{} graphs...".format(idx + 1, len(loader)))

    banner("BASELINE RESULTS (Mozart validation)")
    for tname in dataset_task_order:
        tot = total_by_task[tname]
        if tot == 0:
            print("{:14s}: n/a".format(tname))
        else:
            acc = 100.0 * correct_by_task[tname] / tot
            print("{:14s}: {:7.2f}% ({}/{})".format(tname, acc, correct_by_task[tname], tot))

    if csr_total > 0:
        csr_acc = 100.0 * csr_correct / csr_total
        print("\nCSR-LIKE exact (root+quality+inversion): {:.2f}% ({}/{})".format(csr_acc, csr_correct, csr_total))


if __name__ == "__main__":
    main()
