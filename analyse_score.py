#!/usr/bin/env python3
"""
analyse_score.py (FOOLPROOF VERSION)

Goal:
- Run ChordGNN inference on a MusicXML score using a W&B *model artifact* checkpoint
- Decode outputs using the SAME decoder registry used by the original codebase
  (chordgnn.utils.chord_representations, NOT *_latest*)
- Write predicted Roman Numerals into a new MusicXML part: <score>-analysis.musicxml
- Never crash mid-batch: handle None/invalid pcset, missing keys, decoder mismatch, etc.

Usage:
  python analyse_score.py \
    --score_path data/mozart_test/K279-3.xml \
    --use_ckpt kb9520-princeton-university/chord_rec/mozart-finetuned-model:v1

Notes:
- This script DOES NOT assume PostChordPrediction. It loads your finetuned ChordPrediction ckpt
  and uses the underlying .module.predict(score) method (the one that actually exists).
- If your checkpoint task head sizes don't match the decoder vocab sizes, it will error with a
  clear message showing exactly which rep it tried and what sizes were found.
"""

import argparse
import ast
import glob
import os
import re
from typing import Any, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import partitura as pt
import torch

from chordgnn.models.chord import ChordPrediction
import chordgnn.utils.chord_representations as cr  # IMPORTANT: not *_latest*
from chordgnn.utils.chord_representations import resolveRomanNumeralCosine, formatRomanNumeral


# ----------------------------
# Helpers
# ----------------------------
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def find_ckpt_in_dir(d: str) -> str:
    ckpts = glob.glob(os.path.join(d, "*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt found in: {d}")
    return max(ckpts, key=os.path.getmtime)


def download_wandb_artifact(artifact_name: str, root: str = "./artifacts") -> str:
    """
    Downloads W&B artifact if needed and returns local directory.
    Uses the artifact string like:
      entity/project/name:version
    """
    ensure_dir(root)
    # We make a deterministic directory name under root by using the basename.
    # W&B itself will create something like <root>/<name:version>/...
    import wandb
    api = wandb.Api()
    artifact = api.artifact(artifact_name, type="model")
    return artifact.download(root=root)


def normalize_pcset(pcset: Any) -> Tuple[int, ...]:
    """
    Convert pcset into tuple[int], never None, never crash.
    Accepts:
      - None -> ()
      - tuple/list/np.ndarray -> tuple(ints)
      - string "(0, 4, 7)" or "[0,4,7]" -> parsed
      - anything else -> ()
    """
    if pcset is None:
        return tuple()

    if isinstance(pcset, (tuple, list, np.ndarray)):
        try:
            return tuple(int(x) for x in pcset)
        except Exception:
            return tuple()

    if isinstance(pcset, str):
        s = pcset.strip()
        if s == "" or s.lower() == "none":
            return tuple()
        # try python-literal parse first
        try:
            val = ast.literal_eval(s)
            if isinstance(val, (tuple, list)):
                return tuple(int(x) for x in val)
        except Exception:
            pass
        # fallback: pull ints from string
        nums = re.findall(r"-?\d+", s)
        if nums:
            try:
                return tuple(int(x) for x in nums)
            except Exception:
                return tuple()

    return tuple()


def get_rep_registry() -> Dict[str, Any]:
    """
    Locate the representation registry in chord_representations.py.
    """
    for name in ["available_representations", "AVAILABLE_REPRESENTATIONS", "representations", "REPRESENTATIONS"]:
        if hasattr(cr, name):
            reg = getattr(cr, name)
            if isinstance(reg, dict):
                return reg
    raise RuntimeError(
        "Could not find a dict registry of representations in chordgnn.utils.chord_representations "
        "(expected available_representations / REPRESENTATIONS / etc.)"
    )


def choose_rep_for_task(reg: Dict[str, Any], task: str, head_size: int) -> Tuple[str, Any]:
    """
    Choose a decoder representation with vocab size == head_size.

    Strategy:
      1) exact key match (task) if size matches
      2) keys containing task substring if size matches
      3) any representation with matching size

    If no match, raise a clear error listing candidates.
    """
    def rep_vocab(rep_obj: Any) -> Optional[int]:
        if not hasattr(rep_obj, "classList"):
            return None
        try:
            return len(rep_obj.classList)
        except Exception:
            return None

    # 1) exact key
    if task in reg:
        v = rep_vocab(reg[task])
        if v == head_size:
            return task, reg[task]

    # 2) substring matches
    cands = []
    for k, rep in reg.items():
        v = rep_vocab(rep)
        if v is None:
            continue
        if v != head_size:
            continue
        if task.lower() in k.lower():
            cands.append((k, rep))
    if cands:
        cands.sort(key=lambda x: (len(x[0]), x[0]))
        return cands[0]

    # 3) any size match
    any_match = []
    for k, rep in reg.items():
        v = rep_vocab(rep)
        if v == head_size:
            any_match.append((k, rep))
    if any_match:
        any_match.sort(key=lambda x: (len(x[0]), x[0]))
        return any_match[0]

    # error with helpful info
    romanish = [k for k in reg.keys() if "roman" in k.lower()]
    raise RuntimeError(
        f"No decoder representation found with vocab size {head_size} for task '{task}'.\n"
        f"Registry keys containing 'roman': {romanish}\n"
        f"(This usually means you’re importing the wrong representation module or your ckpt uses a rep not registered.)"
    )


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Convert ckpt keys like 'module.encoder.xxx' -> 'encoder.xxx' for loading into model.module
    """
    out = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            out[k[len("module."):]] = v
        else:
            out[k] = v
    return out


def safe_getattr(obj, name, default=None):
    return getattr(obj, name, default)


# ----------------------------
# Args
# ----------------------------
parser = argparse.ArgumentParser("Chord Prediction")
parser.add_argument("--use_ckpt", type=str, required=True, help="W&B model artifact (entity/project/name:version)")
parser.add_argument("--score_path", type=str, required=True, help="Path to MusicXML input score")
args = parser.parse_args()

ARTIFACT_ROOT = "./artifacts"


# ----------------------------
# Resolve artifact directory + ckpt file
# ----------------------------
artifact_basename = os.path.basename(args.use_ckpt)  # e.g. mozart-finetuned-model:v1
local_art_dir = os.path.normpath(os.path.join(ARTIFACT_ROOT, artifact_basename))

ckpt_path = None
if os.path.isdir(local_art_dir):
    try:
        ckpt_path = find_ckpt_in_dir(local_art_dir)
    except FileNotFoundError:
        ckpt_path = None

if ckpt_path is None:
    print(f"[W&B] Downloading artifact: {args.use_ckpt}")
    downloaded_dir = download_wandb_artifact(args.use_ckpt, root=ARTIFACT_ROOT)
    ckpt_path = find_ckpt_in_dir(downloaded_dir)
    local_art_dir = downloaded_dir

print("Artifact dir:", local_art_dir)
print("Loading checkpoint:", ckpt_path)


# ----------------------------
# Load checkpoint + construct model exactly from hparams
# ----------------------------
ckpt = torch.load(ckpt_path, map_location="cpu")
hparams = ckpt.get("hyper_parameters", {})
if "tasks" not in hparams:
    raise RuntimeError("Checkpoint missing hyper_parameters['tasks']; can’t construct matching model.")

tasks = hparams["tasks"]
in_feats = int(hparams.get("in_feats", 83))
n_hidden = int(hparams.get("n_hidden", 256))
n_layers = int(hparams.get("n_layers", 1))
dropout = float(hparams.get("dropout", 0.0))
use_nade = bool(hparams.get("use_nade", False))
use_jk = bool(hparams.get("use_jk", False))
use_rotograd = bool(hparams.get("use_rotograd", False))

print("Checkpoint hyper_parameters:")
print("  in_feats:", in_feats)
print("  n_hidden:", n_hidden)
print("  n_layers:", n_layers)
print("  dropout:", dropout)
print("  tasks:", tasks)
print()

# Important: build ChordPrediction then use underlying .module (the one that has .predict)
wrapper = ChordPrediction(
    in_feats=in_feats,
    n_hidden=n_hidden,
    tasks=tasks,
    n_layers=n_layers,
    lr=0.0,
    dropout=dropout,
    weight_decay=0.0,
    use_nade=use_nade,
    use_jk=use_jk,
    use_rotograd=use_rotograd,
    device="cpu",
)

model = wrapper.module  # this is the one with predict(score) in your repo


# ----------------------------
# Load weights (robustly)
# ----------------------------
state_dict = ckpt.get("state_dict", ckpt)
state_dict = strip_module_prefix(state_dict)

# Drop loss params if present; they often mismatch when num_tasks changes
for k in list(state_dict.keys()):
    if k.startswith("train_loss.") or k.startswith("val_loss.") or k.startswith("test_loss."):
        state_dict.pop(k, None)

missing, unexpected = model.load_state_dict(state_dict, strict=False)

print(f"Loaded state_dict into model.module")
print(f"  Missing keys: {len(missing)}")
print(f"  Unexpected keys: {len(unexpected)}")
if len(unexpected) > 0:
    print("  (Unexpected example keys):", unexpected[:10])
print()


# ----------------------------
# Predict
# ----------------------------
score = pt.load_score(args.score_path)

if not hasattr(model, "predict"):
    raise RuntimeError(
        "Your model.module does not have .predict(score). "
        "That means your repo version differs from the one we assumed."
    )

with torch.no_grad():
    model.eval()
    prediction = model.predict(score)

if not isinstance(prediction, dict):
    raise RuntimeError(f"model.predict(score) returned {type(prediction)}; expected dict.")

# Must include these for RN overlay; if absent, we will compute fallbacks
onset = prediction.get("onset", None)
s_measure = prediction.get("s_measure", None)


# ----------------------------
# Decode ONLY what we need to generate RomanText
# ----------------------------
reg = get_rep_registry()

NEEDED = ["localkey", "tonkey", "romanNumeral", "bass", "pcset"]
dfdict: Dict[str, Any] = {}

rep_used: Dict[str, str] = {}

for task in NEEDED:
    if task not in tasks:
        raise RuntimeError(f"Checkpoint tasks do not include '{task}'. Present: {list(tasks.keys())}")
    if task not in prediction:
        raise RuntimeError(f"Prediction outputs do not include '{task}'. Present: {list(prediction.keys())}")

    head_size = int(tasks[task])
    rep_name, rep = choose_rep_for_task(reg, task, head_size)
    rep_used[task] = rep_name

    logits = prediction[task]
    if not torch.is_tensor(logits):
        raise RuntimeError(f"prediction['{task}'] is {type(logits)}, expected torch.Tensor")

    pred_idx = torch.argmax(logits, dim=-1).reshape(-1, 1)

    vocab = len(rep.classList)
    max_idx = int(pred_idx.max().item()) if pred_idx.numel() else -1
    if max_idx >= vocab:
        raise RuntimeError(
            f"Decoder mismatch for task '{task}': predicted index {max_idx}, vocab size is {vocab}. "
            f"(ckpt head size for '{task}' is {head_size}, rep='{rep_name}')"
        )

    decoded = rep.decode(pred_idx)

    # normalize pcset right away to avoid NoneType crash later
    if task == "pcset":
        decoded = [normalize_pcset(x) for x in decoded]

    dfdict[task] = decoded

print("Decoder representations used:")
for k, v in rep_used.items():
    print(f"  {k:12s} -> {v}")
print()


# Add onset + measure columns
# If model didn’t provide onset/s_measure, synthesize something stable from note_array
note_array = score.note_array(include_pitch_spelling=True)

if onset is None:
    # fallback: unique onsets in beats, sorted
    if "onset_beat" in note_array.dtype.names:
        onset_vals = np.unique(note_array["onset_beat"])
        onset_vals.sort()
        onset = torch.tensor(onset_vals, dtype=torch.float32)
    else:
        raise RuntimeError("No 'onset' in prediction and no 'onset_beat' in note_array to synthesize onsets.")

if s_measure is None:
    # fallback: try measure number if exists; else zeros
    if "measure" in note_array.dtype.names:
        # crude: broadcast zeros if mismatch
        s_measure = torch.zeros_like(onset)
    else:
        s_measure = torch.zeros_like(onset)

dfdict["onset"] = onset.detach().cpu().numpy().tolist() if torch.is_tensor(onset) else list(onset)
dfdict["s_measure"] = s_measure.detach().cpu().numpy().tolist() if torch.is_tensor(s_measure) else list(s_measure)

df = pd.DataFrame(dfdict)


# ----------------------------
# Build RNA part and save MusicXML
# ----------------------------
prevkey = ""

# Choose bass part as last part like your original script
bass_part = score.parts[-1]

# Build RN part
rn_part = pt.score.Part(
    id="RNA",
    part_name="Roman Numerals",
    quarter_duration=bass_part._quarter_durations[0],
)
rn_part.add(pt.score.Clef(staff=1, sign="percussion", line=2, octave_change=0), 0)
rn_part.add(pt.score.Staff(number=1, lines=1), 0)

annotations = []

# Iterate in order
for row in df.itertuples(index=False):
    # Pull fields safely
    thiskey = getattr(row, "localkey", None) or "C"
    tonicizedKey = getattr(row, "tonkey", None) or thiskey
    numerator = getattr(row, "romanNumeral", None) or "I"
    bass_label = getattr(row, "bass", None) or "C"

    pcs = normalize_pcset(getattr(row, "pcset", None))

    # If pcs is empty, we can still attempt resolveRomanNumeralCosine,
    # but it might not be meaningful; we’ll just keep the numerator.
    try:
        if len(pcs) == 0:
            rn2 = numerator
        else:
            # We don't have SATB, so use bass as fallback for other voices.
            rn2, _ = resolveRomanNumeralCosine(
                bass_label, bass_label, bass_label, bass_label,
                pcs, thiskey, numerator, tonicizedKey
            )
    except Exception:
        rn2 = numerator  # never crash batch analysis

    if thiskey != prevkey:
        rn2fig = f"{thiskey}:{rn2}"
        prevkey = thiskey
    else:
        rn2fig = rn2

    try:
        formatted_RN = formatRomanNumeral(rn2fig, thiskey)
    except Exception:
        formatted_RN = rn2fig

    # Convert onset beat -> onset_div using bass_part mapping
    onset_beat = getattr(row, "onset", None)
    if onset_beat is None:
        continue

    try:
        onset_div = int(bass_part.inv_beat_map(float(onset_beat)).item())
    except Exception:
        continue

    annotations.append((formatted_RN, onset_div))

if not annotations:
    raise RuntimeError("No annotations produced. (Check that prediction['onset'] aligns with score beat mapping.)")

annotations = np.array(annotations, dtype=[("rn", "U120"), ("onset_div", "i4")])

# De-duplicate consecutive identical RN labels
end_onset = note_array["onset_div"].max()
end_duration = note_array[note_array["onset_div"] == end_onset]["duration_div"].max()
keep = np.array([True] + [annotations[i]["rn"] != annotations[i - 1]["rn"] for i in range(1, len(annotations))], dtype=bool)
annotations = annotations[keep]

durations = np.r_[np.diff(annotations["onset_div"]), end_duration]

for i, (rn, onset_div) in enumerate(annotations):
    note = pt.score.UnpitchedNote(step="F", octave=5, staff=1)
    word = pt.score.RomanNumeral(rn)
    rn_part.add(note, int(onset_div), int(onset_div + durations[i].item()))
    rn_part.add(word, int(onset_div))

# Copy time signatures + measures for readability
for item in bass_part.iter_all(pt.score.TimeSignature):
    rn_part.add(item, item.start.t)
for item in bass_part.measures:
    rn_part.add(item, item.start.t, item.end.t)

pt.score.tie_notes(rn_part)
score.parts.append(rn_part)

out_path = f"{os.path.splitext(args.score_path)[0]}-analysis.musicxml"
pt.save_musicxml(score, out_path)

print("Wrote:", out_path)
print("Done.")
