# Mozart Finetuning Fixes - Complete Explanation

## The Problem

Your finetuning achieved only **15.53% accuracy** with bizarre predictions like "It6/vii" appearing hundreds of times. This was caused by two critical bugs:

### Bug #1: Vocabulary Mismatch (31 vs 76 classes)

**What happened:**
- Your codebase has TWO different Roman numeral vocabularies:
  - **RomanNumeral31**: 31 classes (basic chords only)
  - **RomanNumeral76**: 76 classes (includes pre-encoded secondary dominants)

- Which vocab is used depends on `DATA_VERSION`:
  ```python
  # In chordgnn/data/datamodules/mix_vs.py:
  if version == "v1.0.0":
      data_source = AugmentedNetChordGraphDataset(...)  # RomanNumeral76
  else:
      data_source = Augmented2022ChordGraphDataset(...)  # RomanNumeral31
  ```

- Your script used `DATA_VERSION = "v1.0.0"` (76 classes)
- But your pretrained model likely uses 31 classes
- Result: Shape mismatch when loading weights
  - Pretrained head: `[256, 31]`
  - New model head: `[256, 76]`
  - The romanNumeral head **stayed randomly initialized**!

### Bug #2: Learning Rate Too High

**What happened:**
- You used `LR = 5e-4` (same as training from scratch)
- For finetuning, this is **50x too high**
- Caused **catastrophic forgetting** - overwrote pretrained knowledge with noise
- Should use `LR = 1e-5` to `1e-4` for finetuning

---

## The Solution

The fixed script (`finetune_from_checkpoint_FIXED.py`) implements these critical changes:

### Fix 1: Auto-Download Pretrained Model
```python
WANDB_ARTIFACT = "melkisedeath/chord_rec/model-kvd0jic5:v0"

# Automatically downloads from W&B if not found locally
if os.path.exists(expected_ckpt_path):
    PRETRAINED_CKPT = expected_ckpt_path
else:
    api = wandb.Api()
    artifact = api.artifact(WANDB_ARTIFACT, type="model")
    downloaded_dir = artifact.download(root=ARTIFACT_ROOT)
    PRETRAINED_CKPT = find_ckpt_in(downloaded_dir)
```

### Fix 2: Load Checkpoint FIRST to Detect Vocab
```python
# Load checkpoint before creating datamodule
ckpt = torch.load(PRETRAINED_CKPT, map_location="cpu")
pretrained_tasks = ckpt["hyper_parameters"]["tasks"]

# Check romanNumeral vocab size
rn_vocab_size = pretrained_tasks["romanNumeral"]
```

### Fix 3: Auto-Select Correct DATA_VERSION
```python
if rn_vocab_size == 31:
    DATA_VERSION = "v2.0.0"  # or any string != "v1.0.0"
    # → Loads Augmented2022ChordGraphDataset (31 classes)
elif rn_vocab_size == 76:
    DATA_VERSION = "v1.0.0"
    # → Loads AugmentedNetChordGraphDataset (76 classes)
```

**Note**: "v2.0.0" is just an example - could be ANY string except "v1.0.0". The code only checks `if version == "v1.0.0"`, so anything else triggers the 31-class dataset.

### Fix 4: Use Pretrained Tasks (NOT Datamodule Tasks)
```python
# CRITICAL: Use pretrained tasks to build model
tasks = pretrained_tasks  # NOT datamodule.tasks!

model = st.models.chord.ChordPrediction(
    tasks=tasks,  # Ensures exact architecture match
    ...
)
```

This ensures all vocab sizes match exactly, so all weights load successfully.

### Fix 5: Lower Learning Rate (50x Smaller)
```python
LR = 1e-5         # Was 5e-4 (too high!)
WEIGHT_DECAY = 1e-4  # Was 3.5e-3
```

### Fix 6: Match All Architecture Params
```python
n_hidden = pretrained_hparams["n_hidden"]
n_layers = pretrained_hparams["n_layers"]  # Don't use defaults!
```

### Fix 7: Verify Heads Loaded Successfully
```python
# Check which task heads loaded from pretrained
important_heads = ["romanNumeral", "localkey", "tonkey", "pcset", "bass"]
for head in important_heads:
    if head in loaded_weights:
        print(f"✓ {head} head loaded from pretrained")
```

You should see "✓ romanNumeral head loaded from pretrained" - if not, there's still a mismatch!

---

## How to Use

1. **Make sure you have W&B access to the artifact**:
   ```bash
   wandb login
   ```

2. **Run the fixed script**:
   ```bash
   conda activate chordgnn
   python finetune_from_checkpoint_FIXED.py
   ```

3. **Watch for these key outputs**:
   ```
   STEP 2: Inspecting pretrained checkpoint to detect vocab
   ✓ Pretrained uses romanNumeral: 31 classes
     → Setting DATA_VERSION = 'v2.0.0'

   STEP 4: Loading Mozart datamodule
   ✓ All task vocabs match!

   STEP 7: Loading pretrained weights
   ✓ Task heads loaded successfully: romanNumeral, localkey, tonkey, pcset, bass
   ```

4. **Expected results**:
   - Training converges smoothly (no wild loss spikes)
   - Validation accuracy **60-80%+** (not 15%!)
   - Predictions are sensible: I, V, IV, ii, etc. (not "It6/vii")

---

## Understanding the Vocab Versions

### Why Two Different Vocabularies Exist

**RomanNumeral76** (chord_representations.py):
- 76 classes including: `'I', 'V7', 'V', 'ii', 'V7/V', 'viio7/IV', 'V/IV'`, etc.
- Secondary dominants are **pre-encoded** as separate classes
- Example: "V7/V" is a single class

**RomanNumeral31** (chord_representations_latest.py):
- 31 classes: `'I', 'IV', 'V', 'V7', 'ii', 'viio7', 'It', 'Fr7', 'Ger7'`, etc.
- Only basic chords
- Secondary dominants handled via **tonicization** (separate tonkey field)
- Example: "V7/V" = romanNumeral="V7" + tonkey="V"

### Which Version is Better?

**RomanNumeral31** (newer approach):
- ✓ Smaller vocab = easier to learn
- ✓ More compositional (separates chord identity from tonicization)
- ✓ Better generalization to unseen tonicizations
- Recommended for most use cases

**RomanNumeral76** (older approach):
- ✓ Explicit representation of secondary dominants
- More classes = harder to learn but might capture subtle distinctions
- Legacy compatibility

---

## What Was Happening Before vs Now

### Before (BROKEN):
1. Pretrained model: `romanNumeral: 31`
2. Your script: `DATA_VERSION = "v1.0.0"` (hardcoded)
3. Datamodule: `romanNumeral: 76`
4. Model built with 76-class head
5. Load pretrained:
   - Encoder: ✓ loaded
   - romanNumeral head: ✗ **shape mismatch** → stayed random
6. Training with LR=5e-4:
   - Random head + high LR = catastrophic forgetting
7. Result: 15% accuracy, nonsense predictions

### After (FIXED):
1. Load pretrained checkpoint first
2. Detect: `romanNumeral: 31`
3. Auto-set: `DATA_VERSION = "v2.0.0"`
4. Datamodule: `romanNumeral: 31`
5. Model built with 31-class head (matches pretrained!)
6. Load pretrained:
   - Encoder: ✓ loaded
   - romanNumeral head: ✓ **loaded** (shapes match!)
7. Training with LR=1e-5:
   - Pretrained head + low LR = smooth finetuning
8. Expected: 70%+ accuracy, sensible predictions

---

## Troubleshooting

### If you still see low accuracy:

1. **Check the vocab match**:
   ```
   STEP 4: Loading Mozart datamodule
   romanNumeral: 31 (pretrained: 31) ✓
   ```
   Should all be ✓ with matching numbers

2. **Check heads loaded**:
   ```
   STEP 7: Loading pretrained weights
   ✓ Task heads loaded successfully: romanNumeral, localkey, tonkey, pcset, bass
   ```
   If romanNumeral is missing, vocab still doesn't match!

3. **Check W&B artifact path**:
   Make sure `melkisedeath/chord_rec/model-kvd0jic5:v0` is the correct path

4. **Check Mozart data quality**:
   - Are the TSV files properly formatted?
   - Do they have the correct fields?

---

## Summary of Key Files

- **`finetune_from_checkpoint_FIXED.py`**: The corrected finetuning script
- **`finetune_from_checkpoint.py`**: Your original (broken) script
- **`check_pretrained_vocab.py`**: Diagnostic script to inspect pretrained model

Use the FIXED version for all future finetuning!
