MOZART FINETUNING DATASET
=========================

Created: 2026-01-07
Source: 104 Mozart/Beethoven/Haydn piano sonata movements (Tymoczko annotations)
Validation: ALL fields confirmed to be in ChordGNN vocabularies

DATASET STATISTICS
------------------
Total: 39 works → 104 TSV files

Training:   29 works →  76 files
Validation:  5 works →  15 files
Test:        5 works →  13 files

IMPORTANT: Split by WORK, not by file
- All movements from the same work stay in the same split
- Prevents data leakage between train/val/test sets

VOCABULARIES
------------
All TSV columns are validated against ChordGNN's vocabularies:
- Roman numerals: 31 classes (I, ii, V7, viio, etc.)
- PCset: 121 classes (computed from music21 RomanNumeral, not actual notes)
- Keys: 38 classes (C, d, A-, etc.)
- Voice leading: SATB extracted from actual MIDI pitches
- Harmonic rhythm: 7 categories (computed from annotation durations)
- Secondary dominants: degree2 properly encoded

FIXED ISSUES
------------
✓ Voice leading uses actual notes (not fake/all same)
✓ Secondary dominants properly computed (tonicizedKey, degree2)
✓ Harmonic rhythm from annotation durations (not hardcoded 0)
✓ PCset from music21 realization (guaranteed in vocabulary)
✓ All invalid Roman numerals normalized to 31-class vocab
✓ All keys validated (no empty strings or '-')

REPRODUCIBILITY
---------------
See split_manifest.json for exact file assignments
Split is deterministic - re-running the script produces the same split

USAGE ON ADROIT
---------------
1. Upload the entire mozart_dataset_final/ directory to Adroit
2. Set MOZART_ROOT=./mozart_dataset_final in your training script
3. Run finetuning - all vocab errors should be resolved
