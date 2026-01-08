#!/usr/bin/env python3
"""
Debug script to test resolveRomanNumeralCosine function
"""

import sys
import os

# Add the directory to sys.path to avoid import issues
sys.path.insert(0, os.path.dirname(__file__))

# Direct import to avoid module-level dependencies
import importlib.util
spec = importlib.util.spec_from_file_location(
    "chord_representations",
    "chordgnn/utils/chord_representations.py"
)
chord_rep = importlib.util.module_from_spec(spec)
# Set it in sys.modules so relative imports work
sys.modules['chord_representations'] = chord_rep
spec.loader.exec_module(chord_rep)

resolveRomanNumeralCosine = chord_rep.resolveRomanNumeralCosine
COMMON_ROMAN_NUMERALS = chord_rep.COMMON_ROMAN_NUMERALS
KEYS = chord_rep.KEYS

# Test case 1: Simple I chord in root position
# Ground truth: I (root position)
# Prediction: bass=F, tenor=A, alto=C, soprano=F, key=F
print("=" * 70)
print("Test 1: I chord in F major (root position)")
print("=" * 70)

b = "F"  # bass
t = "A"  # tenor
a = "C"  # alto
s = "F"  # soprano
pcs = []  # pitch class set (empty for now)
key = "F"  # local key
numerator = "I"  # predicted romanNumeral
tonicizedKey = "F"  # tonicized key (same as local key)

try:
    result = resolveRomanNumeralCosine(b, t, a, s, pcs, key, numerator, tonicizedKey)
    resolved_figure, chord_label = result
    print(f"Input: bass={b}, tenor={t}, alto={a}, soprano={s}")
    print(f"Key: {key}, numerator: {numerator}")
    print(f"Resolved figure: {resolved_figure}")
    print(f"Chord label: {chord_label}")
    print()
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    print()

# Test case 2: I6 chord (first inversion)
# Ground truth: I6
# Prediction: bass=A, tenor=C, alto=F, soprano=A
print("=" * 70)
print("Test 2: I6 chord in F major (first inversion)")
print("=" * 70)

b = "A"  # bass (3rd in bass = first inversion)
t = "C"
a = "F"
s = "A"
pcs = []
key = "F"
numerator = "I"
tonicizedKey = "F"

try:
    result = resolveRomanNumeralCosine(b, t, a, s, pcs, key, numerator, tonicizedKey)
    resolved_figure, chord_label = result
    print(f"Input: bass={b}, tenor={t}, alto={a}, soprano={s}")
    print(f"Key: {key}, numerator: {numerator}")
    print(f"Resolved figure: {resolved_figure}")
    print(f"Chord label: {chord_label}")
    print(f"Expected: I6")
    print(f"Match: {resolved_figure == 'I6'}")
    print()
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    print()

# Test case 3: V7 chord (root position)
print("=" * 70)
print("Test 3: V7 chord in F major (root position)")
print("=" * 70)

b = "C"  # bass (root of V7 in F major)
t = "E"
a = "G"
s = "B-"  # B-flat
pcs = []
key = "F"
numerator = "V7"
tonicizedKey = "F"

try:
    result = resolveRomanNumeralCosine(b, t, a, s, pcs, key, numerator, tonicizedKey)
    resolved_figure, chord_label = result
    print(f"Input: bass={b}, tenor={t}, alto={a}, soprano={s}")
    print(f"Key: {key}, numerator: {numerator}")
    print(f"Resolved figure: {resolved_figure}")
    print(f"Chord label: {chord_label}")
    print(f"Expected: V7")
    print(f"Match: {resolved_figure == 'V7'}")
    print()
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    print()

# Test case 4: Check what keys are available
print("=" * 70)
print("Available KEYS vocabulary (first 20):")
print("=" * 70)
print(KEYS[:20])
print()

print("=" * 70)
print("Available ROMAN_NUMERALS vocabulary (first 20):")
print("=" * 70)
print(COMMON_ROMAN_NUMERALS[:20])
print()
