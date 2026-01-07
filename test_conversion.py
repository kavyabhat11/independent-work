#!/usr/bin/env python3
"""
Quick test to verify the conversion is working correctly.
Tests key functions with known examples from Tymoczko data.
"""

import sys
sys.path.insert(0, '.')

from convert_mozart_to_tsv_FIXED import (
    simplify_roman_numeral_to_31class,
    compute_tonicized_key,
    roman_numeral_to_scale_degree,
    extract_voices_from_midi,
    duration_to_harmonic_rhythm_category,
)

def test_simplify_roman_numeral():
    """Test Roman numeral simplification to 31-class vocab."""
    print("=" * 60)
    print("TEST 1: Roman Numeral Simplification")
    print("=" * 60)

    tests = [
        ("V6/5/V", ("V", "V")),
        ("V4/3", ("V", None)),
        ("viio6", ("viio", None)),
        ("I6/4", ("I", None)),
        ("V7/ii", ("V7", "ii")),
        ("Fr4/3", ("Fr7", None)),
        ("It6/V", ("It", "V")),
        ("ii/o2", ("iio", None)),
    ]

    passed = 0
    for input_rn, expected in tests:
        result = simplify_roman_numeral_to_31class(input_rn)
        status = "✓" if result == expected else "✗"
        if result == expected:
            passed += 1
        print(f"{status} {input_rn:15s} -> {result} (expected {expected})")

    print(f"\nPassed: {passed}/{len(tests)}\n")
    return passed == len(tests)


def test_tonicized_key():
    """Test tonicized key computation."""
    print("=" * 60)
    print("TEST 2: Tonicized Key Computation")
    print("=" * 60)

    tests = [
        ("V", "C", "G"),      # V of C = G major
        ("ii", "C", "d"),     # ii of C = d minor
        ("IV", "G", "C"),     # IV of G = C major
        ("vi", "F", "d"),     # vi of F = d minor
        ("V", "d", "A"),      # V of d = A major (dominant in minor is major!)
    ]

    passed = 0
    for tonkey, local_key, expected in tests:
        result = compute_tonicized_key(tonkey, local_key)
        status = "✓" if result == expected else "✗"
        if result == expected:
            passed += 1
        print(f"{status} {tonkey:5s} in {local_key:2s} -> {result:3s} (expected {expected})")

    print(f"\nPassed: {passed}/{len(tests)}\n")
    return passed == len(tests)


def test_scale_degree():
    """Test Roman numeral to scale degree conversion."""
    print("=" * 60)
    print("TEST 3: Scale Degree Extraction")
    print("=" * 60)

    tests = [
        ("I", 1),
        ("ii", 2),
        ("V7", 5),
        ("viio", 7),
        ("N", 2),  # Neapolitan
    ]

    passed = 0
    for rn, expected in tests:
        result = roman_numeral_to_scale_degree(rn)
        status = "✓" if result == expected else "✗"
        if result == expected:
            passed += 1
        print(f"{status} {rn:5s} -> {result} (expected {expected})")

    print(f"\nPassed: {passed}/{len(tests)}\n")
    return passed == len(tests)


def test_voice_extraction():
    """Test SATB voice extraction from MIDI pitches."""
    print("=" * 60)
    print("TEST 4: Voice Leading Extraction")
    print("=" * 60)

    tests = [
        ([60], ("C", "C", "C", "C")),                      # 1 note
        ([60, 67], ("G", "G", "C", "C")),                  # 2 notes (S/A=high, T/B=low)
        ([60, 64, 67], ("G", "E", "E", "C")),              # 3 notes
        ([60, 64, 67, 72], ("C", "G", "E", "C")),          # 4 notes (S=high, A=2nd high, T=2nd low, B=low)
    ]

    passed = 0
    for midi_pitches, expected in tests:
        result = extract_voices_from_midi(midi_pitches)
        status = "✓" if result == expected else "✗"
        if result == expected:
            passed += 1
        print(f"{status} MIDI {midi_pitches} -> S={result[0]} A={result[1]} T={result[2]} B={result[3]}")
        if result != expected:
            print(f"   Expected: S={expected[0]} A={expected[1]} T={expected[2]} B={expected[3]}")

    print(f"\nPassed: {passed}/{len(tests)}\n")
    return passed == len(tests)


def test_harmonic_rhythm():
    """Test harmonic rhythm duration categorization."""
    print("=" * 60)
    print("TEST 5: Harmonic Rhythm Categories")
    print("=" * 60)

    tests = [
        (0.1, 0),    # onset
        (0.25, 1),   # thirty-second
        (0.5, 2),    # sixteenth
        (1.0, 3),    # eighth
        (2.0, 4),    # quarter
        (4.0, 5),    # half
        (8.0, 6),    # whole
    ]

    passed = 0
    for duration, expected in tests:
        result = duration_to_harmonic_rhythm_category(duration)
        status = "✓" if result == expected else "✗"
        if result == expected:
            passed += 1
        print(f"{status} {duration:4.1f} beats -> category {result} (expected {expected})")

    print(f"\nPassed: {passed}/{len(tests)}\n")
    return passed == len(tests)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TESTING MOZART TSV CONVERSION (FIXED VERSION)")
    print("=" * 60 + "\n")

    all_passed = True
    all_passed &= test_simplify_roman_numeral()
    all_passed &= test_tonicized_key()
    all_passed &= test_scale_degree()
    all_passed &= test_voice_extraction()
    all_passed &= test_harmonic_rhythm()

    print("=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("The conversion script is working correctly!")
    else:
        print("✗ SOME TESTS FAILED")
        print("Review the failures above before running conversion.")
    print("=" * 60 + "\n")

    sys.exit(0 if all_passed else 1)
