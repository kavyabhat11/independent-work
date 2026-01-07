#!/usr/bin/env python3
"""Test that work grouping logic is correct."""

from split_mozart_data_FIXED import extract_work_id

# Test cases
test_cases = [
    ("K545-1.tsv", "K545"),
    ("K545-2.tsv", "K545"),
    ("K545-3.tsv", "K545"),
    ("sym100a-EXPO.tsv", "sym100a"),
    ("sym100a-DEV.tsv", "sym100a"),
    ("sym100a-RECAP.tsv", "sym100a"),
    ("sym100b-1.tsv", "sym100b"),
    ("op31n2-1.tsv", "op31n2"),
    ("op31n2-2.tsv", "op31n2"),
    ("K279-1.tsv", "K279"),
    ("K279-2.tsv", "K279"),
    ("K279-3.tsv", "K279"),
]

print("Testing work grouping logic...\n")

all_passed = True
for filename, expected_work_id in test_cases:
    result = extract_work_id(filename)
    status = "✓" if result == expected_work_id else "✗"
    if result != expected_work_id:
        all_passed = False
    print(f"{status} {filename:25s} → {result:15s} (expected: {expected_work_id})")

print("\n" + "="*60)
if all_passed:
    print("✓ ALL TESTS PASSED - Grouping logic is correct!")
else:
    print("✗ SOME TESTS FAILED - Check grouping logic")
print("="*60)
