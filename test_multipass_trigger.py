#!/usr/bin/env python3
"""
Test if multi-pass extraction is triggered when extract_by_role is set.
"""

import sys
sys.path.insert(0, '/home/lisabdunlap/StringSightNew')

from stringsight.extractors.openai import OpenAIExtractor

print("=" * 80)
print("TEST: Multi-Pass Extraction Trigger")
print("=" * 80)
print()

# Test 1: Without extract_by_role
print("[Test 1] Creating extractor WITHOUT extract_by_role")
extractor1 = OpenAIExtractor(model="gpt-4o-mini")
print(f"  extract_by_role = {extractor1.extract_by_role}")
print(f"  Should trigger multi-pass? {extractor1.extract_by_role is not None}")
print()

# Test 2: With extract_by_role
print("[Test 2] Creating extractor WITH extract_by_role=['user', 'assistant']")
extractor2 = OpenAIExtractor(
    model="gpt-4o-mini",
    extract_by_role=["user", "assistant"]
)
print(f"  extract_by_role = {extractor2.extract_by_role}")
print(f"  Should trigger multi-pass? {extractor2.extract_by_role is not None}")
print()

# Verdict
print("=" * 80)
if extractor1.extract_by_role is None and extractor2.extract_by_role == ["user", "assistant"]:
    print("✅ SUCCESS: extract_by_role parameter is being stored correctly")
    print()
    print("This means if multi-pass extraction isn't running, the issue is:")
    print("  1. extract_by_role is not being passed from the API endpoint")
    print("  2. OR there's an issue in the async run() method")
else:
    print("❌ FAIL: extract_by_role parameter is NOT being stored correctly")
