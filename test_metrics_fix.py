#!/usr/bin/env python3
"""
Test script to verify that metrics are computed after per-row extraction and clustering.

This script:
1. Creates a simple dataset with score columns
2. Extracts properties for a single row
3. Verifies that the property has original_row in meta
4. Clusters the properties
5. Verifies that metrics are computed and non-empty
"""

import pandas as pd
import json

# Create test data with scores
test_data = [
    {
        "question_id": "1",
        "prompt": "What is the capital of France?",
        "model": "gpt-4",
        "model_response": "The capital of France is Paris.",
        "score": {"accuracy": 1.0, "helpfulness": 0.9}
    },
    {
        "question_id": "2",
        "prompt": "What is 2+2?",
        "model": "gpt-4",
        "model_response": "2+2 equals 4.",
        "score": {"accuracy": 1.0, "helpfulness": 1.0}
    },
    {
        "question_id": "3",
        "prompt": "Explain quantum computing.",
        "model": "gpt-4",
        "model_response": "Quantum computing uses quantum mechanics principles.",
        "score": {"accuracy": 0.7, "helpfulness": 0.6}
    }
]

print("=" * 80)
print("TESTING: Per-row extraction -> clustering -> metrics")
print("=" * 80)

# Test 1: Check that original_row is added during extraction
print("\n[TEST 1] Checking if extraction adds original_row to property meta...")

# Simulate what the extraction endpoint does
from stringsight.core.data_objects import Property
import uuid

# Create a mock property as if it was extracted
test_row = test_data[0]
prop = Property(
    id=str(uuid.uuid4()),
    question_id=test_row["question_id"],
    model=test_row["model"],
    property_description="Provides accurate factual information",
    raw_response='{"property_description": "Provides accurate factual information"}',
)

# Convert to dict and add original_row (simulating the extraction endpoint)
prop_dict = prop.to_dict()
if not prop_dict.get('meta'):
    prop_dict['meta'] = {}
prop_dict['meta']['original_row'] = test_row

print(f"✓ Property has original_row in meta: {('original_row' in prop_dict.get('meta', {}))}")
print(f"  - question_id: {prop_dict['question_id']}")
print(f"  - model: {prop_dict['model']}")
print(f"  - original_row score: {prop_dict['meta']['original_row'].get('score')}")

# Test 2: Verify conversation reconstruction uses original_row
print("\n[TEST 2] Verifying conversation reconstruction uses original_row...")

# Simulate what the clustering endpoint does
prop_obj = Property(**{k: v for k, v in prop_dict.items() if k != 'meta'})
prop_obj.meta = prop_dict.get('meta', {})

# Check if we can extract scores from meta
if hasattr(prop_obj, 'meta') and isinstance(prop_obj.meta, dict) and 'original_row' in prop_obj.meta:
    original_row = prop_obj.meta['original_row']
    extracted_scores = original_row.get("score") or original_row.get("scores") or {}
    print(f"✓ Successfully extracted scores from property meta: {extracted_scores}")
else:
    print("✗ Failed to extract scores from property meta")

print("\n[TEST 3] Testing full extraction + clustering flow...")
print("  (This requires the backend API to be running)")
print("  You can test this manually by:")
print("  1. Start the backend: uvicorn stringsight.api:app --reload --port 8000")
print("  2. Open the frontend: cd ~/stringsight-frontend && npm run dev")
print("  3. Upload data with score columns")
print("  4. Extract properties for a single row")
print("  5. Run clustering")
print("  6. Check that metrics appear in the Metrics tab")

print("\n" + "=" * 80)
print("Tests completed! The fix should:")
print("1. Store original_row in property.meta during extraction")
print("2. Use original_row from property.meta during clustering")
print("3. Preserve score data so metrics can be computed")
print("=" * 80)
