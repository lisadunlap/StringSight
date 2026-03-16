#!/usr/bin/env python3
"""
Quick script to check if your data has the right format for metrics.

Usage:
    python check_data_format.py <path_to_your_data.jsonl>

Example:
    python check_data_format.py airline_data_demo.jsonl
"""

import sys
import json

if len(sys.argv) < 2:
    print("Usage: python check_data_format.py <path_to_data.jsonl>")
    sys.exit(1)

filepath = sys.argv[1]

print("=" * 80)
print(f"Checking data format: {filepath}")
print("=" * 80)

try:
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        print("❌ File is empty!")
        sys.exit(1)

    print(f"\n✅ File has {len(lines)} rows")

    # Parse first row
    first_row = json.loads(lines[0])

    print(f"\n📋 Columns in data:")
    for key in sorted(first_row.keys()):
        value = first_row[key]
        value_type = type(value).__name__
        if isinstance(value, dict):
            value_preview = f"dict with keys: {list(value.keys())[:5]}"
        elif isinstance(value, str) and len(value) > 50:
            value_preview = f"{value[:50]}..."
        else:
            value_preview = str(value)[:100]
        print(f"  - {key}: {value_type} = {value_preview}")

    print(f"\n🔍 Checking for required fields:")

    # Check required fields
    required = ['question_id', 'prompt', 'model']
    for field in required:
        if field in first_row:
            print(f"  ✅ {field}: present")
        else:
            print(f"  ❌ {field}: MISSING")

    # Check for response fields
    response_fields = ['model_response', 'responses']
    has_response = any(field in first_row for field in response_fields)
    if has_response:
        print(f"  ✅ response field: present")
    else:
        print(f"  ❌ response field: MISSING (need 'model_response' or 'responses')")

    # Check for score columns (CRITICAL for metrics!)
    print(f"\n⭐ Checking for SCORE data (needed for metrics):")

    has_score_dict = 'score' in first_row or 'scores' in first_row
    if has_score_dict:
        score_field = 'score' if 'score' in first_row else 'scores'
        score_data = first_row[score_field]

        if isinstance(score_data, dict):
            print(f"  ✅ '{score_field}' field: present as dict")
            print(f"     Score keys: {list(score_data.keys())}")
            print(f"     Sample values: {dict(list(score_data.items())[:3])}")
        else:
            print(f"  ⚠️  '{score_field}' field exists but is not a dict (type: {type(score_data).__name__})")
    else:
        # Check for flattened score columns
        score_cols = [k for k in first_row.keys() if 'score' in k.lower() or 'rating' in k.lower() or
                     any(metric in k.lower() for metric in ['accuracy', 'helpfulness', 'quality', 'fluency'])]

        if score_cols:
            print(f"  ⚠️  No 'score' dict, but found potential score columns:")
            for col in score_cols[:5]:
                print(f"     - {col}: {first_row[col]}")
            print(f"\n  ℹ️  These will be auto-converted to a 'score' dict during clustering")
        else:
            print(f"  ❌ NO SCORE DATA FOUND!")
            print(f"     Without scores, metrics cannot be computed.")
            print(f"     Add a 'score' field with metric values:")
            print(f'     {{"score": {{"accuracy": 0.9, "helpfulness": 0.8}}}}')

    print("\n" + "=" * 80)
    print("Summary:")
    if has_score_dict or score_cols:
        print("✅ Your data looks good for computing metrics!")
    else:
        print("❌ Your data is MISSING score columns!")
        print("   Add score data to enable metrics computation.")
    print("=" * 80)

except FileNotFoundError:
    print(f"❌ File not found: {filepath}")
except json.JSONDecodeError as e:
    print(f"❌ Invalid JSON in file: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
