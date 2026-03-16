#!/usr/bin/env python3
"""
Add score columns to data that only has 'reward'.

This converts:
    {"reward": 0, ...}

To:
    {"reward": 0, "score": {"reward": 0.0}, ...}
"""

import json
import sys

if len(sys.argv) < 3:
    print("Usage: python add_scores_to_data.py <input.jsonl> <output.jsonl>")
    print("\nExample:")
    print("  python add_scores_to_data.py airline_data_demo.jsonl airline_data_with_scores.jsonl")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]

print(f"Reading from: {input_file}")
print(f"Writing to: {output_file}")

rows_processed = 0
rows_with_scores_added = 0

with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
    for line in f_in:
        if not line.strip():
            continue

        row = json.loads(line)

        # Add question_id if missing
        if 'question_id' not in row:
            row['question_id'] = str(rows_processed)

        # Add score field if missing
        if 'score' not in row and 'scores' not in row:
            # Create score dict from reward
            if 'reward' in row:
                row['score'] = {'reward': float(row['reward'])}
                rows_with_scores_added += 1
            else:
                # No reward or score - add a dummy score
                row['score'] = {'dummy': 1.0}
                rows_with_scores_added += 1

        f_out.write(json.dumps(row) + '\n')
        rows_processed += 1

print(f"\n✅ Processed {rows_processed} rows")
print(f"✅ Added score field to {rows_with_scores_added} rows")
print(f"\nNow you can use {output_file} for testing!")
