#!/usr/bin/env python3
"""
Unit test to verify role field is preserved through the pipeline.
This doesn't make LLM calls - it tests the data flow.
"""

import sys
sys.path.insert(0, '/home/lisabdunlap/StringSightNew')

from stringsight.core.data_objects import Property
from stringsight.postprocess.parser import LLMJsonParser
from stringsight.core.data_objects import PropertyDataset

print("=" * 80)
print("TEST: Role Field Preservation Through Parser")
print("=" * 80)
print()

# Step 1: Create a Property with role set (as extractor does)
print("[Step 1] Creating property with role='user' (simulating extractor)")
raw_response = '''
[
  {
    "behavior_type": "Positive",
    "property_description": "acknowledges greeting",
    "category": "Tone",
    "evidence": ["Hello!"],
    "reason": "Friendly interaction"
  }
]
'''

original_prop = Property(
    id="test-1",
    question_id="q1",
    model="gpt-4",
    raw_response=raw_response,
    role="user"  # ← This is set by multi-pass extraction
)

print(f"✓ Created property with role='{original_prop.role}'")
print()

# Step 2: Parse it (as parser does)
print("[Step 2] Parsing property through LLMJsonParser")
parser = LLMJsonParser()
dataset = PropertyDataset(
    conversations=[],
    all_models=["gpt-4"],
    properties=[original_prop],
    clusters=[],
    model_stats=[]
)

parsed_dataset = parser.run(dataset)
print(f"✓ Parser returned {len(parsed_dataset.properties)} properties")
print()

# Step 3: Check if role is preserved
print("[Step 3] Checking if role was preserved")
if len(parsed_dataset.properties) == 0:
    print("❌ FAIL: No properties returned from parser!")
    sys.exit(1)

parsed_prop = parsed_dataset.properties[0]
print(f"Parsed property role: '{parsed_prop.role}'")
print()

# Step 4: Verdict
print("=" * 80)
if parsed_prop.role == "user":
    print("✅ SUCCESS: Role field was preserved!")
    print(f"   Original: role='user'")
    print(f"   After parsing: role='{parsed_prop.role}'")
    sys.exit(0)
elif parsed_prop.role is None:
    print("❌ FAIL: Role field was lost!")
    print(f"   Original: role='user'")
    print(f"   After parsing: role=None")
    print()
    print("This means the parser is NOT preserving the role field.")
    print("Check stringsight/postprocess/parser.py line ~147")
    sys.exit(1)
else:
    print(f"⚠️  UNEXPECTED: Role changed from 'user' to '{parsed_prop.role}'")
    sys.exit(1)
