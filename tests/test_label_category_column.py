#!/usr/bin/env python3
"""
Test script to verify that the category column (and other original columns)
are preserved in the label pipeline output.
"""

import pandas as pd
from stringsight.routers.extraction import label_run
from stringsight.schemas import LabelRequest
import asyncio

# Create test data with a category column
test_rows = [
    {
        "question_id": "1",
        "prompt": "What is the capital of France?",
        "model": "gpt-4",
        "responses": "Paris is the capital of France.",
        "category": "geography",  # This should be preserved!
        "custom_field": "test_value",  # This too!
    },
    {
        "question_id": "2",
        "prompt": "What is 2+2?",
        "model": "gpt-4",
        "responses": "2+2 equals 4.",
        "category": "mathematics",  # This should be preserved!
        "custom_field": "test_value2",
    },
    {
        "question_id": "3",
        "prompt": "Who wrote Romeo and Juliet?",
        "model": "gpt-4",
        "responses": "William Shakespeare wrote Romeo and Juliet.",
        "category": "literature",  # This should be preserved!
        "custom_field": "test_value3",
    },
]

taxonomy = {
    "factual": "Questions asking for factual information",
    "computational": "Questions involving calculations or computations",
}

print("="*80)
print("Testing Label Pipeline - Category Column Preservation")
print("="*80)

print("\n1. Input Data:")
print(f"   Rows: {len(test_rows)}")
print(f"   Columns: {list(test_rows[0].keys())}")
print(f"   Category values: {[r['category'] for r in test_rows]}")

print("\n2. Creating LabelRequest...")
request = LabelRequest(
    rows=test_rows,
    taxonomy=taxonomy,
    model_name="gpt-4.1-mini",
    temperature=0.0,
    max_workers=4,
    use_wandb=False,
    verbose=True,
)

print("\n3. Running label pipeline...")
print("   (This will make LLM API calls - may take 10-30 seconds)")

async def run_test():
    try:
        result = await label_run(request)
        
        print("\n4. ✅ Pipeline completed successfully!")
        print(f"   Properties returned: {len(result['properties'])}")
        
        if result['properties']:
            print("\n5. Checking for category column...")
            
            first_prop = result['properties'][0]
            print(f"\n   First property keys: {list(first_prop.keys())}")
            
            has_category = 'category' in first_prop
            has_custom = 'custom_field' in first_prop
            
            print(f"\n   ✓ Has 'category' column: {has_category}")
            print(f"   ✓ Has 'custom_field' column: {has_custom}")
            
            if has_category:
                categories = [p.get('category') for p in result['properties']]
                print(f"\n   Category values preserved: {categories}")
                print("\n   ✅ SUCCESS: Original columns are being preserved!")
            else:
                print("\n   ❌ FAILURE: 'category' column is missing!")
                print(f"\n   Available columns: {list(first_prop.keys())}")
            
            if has_custom:
                custom_values = [p.get('custom_field') for p in result['properties']]
                print(f"   Custom field values preserved: {custom_values}")
        
        # Show sample property
        print("\n6. Sample property (first result):")
        import json
        print(json.dumps(result['properties'][0], indent=2))
        
        return result
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

# Run the async test
result = asyncio.run(run_test())

print("\n" + "="*80)
print("Test Complete")
print("="*80)

if result and result['properties']:
    first_prop = result['properties'][0]
    if 'category' in first_prop and 'custom_field' in first_prop:
        print("\n✅ ALL CHECKS PASSED - Category column is preserved!")
    else:
        print("\n⚠️  SOME CHECKS FAILED - Review output above")
else:
    print("\n❌ TEST FAILED - No results returned")







