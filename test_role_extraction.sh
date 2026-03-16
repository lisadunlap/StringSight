#!/bin/bash

# Test script to verify role-based extraction works
# This tests the backend API directly

echo "=========================================="
echo "Testing Role-Based Extraction Backend"
echo "=========================================="
echo ""

# Check if backend is running
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "❌ Backend is NOT running!"
    echo "Start it with: uvicorn stringsight.api:app --reload --port 8000"
    exit 1
fi

echo "✅ Backend is running"
echo ""

# Create a test request with role-based extraction
echo "📤 Sending test extraction request with extract_by_role=['user', 'assistant']..."
echo ""

RESPONSE=$(curl -s -X POST http://localhost:8000/extract/single \
  -H "Content-Type: application/json" \
  -d '{
    "row": {
      "question_id": "test-1",
      "prompt": "Hello, how are you?",
      "model": "gpt-4",
      "model_response": [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I am doing well, thank you!"}
      ],
      "score": {"helpfulness": 0.9}
    },
    "method": "single_model",
    "system_prompt": "universal",
    "extract_by_role": ["user", "assistant"],
    "use_dynamic_role_prompts": false,
    "model_name": "gpt-4o-mini",
    "temperature": 0.7,
    "return_debug": true
  }')

# Check if response is valid JSON
if ! echo "$RESPONSE" | jq . > /dev/null 2>&1; then
    echo "❌ Got invalid JSON response:"
    echo "$RESPONSE"
    exit 1
fi

echo "📥 Response received:"
echo ""

# Extract and display properties
PROPERTIES=$(echo "$RESPONSE" | jq '.properties')
PROP_COUNT=$(echo "$PROPERTIES" | jq 'length')

echo "Properties extracted: $PROP_COUNT"
echo ""

# Check if properties have role field
for i in $(seq 0 $(($PROP_COUNT - 1))); do
    PROP=$(echo "$PROPERTIES" | jq ".[$i]")
    ROLE=$(echo "$PROP" | jq -r '.role')
    PROP_DESC=$(echo "$PROP" | jq -r '.property_description' | cut -c1-50)

    if [ "$ROLE" = "null" ] || [ -z "$ROLE" ]; then
        echo "❌ Property $i: role=$ROLE (EMPTY!)"
    else
        echo "✅ Property $i: role=$ROLE"
    fi
    echo "   Description: $PROP_DESC..."
    echo ""
done

# Summary
ROLE_COUNT=$(echo "$PROPERTIES" | jq '[.[] | select(.role != null)] | length')

echo "=========================================="
echo "Summary:"
echo "  Total properties: $PROP_COUNT"
echo "  Properties with role: $ROLE_COUNT"
echo ""

if [ "$ROLE_COUNT" -eq 0 ]; then
    echo "❌ FAIL: No properties have role field!"
    echo ""
    echo "Check backend logs for:"
    echo "  - 'Running multi-pass extraction for roles:'"
    echo "  - 'role from original property: user'"
    echo "  - 'final role value: user'"
    exit 1
elif [ "$ROLE_COUNT" -eq "$PROP_COUNT" ]; then
    echo "✅ SUCCESS: All properties have role field!"
    exit 0
else
    echo "⚠️  PARTIAL: Only some properties have role field"
    exit 1
fi
