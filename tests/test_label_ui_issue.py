#!/usr/bin/env python3
"""
Quick test to diagnose the /label/run endpoint issue.
This simulates what the UI is sending to help identify the problem.
"""

import requests
import json

API_BASE = "http://localhost:8000"

# Minimal test payload - similar to what the UI would send
test_payload = {
    "rows": [
        {
            "question_id": "1",
            "prompt": "What is the capital of France?",
            "model": "gpt-4",
            "responses": "Paris is the capital of France.",
        },
        {
            "question_id": "2",
            "prompt": "What is 2+2?",
            "model": "gpt-4",
            "responses": "2+2 equals 4.",
        },
    ],
    "taxonomy": {
        "factual": "Questions asking for factual information",
        "mathematical": "Questions involving mathematical operations",
    },
    "model_name": "gpt-4.1-mini",
    "temperature": 0.0,
    "max_workers": 4,
    "sample_size": 2,
    "method": "single_model",
}

print("Testing /label/run endpoint...")
print(f"API Base: {API_BASE}")
print(f"Payload size: {len(json.dumps(test_payload))} bytes")
print("\n" + "="*80)

# Test 1: Check if server is up
print("\n1. Testing backend health...")
try:
    health_res = requests.get(f"{API_BASE}/health", timeout=5)
    if health_res.status_code == 200:
        print("✅ Backend is running")
    else:
        print(f"⚠️ Backend returned: {health_res.status_code}")
except requests.exceptions.ConnectionError:
    print("❌ Backend is not running! Start it with: uvicorn stringsight.api:app --reload")
    exit(1)
except Exception as e:
    print(f"❌ Error checking health: {e}")
    exit(1)

# Test 2: Send OPTIONS request (CORS preflight)
print("\n2. Testing OPTIONS request (CORS preflight)...")
try:
    options_res = requests.options(
        f"{API_BASE}/label/run",
        headers={
            "Origin": "http://localhost:5180",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
        timeout=5
    )
    print(f"OPTIONS response: {options_res.status_code}")
    if options_res.status_code == 200:
        print("✅ CORS preflight passed")
    else:
        print(f"⚠️ CORS preflight returned: {options_res.status_code}")
except Exception as e:
    print(f"❌ OPTIONS request failed: {e}")

# Test 3: Send actual POST request
print("\n3. Testing POST request to /label/run...")
try:
    print("Sending request...")
    response = requests.post(
        f"{API_BASE}/label/run",
        json=test_payload,
        headers={"Content-Type": "application/json"},
        timeout=120  # 2 minute timeout
    )
    
    print(f"Response status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Request succeeded!")
        print(f"\nResults:")
        print(f"  - Properties: {len(result.get('properties', []))}")
        print(f"  - Clusters: {len(result.get('clusters', []))}")
        
        if result.get('clusters'):
            print(f"\nCluster breakdown:")
            for cluster in result['clusters']:
                print(f"  - {cluster.get('cluster_label', 'Unknown')}: {cluster.get('size', 0)} instances")
    else:
        print(f"❌ Request failed with status {response.status_code}")
        print(f"Response body:\n{response.text[:500]}")
        
except requests.exceptions.Timeout:
    print("❌ Request timed out after 120 seconds")
except requests.exceptions.ConnectionError as e:
    print(f"❌ Connection error: {e}")
    print("\nThis could mean:")
    print("  - Backend server crashed")
    print("  - Server is not listening on port 8000")
    print("  - Network issue")
except Exception as e:
    print(f"❌ Unexpected error: {type(e).__name__}: {e}")

print("\n" + "="*80)
print("\nNext steps:")
print("1. Check the backend terminal for any error messages")
print("2. Check the browser console for detailed error messages")
print("3. Try with a smaller dataset if the payload is too large")
print("4. Verify VITE_BACKEND env var matches the backend URL")







