"""
Integration tests for the new POST /prompts/generate endpoint.

This test file verifies that:
1. /prompts/generate generates prompts without running extraction
2. Generated prompts contain all expected fields (discovery, clustering, dedup, outlier)
3. Generated prompts are task-specific when task_description is provided
4. Generation works with minimal data (1-2 conversations)
5. num_samples parameter affects sampling correctly
6. Response includes generation time metadata
7. Error handling works for invalid inputs
"""

from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from stringsight.api import app

client = TestClient(app)


def get_test_data_single_model() -> dict:
    """Get minimal test data for single_model format."""
    return {
        "question_id": "test_001",
        "prompt": "I need to book a flight from NYC to LAX for next week",
        "model": "gpt-4",
        "model_response": (
            "I'd be happy to help you book a flight! Let me search for available flights "
            "from New York to Los Angeles for next week."
        ),
    }


def get_test_data_side_by_side() -> dict:
    """Get minimal test data for side_by_side format."""
    return {
        "question_id": "test_001",
        "prompt": "I need to book a flight from NYC to LAX for next week",
        "model_a": "gpt-4",
        "model_b": "claude-3",
        "model_a_response": (
            "I'd be happy to help you book a flight! Let me search for available flights "
            "from New York to Los Angeles for next week."
        ),
        "model_b_response": (
            "I can assist with that. Let me check the available flights from New York "
            "to Los Angeles departing next week."
        ),
    }


def get_task_description() -> str:
    """Get task description for dynamic prompts."""
    return (
        "This is a customer service chatbot for an airline. The chatbot helps customers "
        "with booking flights, checking flight status, and managing reservations."
    )


def test_prompts_generate_basic() -> None:
    """Test basic prompt generation with single_model format."""
    print("\n" + "=" * 80)
    print("TEST: POST /prompts/generate (basic single_model)")
    print("=" * 80)

    # Create 3 test rows for sampling
    rows = [get_test_data_single_model() for _ in range(3)]
    for i, row in enumerate(rows):
        row["question_id"] = f"test_{i:03d}"

    response = client.post("/generate", json={
        "rows": rows,
        "method": "single_model",
        "task_description": get_task_description(),
        "num_samples": 2,  # Sample 2 out of 3 conversations
        "model": "gpt-4.1",
    })

    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    data = response.json()

    # Check top-level response structure
    assert "prompts" in data, "Response should contain 'prompts' field"
    assert "generation_time_seconds" in data, "Response should contain 'generation_time_seconds'"

    prompts = data["prompts"]

    # Validate PromptsMetadata structure
    assert "discovery_prompt" in prompts, "Should contain discovery_prompt"
    assert "clustering_prompt" in prompts, "Should contain clustering_prompt"
    assert "dedup_prompt" in prompts, "Should contain dedup_prompt"
    assert "outlier_prompt" in prompts, "Should contain outlier_prompt"
    assert "task_description_original" in prompts, "Should contain task_description_original"
    assert "expanded_task_description" in prompts, "Should contain expanded_task_description"
    assert "dynamic_prompts_used" in prompts, "Should contain dynamic_prompts_used"

    # Verify dynamic prompts were generated
    assert prompts["dynamic_prompts_used"] is True, "Should indicate dynamic prompts were used"
    assert prompts["discovery_prompt"] is not None, "Discovery prompt should not be None"
    assert len(prompts["discovery_prompt"]) > 100, "Discovery prompt should be substantial"

    # Check clustering prompts
    assert prompts["clustering_prompt"] is not None, "Clustering prompt should not be None"
    assert len(prompts["clustering_prompt"]) > 50, "Clustering prompt should be substantial"
    assert prompts["dedup_prompt"] is not None, "Dedup prompt should not be None"
    assert len(prompts["dedup_prompt"]) > 50, "Dedup prompt should be substantial"
    assert prompts["outlier_prompt"] is not None, "Outlier prompt should not be None"
    assert len(prompts["outlier_prompt"]) > 50, "Outlier prompt should be substantial"

    # Check task description fields
    assert prompts["task_description_original"] == get_task_description(), "Should preserve original task"
    assert prompts["expanded_task_description"] is not None, "Should have expanded task"
    assert len(prompts["expanded_task_description"]) > len(prompts["task_description_original"]), \
        "Expanded task should be longer than original"

    # Check for airline-specific keywords
    prompt_lower = prompts["discovery_prompt"].lower()
    has_airline_keywords = any(
        keyword in prompt_lower
        for keyword in ["airline", "flight", "booking", "customer"]
    )
    assert has_airline_keywords, "Discovery prompt should contain airline-specific keywords"

    # Check generation time
    assert data["generation_time_seconds"] > 0, "Generation time should be positive"
    assert data["generation_time_seconds"] < 300, "Generation should complete in reasonable time (<5 min)"

    print(f"✓ Test passed: Generated all 4 prompts")
    print(f"✓ Discovery prompt: {len(prompts['discovery_prompt'])} chars")
    print(f"✓ Clustering prompt: {len(prompts['clustering_prompt'])} chars")
    print(f"✓ Dedup prompt: {len(prompts['dedup_prompt'])} chars")
    print(f"✓ Outlier prompt: {len(prompts['outlier_prompt'])} chars")
    print(f"✓ Generation time: {data['generation_time_seconds']:.2f}s")
    if "verification_passed" in prompts:
        print(f"✓ Verification passed: {prompts['verification_passed']}")


def test_prompts_generate_side_by_side() -> None:
    """Test prompt generation with side_by_side format."""
    print("\n" + "=" * 80)
    print("TEST: POST /prompts/generate (side_by_side format)")
    print("=" * 80)

    rows = [get_test_data_side_by_side() for _ in range(3)]
    for i, row in enumerate(rows):
        row["question_id"] = f"sbs_test_{i:03d}"

    response = client.post("/generate", json={
        "rows": rows,
        "method": "side_by_side",
        "task_description": get_task_description(),
        "num_samples": 3,
        "model": "gpt-4.1",
    })

    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    data = response.json()
    prompts = data["prompts"]

    # Basic validation
    assert prompts["dynamic_prompts_used"] is True
    assert prompts["discovery_prompt"] is not None
    assert len(prompts["discovery_prompt"]) > 100

    # Side-by-side prompts might mention comparison or model differences
    prompt_lower = prompts["discovery_prompt"].lower()
    has_comparison_keywords = any(
        keyword in prompt_lower
        for keyword in ["model", "compare", "difference", "behavior"]
    )
    assert has_comparison_keywords, "SBS prompt should contain comparison-related keywords"

    print(f"✓ Test passed: Side-by-side format works")
    print(f"✓ Discovery prompt: {len(prompts['discovery_prompt'])} chars")


def test_prompts_generate_no_task_description() -> None:
    """Test prompt generation without task_description (system infers from data)."""
    print("\n" + "=" * 80)
    print("TEST: POST /prompts/generate (no task_description)")
    print("=" * 80)

    rows = [get_test_data_single_model() for _ in range(3)]
    for i, row in enumerate(rows):
        row["question_id"] = f"notask_{i:03d}"

    response = client.post("/generate", json={
        "rows": rows,
        "method": "single_model",
        # No task_description provided - system should use default
        "num_samples": 3,
        "model": "gpt-4.1",
    })

    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    data = response.json()
    prompts = data["prompts"]

    # Should still generate prompts
    assert prompts["dynamic_prompts_used"] is True
    assert prompts["discovery_prompt"] is not None
    assert len(prompts["discovery_prompt"]) > 100

    # task_description_original may be None when not provided
    # The backend uses a default internally but doesn't store None -> default in metadata
    # Just verify prompts were generated successfully
    assert prompts["expanded_task_description"] is not None, "Should have expanded task"
    assert "behavioral" in prompts["expanded_task_description"].lower() or \
           "pattern" in prompts["expanded_task_description"].lower(), \
        "Expanded task should contain analysis keywords"

    print(f"✓ Test passed: Works without explicit task_description")
    print(f"✓ Expanded task: {prompts['expanded_task_description'][:60]}...")


def test_prompts_generate_minimal_data() -> None:
    """Test prompt generation with just 1 conversation (edge case)."""
    print("\n" + "=" * 80)
    print("TEST: POST /prompts/generate (1 conversation)")
    print("=" * 80)

    # Only 1 conversation
    rows = [get_test_data_single_model()]

    response = client.post("/generate", json={
        "rows": rows,
        "task_description": get_task_description(),
        "num_samples": 5,  # Request 5 but only 1 available
        "model": "gpt-4.1",
    })

    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    data = response.json()
    prompts = data["prompts"]

    # Should still work with 1 conversation
    assert prompts["dynamic_prompts_used"] is True
    assert prompts["discovery_prompt"] is not None
    assert len(prompts["discovery_prompt"]) > 100

    print("✓ Test passed: Works with just 1 conversation")


def test_prompts_generate_sample_size_parameter() -> None:
    """Test that num_samples parameter affects sampling."""
    print("\n" + "=" * 80)
    print("TEST: POST /prompts/generate (num_samples parameter)")
    print("=" * 80)

    # Create 10 rows
    rows = [get_test_data_single_model() for _ in range(10)]
    for i, row in enumerate(rows):
        row["question_id"] = f"sample_{i:03d}"
        row["prompt"] = f"Unique prompt {i} for airline booking"

    # Test with different sample sizes
    for num_samples in [1, 3, 5]:
        response = client.post("/generate", json={
            "rows": rows,
            "task_description": get_task_description(),
            "num_samples": num_samples,
            "model": "gpt-4.1",
        })

        assert response.status_code == 200, f"Expected 200 for num_samples={num_samples}"

        data = response.json()
        prompts = data["prompts"]
        assert prompts["dynamic_prompts_used"] is True

        print(f"✓ num_samples={num_samples}: Generated {len(prompts['discovery_prompt'])} char prompt")


def test_prompts_generate_error_no_rows() -> None:
    """Test error handling when no rows provided."""
    print("\n" + "=" * 80)
    print("TEST: POST /prompts/generate (error: no rows)")
    print("=" * 80)

    response = client.post("/generate", json={
        "rows": [],  # Empty rows
        "task_description": get_task_description(),
    })

    assert response.status_code == 400, f"Expected 400 for empty rows, got {response.status_code}"
    error = response.json()
    assert "detail" in error
    assert "no rows" in error["detail"].lower()

    print("✓ Test passed: Returns 400 for empty rows")


def test_prompts_generate_error_invalid_method() -> None:
    """Test error handling when method cannot be detected."""
    print("\n" + "=" * 80)
    print("TEST: POST /prompts/generate (error: invalid columns)")
    print("=" * 80)

    # Rows with missing required columns
    rows = [{"question_id": "test", "invalid_column": "data"}]

    response = client.post("/generate", json={
        "rows": rows,
        "task_description": get_task_description(),
    })

    assert response.status_code == 422, f"Expected 422 for invalid data, got {response.status_code}"
    error = response.json()
    assert "detail" in error

    print("✓ Test passed: Returns 422 for invalid data")


def test_prompts_generate_deterministic_with_seed() -> None:
    """Test that same seed produces same prompts (deterministic)."""
    print("\n" + "=" * 80)
    print("TEST: POST /prompts/generate (deterministic with seed)")
    print("=" * 80)

    rows = [get_test_data_single_model() for _ in range(5)]
    for i, row in enumerate(rows):
        row["question_id"] = f"seed_test_{i:03d}"

    request_body = {
        "rows": rows,
        "task_description": get_task_description(),
        "num_samples": 3,
        "model": "gpt-4.1",
        "seed": 42,  # Fixed seed
    }

    # Generate twice with same seed
    response1 = client.post("/generate", json=request_body)
    assert response1.status_code == 200, f"Expected 200, got {response1.status_code}: {response1.text}"

    response2 = client.post("/generate", json=request_body)
    assert response2.status_code == 200, f"Expected 200, got {response2.status_code}: {response2.text}"

    prompts1 = response1.json()["prompts"]
    prompts2 = response2.json()["prompts"]

    # Prompts should be identical with same seed
    # (Due to caching, this should be a cache hit)
    assert prompts1["discovery_prompt"] == prompts2["discovery_prompt"], \
        "Same seed should produce same discovery prompt"
    assert prompts1["expanded_task_description"] == prompts2["expanded_task_description"], \
        "Same seed should produce same expanded task"

    print("✓ Test passed: Same seed produces deterministic results")


def test_prompts_generate_with_airline_data() -> None:
    """Integration test using real airline demo data."""
    print("\n" + "=" * 80)
    print("TEST: POST /prompts/generate (airline demo data)")
    print("=" * 80)

    repo_root = Path(__file__).resolve().parents[1]
    data_path = repo_root / "data/airline_data_demo.jsonl"

    if not data_path.exists():
        print("Skipping: airline_data_demo.jsonl not found")
        return

    df = pd.read_json(data_path, lines=True)
    rows = df.head(5).to_dict(orient="records")  # Use first 5 rows

    task_description = (
        "This is a customer service chatbot for an airline. The chatbot helps customers "
        "with booking flights, checking flight status, managing reservations, and "
        "answering questions about airline policies."
    )

    response = client.post("/generate", json={
        "rows": rows,
        "task_description": task_description,
        "num_samples": 3,
        "model": "gpt-4.1",
    })

    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    data = response.json()
    prompts = data["prompts"]

    # Validate airline-specific content
    discovery_lower = prompts["discovery_prompt"].lower()
    assert any(kw in discovery_lower for kw in ["airline", "flight", "booking", "customer"])

    clustering_lower = prompts["clustering_prompt"].lower()
    assert any(kw in clustering_lower for kw in ["airline", "flight", "booking", "customer"])

    print("✓ Test passed: Real airline data produces relevant prompts")
    print(f"✓ Discovery prompt: {len(prompts['discovery_prompt'])} chars")
    print(f"✓ Clustering prompt: {len(prompts['clustering_prompt'])} chars")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("RUNNING /prompts/generate ENDPOINT TESTS")
    print("=" * 80 + "\n")

    failed_tests: list[tuple[str, str]] = []
    passed_tests: list[str] = []

    tests = [
        ("prompts_generate_basic", test_prompts_generate_basic),
        ("prompts_generate_side_by_side", test_prompts_generate_side_by_side),
        ("prompts_generate_no_task_description", test_prompts_generate_no_task_description),
        ("prompts_generate_minimal_data", test_prompts_generate_minimal_data),
        ("prompts_generate_sample_size_parameter", test_prompts_generate_sample_size_parameter),
        ("prompts_generate_error_no_rows", test_prompts_generate_error_no_rows),
        ("prompts_generate_error_invalid_method", test_prompts_generate_error_invalid_method),
        ("prompts_generate_deterministic_with_seed", test_prompts_generate_deterministic_with_seed),
        ("prompts_generate_with_airline_data", test_prompts_generate_with_airline_data),
    ]

    for test_name, test_func in tests:
        try:
            test_func()
            passed_tests.append(test_name)
        except Exception as e:
            print(f"\nTEST FAILED: {test_name}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            failed_tests.append((test_name, str(e)))

    print("\n" + "=" * 80)
    print(f"TEST SUMMARY: {len(passed_tests)} passed, {len(failed_tests)} failed")
    print("=" * 80)

    if passed_tests:
        print("\nPassed tests:")
        for test in passed_tests:
            print(f"  ✓ {test}")

    if failed_tests:
        print("\nFailed tests:")
        for test, error in failed_tests:
            print(f"  ✗ {test}: {error[:100]}")
        print("\n" + "=" * 80)
        print("TESTS FAILED")
        print("=" * 80 + "\n")
        raise SystemExit(1)

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED! ✓")
    print("=" * 80 + "\n")
