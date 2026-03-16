"""
Integration tests for prompts metadata in extraction API endpoints.

This test file verifies that:
1. /extract/single returns prompts metadata when use_dynamic_prompts=True
2. /extract/batch returns prompts metadata when use_dynamic_prompts=True
3. Background jobs store and return prompts metadata
4. Static prompts work when use_dynamic_prompts=False
5. Behavior when task_description is not provided
"""

from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from stringsight.app import app

client = TestClient(app)


def assert_test(condition: bool, message: str) -> None:
    """Simple assertion helper."""
    if not condition:
        raise AssertionError(message)


def get_test_data() -> dict:
    """Get minimal airline demo data for testing."""
    return {
        "question_id": "test_001",
        "prompt": "I need to book a flight from NYC to LAX for next week",
        "model": "gpt-4",
        "model_response": (
            "I'd be happy to help you book a flight! Let me search for available flights "
            "from New York to Los Angeles for next week."
        ),
    }


def get_task_description() -> str:
    """Get task description for dynamic prompts."""
    return (
        "This is a customer service chatbot for an airline. The chatbot helps customers\n"
        "with booking flights, checking flight status, and managing reservations."
    )


def test_extract_single_with_dynamic_prompts() -> None:
    """Test that /extract/single returns prompts metadata with dynamic prompts enabled."""
    print("\n" + "=" * 80)
    print("TEST: /extract/single with dynamic prompts")
    print("=" * 80)

    response = client.post("/extract/single", json={
        "row": get_test_data(),
        "task_description": get_task_description(),
        "use_dynamic_prompts": True,
        "dynamic_prompt_samples": 2,  # Use fewer samples for faster test
        "model_name": "gpt-4.1-mini",  # Use cheaper model
        "return_debug": False,
    })

    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    data = response.json()

    # Check response structure
    assert "properties" in data, "Response should contain 'properties'"
    assert "counts" in data, "Response should contain 'counts'"
    assert "prompts" in data, "Response should contain 'prompts' metadata"

    # Validate prompts metadata structure
    prompts = data["prompts"]
    assert prompts is not None, "Prompts should not be None"
    assert "discovery_prompt" in prompts, "Should contain discovery_prompt"
    assert "dynamic_prompts_used" in prompts, "Should contain dynamic_prompts_used"
    assert "task_description_original" in prompts, "Should contain task_description_original"

    # Verify dynamic prompts were used
    assert prompts["dynamic_prompts_used"] is True, "Should indicate dynamic prompts were used"
    assert prompts["discovery_prompt"] is not None, "Discovery prompt should not be None"
    assert len(prompts["discovery_prompt"]) > 100, "Discovery prompt should be substantial"

    # Check for task-specific content
    prompt_lower = prompts["discovery_prompt"].lower()
    has_airline_keywords = any(
        keyword in prompt_lower
        for keyword in ["airline", "flight", "booking", "customer service"]
    )
    assert has_airline_keywords, "Discovery prompt should contain airline-specific keywords"

    # Check clustering prompts if present
    if prompts.get("clustering_prompt"):
        assert len(prompts["clustering_prompt"]) > 100, "Clustering prompt should be substantial"

    print(f"✓ Test passed: Received prompts metadata with {len(prompts['discovery_prompt'])} char discovery prompt")
    print(f"✓ Dynamic prompts used: {prompts['dynamic_prompts_used']}")
    if prompts.get("verification_passed") is not None:
        print(f"✓ Verification passed: {prompts['verification_passed']}")


def test_extract_single_with_static_prompts() -> None:
    """Test that /extract/single works with dynamic prompts disabled."""
    print("\n" + "=" * 80)
    print("TEST: /extract/single with static prompts")
    print("=" * 80)

    response = client.post("/extract/single", json={
        "row": get_test_data(),
        "task_description": get_task_description(),
        "use_dynamic_prompts": False,  # Explicitly disable
        "model_name": "gpt-4.1-mini",
        "return_debug": False,
    })

    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    data = response.json()

    # Prompts field may or may not be present with static prompts
    # If present, it should indicate static prompts were used
    if "prompts" in data and data["prompts"] is not None:
        prompts = data["prompts"]
        assert prompts["dynamic_prompts_used"] is False, "Should indicate static prompts were used"

    print("✓ Test passed: Static prompts mode works correctly")


def test_extract_single_no_task_description() -> None:
    """Test behavior when task_description is not provided."""
    print("\n" + "=" * 80)
    print("TEST: /extract/single (no task_description)")
    print("=" * 80)

    response = client.post("/extract/single", json={
        "row": get_test_data(),
        # No task_description provided
        "model_name": "gpt-4.1-mini",
        "return_debug": False,
    })

    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    data = response.json()

    assert "properties" in data, "Response should contain 'properties'"
    assert "counts" in data, "Response should contain 'counts'"

    # Prompts field should not be present or should indicate static mode
    if "prompts" in data and data["prompts"] is not None:
        assert data["prompts"]["dynamic_prompts_used"] is False

    print("✓ Test passed: No task_description works")


def test_extract_batch_with_dynamic_prompts() -> None:
    """Test that /extract/batch returns prompts metadata with dynamic prompts enabled."""
    print("\n" + "=" * 80)
    print("TEST: /extract/batch with dynamic prompts")
    print("=" * 80)

    # Create 3 test rows
    rows = [get_test_data() for _ in range(3)]
    for i, row in enumerate(rows):
        row["question_id"] = f"test_{i:03d}"

    response = client.post("/extract/batch", json={
        "rows": rows,
        "task_description": get_task_description(),
        "use_dynamic_prompts": True,
        "dynamic_prompt_samples": 2,
        "model_name": "gpt-4.1-mini",
        "return_debug": False,
    })

    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    data = response.json()

    # Check response structure
    assert "rows" in data, "Response should contain 'rows'"
    assert "columns" in data, "Response should contain 'columns'"
    assert "counts" in data, "Response should contain 'counts'"
    assert "prompts" in data, "Response should contain 'prompts' metadata"

    # Validate prompts metadata
    prompts = data["prompts"]
    assert prompts is not None, "Prompts should not be None"
    assert prompts["dynamic_prompts_used"] is True, "Should indicate dynamic prompts were used"
    assert prompts["discovery_prompt"] is not None, "Discovery prompt should not be None"

    print("✓ Test passed: Batch extraction returned prompts metadata")
    print(f"✓ Processed {len(data['rows'])} property rows")


def test_background_job_with_prompts() -> None:
    """Test that background jobs store and return prompts metadata."""
    print("\n" + "=" * 80)
    print("TEST: Background job with dynamic prompts")
    print("=" * 80)

    # Start a background job
    rows = [get_test_data() for _ in range(2)]  # Small batch for faster test
    for i, row in enumerate(rows):
        row["question_id"] = f"job_test_{i:03d}"

    start_response = client.post("/extract/jobs/start", json={
        "rows": rows,
        "task_description": get_task_description(),
        "use_dynamic_prompts": True,
        "dynamic_prompt_samples": 2,
        "model_name": "gpt-4.1-mini",
    })

    assert start_response.status_code == 200, f"Expected 200, got {start_response.status_code}"
    job_id = start_response.json()["job_id"]
    print(f"✓ Job started: {job_id}")

    # Poll for job completion
    import time
    max_wait = 120  # 2 minutes max
    start_time = time.time()

    while time.time() - start_time < max_wait:
        status_response = client.get(f"/extract/jobs/status?job_id={job_id}")
        assert status_response.status_code == 200

        status = status_response.json()
        print(f"  Job status: {status['state']} ({status['progress']:.0%} complete)")

        if status["state"] == "done":
            break
        if status["state"] == "error":
            raise AssertionError(f"Job failed with error: {status.get('error')}")

        time.sleep(2)
    else:
        raise AssertionError(f"Job did not complete within {max_wait} seconds")

    # Get job results
    result_response = client.get(f"/extract/jobs/result?job_id={job_id}")
    assert result_response.status_code == 200

    result = result_response.json()

    # Check for prompts in result
    assert "properties" in result, "Result should contain 'properties'"
    assert "prompts" in result, "Result should contain 'prompts' metadata"

    prompts = result["prompts"]
    assert prompts is not None, "Prompts should not be None"
    assert prompts["dynamic_prompts_used"] is True, "Should indicate dynamic prompts were used"
    assert prompts["discovery_prompt"] is not None, "Discovery prompt should not be None"

    print("✓ Test passed: Background job returned prompts metadata")
    print(f"✓ Job completed with {result['count']} properties")


def test_full_pipeline_with_airline_data() -> None:
    """Integration test using real airline demo data."""
    print("\n" + "=" * 80)
    print("TEST: Full pipeline with airline demo data")
    print("=" * 80)

    repo_root = Path(__file__).resolve().parents[1]

    # Load airline data
    data_path = repo_root / "data/airline_data_demo.jsonl"
    if not data_path.exists():
        print("Skipping: airline_data_demo.jsonl not found")
        return
    df = pd.read_json(data_path, lines=True)

    # Take first row
    row = df.iloc[0].to_dict()

    task_description = (
        "This is a customer service chatbot for an airline. The chatbot helps customers\n"
        "with booking flights, checking flight status, managing reservations, and\n"
        "answering questions about airline policies."
    )

    response = client.post("/extract/single", json={
        "row": row,
        "task_description": task_description,
        "use_dynamic_prompts": True,
        "dynamic_prompt_samples": 3,
        "model_name": "gpt-4.1-mini",
        "return_debug": False,
    })

    assert response.status_code == 200
    data = response.json()

    # Validate prompts
    assert "prompts" in data
    prompts = data["prompts"]
    assert prompts["dynamic_prompts_used"] is True

    # Check for airline-specific content in prompts
    discovery_lower = prompts["discovery_prompt"].lower()
    assert any(kw in discovery_lower for kw in ["airline", "flight", "booking"])

    if prompts.get("clustering_prompt"):
        clustering_lower = prompts["clustering_prompt"].lower()
        assert any(kw in clustering_lower for kw in ["airline", "flight", "booking"])

    print("✓ Test passed: Full pipeline with real airline data")
    print(f"✓ Generated {len(data['properties'])} properties")
    print(f"✓ Discovery prompt: {len(prompts['discovery_prompt'])} chars")
    if prompts.get("clustering_prompt"):
        print(f"✓ Clustering prompt: {len(prompts['clustering_prompt'])} chars")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("RUNNING PROMPTS API INTEGRATION TESTS")
    print("=" * 80 + "\n")

    failed_tests: list[tuple[str, str]] = []
    passed_tests: list[str] = []

    tests = [
        ("extract_single_with_dynamic_prompts", test_extract_single_with_dynamic_prompts),
        ("extract_single_with_static_prompts", test_extract_single_with_static_prompts),
        ("extract_single_no_task_description", test_extract_single_no_task_description),
        ("extract_batch_with_dynamic_prompts", test_extract_batch_with_dynamic_prompts),
        ("background_job_with_prompts", test_background_job_with_prompts),
        ("full_pipeline_with_airline_data", test_full_pipeline_with_airline_data),
    ]

    for test_name, test_func in tests:
        try:
            test_func()
            passed_tests.append(test_name)
        except Exception as e:
            print(f"\nTEST FAILED: {test_name}")
            print(f"  Error: {e}")
            failed_tests.append((test_name, str(e)))

    print("\n" + "=" * 80)
    print(f"TEST SUMMARY: {len(passed_tests)} passed, {len(failed_tests)} failed")
    print("=" * 80)

    if passed_tests:
        print("\nPassed tests:")
        for test in passed_tests:
            print(f"  - {test}")

    if failed_tests:
        print("\nFailed tests:")
        for test, error in failed_tests:
            print(f"  - {test}: {error[:100]}")
        print("\n" + "=" * 80)
        print("TESTS FAILED")
        print("=" * 80 + "\n")
        raise SystemExit(1)

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED!")
    print("=" * 80 + "\n")



