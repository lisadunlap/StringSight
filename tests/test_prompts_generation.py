"""Test prompt generation and extraction endpoints."""
import asyncio
import json
import pandas as pd
from stringsight.routers.extraction import extract_single
from stringsight.schemas import ExtractSingleRequest

# Sample conversation data for single_model
SAMPLE_ROW = {
    "question_id": "test_0",
    "prompt": "Explain how transformers work",
    "model": "gpt-4",
    "model_response": "Transformers are neural network architectures that use self-attention mechanisms to process sequential data. They consist of encoder and decoder layers with multi-head attention.",
}

# Create 5 sample rows for prompt generation
SAMPLE_ROWS = [
    {
        "question_id": f"test_{i}",
        "prompt": f"Question {i}",
        "model": "gpt-4",
        "model_response": f"Answer {i} with some detailed explanation about AI models and their behavior.",
    }
    for i in range(5)
]


async def test_with_task_description():
    """Test extraction WITH a task description."""
    print("\n" + "=" * 80)
    print("TEST 1: Extraction WITH task description")
    print("=" * 80)

    req = ExtractSingleRequest(
        row=SAMPLE_ROW,
        sample_rows=SAMPLE_ROWS,
        method="single_model",
        task_description="Analyze how the AI model explains technical concepts",
        use_dynamic_prompts=True,
        dynamic_prompt_samples=5,
        model_name="gpt-4o-mini",
        output_dir="test_output_with_task"
    )

    try:
        result = await extract_single(req)
        print("\n✅ SUCCESS!")
        print(f"Properties extracted: {len(result.get('properties', []))}")

        if 'prompts' in result:
            prompts = result['prompts']
            print(f"\n📋 Prompts Metadata:")
            print(f"  - Dynamic Prompts Used: {prompts.get('dynamic_prompts_used')}")
            print(f"  - Verification Passed: {prompts.get('verification_passed')}")
            print(f"  - Reflection Attempts: {prompts.get('reflection_attempts')}")
            print(f"  - Has Discovery Prompt: {bool(prompts.get('discovery_prompt'))}")
            print(f"  - Has Clustering Prompt: {bool(prompts.get('clustering_prompt'))}")
            print(f"  - Has Dedup Prompt: {bool(prompts.get('dedup_prompt'))}")
            print(f"  - Has Outlier Prompt: {bool(prompts.get('outlier_prompt'))}")

            # Show first 200 chars of discovery prompt
            if prompts.get('discovery_prompt'):
                print(f"\n📝 Discovery Prompt Preview:")
                print(f"  {prompts['discovery_prompt'][:200]}...")
        else:
            print("\n❌ NO PROMPTS METADATA IN RESPONSE!")

        return result
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_without_task_description():
    """Test extraction WITHOUT a task description."""
    print("\n" + "=" * 80)
    print("TEST 2: Extraction WITHOUT task description")
    print("=" * 80)

    req = ExtractSingleRequest(
        row=SAMPLE_ROW,
        sample_rows=SAMPLE_ROWS,
        method="single_model",
        task_description=None,  # No task description
        use_dynamic_prompts=True,
        dynamic_prompt_samples=5,
        model_name="gpt-4o-mini",
        output_dir="test_output_without_task"
    )

    try:
        result = await extract_single(req)
        print("\n✅ SUCCESS!")
        print(f"Properties extracted: {len(result.get('properties', []))}")

        if 'prompts' in result:
            prompts = result['prompts']
            print(f"\n📋 Prompts Metadata:")
            print(f"  - Dynamic Prompts Used: {prompts.get('dynamic_prompts_used')}")
            print(f"  - Verification Passed: {prompts.get('verification_passed')}")
            print(f"  - Reflection Attempts: {prompts.get('reflection_attempts')}")
            print(f"  - Has Discovery Prompt: {bool(prompts.get('discovery_prompt'))}")
            print(f"  - Has Clustering Prompt: {bool(prompts.get('clustering_prompt'))}")
            print(f"  - Has Dedup Prompt: {bool(prompts.get('dedup_prompt'))}")
            print(f"  - Has Outlier Prompt: {bool(prompts.get('outlier_prompt'))}")

            # Show first 200 chars of discovery prompt
            if prompts.get('discovery_prompt'):
                print(f"\n📝 Discovery Prompt Preview:")
                print(f"  {prompts['discovery_prompt'][:200]}...")
        else:
            print("\n❌ NO PROMPTS METADATA IN RESPONSE!")

        return result
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_with_system_prompt_override():
    """Test extraction WITH system_prompt override (should skip dynamic generation)."""
    print("\n" + "=" * 80)
    print("TEST 3: Extraction WITH system_prompt override (should use static)")
    print("=" * 80)

    req = ExtractSingleRequest(
        row=SAMPLE_ROW,
        sample_rows=SAMPLE_ROWS,
        method="single_model",
        # Override: keep it intentionally simple, but long enough to pass the system prompt length guardrails.
        system_prompt=(
            "You are a helpful assistant. Extract behavioral properties from the model response. "
            "Return a valid JSON object that matches the required schema."
        ),
        task_description="This should be ignored",
        use_dynamic_prompts=True,
        dynamic_prompt_samples=5,
        model_name="gpt-4o-mini",
        output_dir="test_output_with_override"
    )

    try:
        result = await extract_single(req)
        print("\n✅ SUCCESS!")
        print(f"Properties extracted: {len(result.get('properties', []))}")

        if 'prompts' in result:
            prompts = result['prompts']
            print(f"\n📋 Prompts Metadata:")
            print(f"  - Dynamic Prompts Used: {prompts.get('dynamic_prompts_used')} (should be False)")
            print(f"  - Has Discovery Prompt: {bool(prompts.get('discovery_prompt'))}")
        else:
            print("\n❌ NO PROMPTS METADATA IN RESPONSE!")

        return result
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


async def main():
    """Run all tests."""
    print("\n🧪 TESTING PROMPT GENERATION AND EXTRACTION")
    print("=" * 80)

    # Test 1: With task description
    result1 = await test_with_task_description()

    # Test 2: Without task description
    result2 = await test_without_task_description()

    # Test 3: With system_prompt override
    result3 = await test_with_system_prompt_override()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Test 1 (with task desc):    {'✅ PASSED' if result1 else '❌ FAILED'}")
    print(f"Test 2 (without task desc): {'✅ PASSED' if result2 else '❌ FAILED'}")
    print(f"Test 3 (with override):     {'✅ PASSED' if result3 else '❌ FAILED'}")

    # Check files created
    import os
    print("\n📁 Files created:")
    for dir_name in ["test_output_with_task", "test_output_without_task", "test_output_with_override"]:
        if os.path.exists(dir_name):
            files = os.listdir(dir_name)
            print(f"  {dir_name}: {files}")


if __name__ == "__main__":
    asyncio.run(main())
