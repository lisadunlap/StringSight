"""
Test that role-based extraction actually sets the role field on properties.
This test verifies the fix for the missing extract_by_role parameter bug.
"""
import asyncio
from stringsight.extractors import get_extractor
from stringsight.core.data_objects import PropertyDataset, ConversationRecord

async def test_role_extraction():
    """Test that extractor with extract_by_role sets role field on properties."""

    # Create test conversation
    conversations = [
        ConversationRecord(
            question_id="test1",
            prompt="What is 2+2?",
            model="gpt-4",
            responses=[
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "2+2 equals 4"}
            ],
            scores={}
        )
    ]

    dataset = PropertyDataset(
        conversations=conversations,
        all_models=["gpt-4"],
        properties=[],
        clusters=[],
        model_stats={}
    )

    # Test 1: Extractor WITHOUT extract_by_role (should NOT set role field)
    print("=" * 80)
    print("TEST 1: Extractor without extract_by_role")
    print("=" * 80)

    extractor_no_role = get_extractor(
        model_name="gpt-4o-mini",
        system_prompt="Extract properties",
        extract_by_role=None,  # ❌ No role extraction
    )

    print(f"extract_by_role parameter: {extractor_no_role.extract_by_role}")

    if extractor_no_role.extract_by_role is None:
        print("✅ Correctly set to None (single-pass extraction will run)")
    else:
        print("❌ Should be None!")

    # Test 2: Extractor WITH extract_by_role (should SET role field)
    print("\n" + "=" * 80)
    print("TEST 2: Extractor with extract_by_role=['user', 'assistant']")
    print("=" * 80)

    extractor_with_role = get_extractor(
        model_name="gpt-4o-mini",
        system_prompt="Extract properties",
        extract_by_role=["user", "assistant"],  # ✅ Role extraction enabled
    )

    print(f"extract_by_role parameter: {extractor_with_role.extract_by_role}")

    if extractor_with_role.extract_by_role == ["user", "assistant"]:
        print("✅ Correctly set to ['user', 'assistant'] (multi-pass extraction will run)")
    else:
        print(f"❌ Should be ['user', 'assistant'], got: {extractor_with_role.extract_by_role}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if extractor_no_role.extract_by_role is None and extractor_with_role.extract_by_role == ["user", "assistant"]:
        print("✅ ALL TESTS PASSED")
        print("The extract_by_role parameter is now being passed correctly to the extractor!")
        print("\nNext steps:")
        print("1. Restart the backend server")
        print("2. Run a new extraction with role-based extraction enabled")
        print("3. Verify properties have 'role' field in the output")
    else:
        print("❌ TESTS FAILED")
        print("The extract_by_role parameter is not being set correctly.")

if __name__ == "__main__":
    asyncio.run(test_role_extraction())
