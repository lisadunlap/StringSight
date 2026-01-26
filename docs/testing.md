# Backend Testing Guide - Prompt Generation Endpoint

## Test File

**Location:** `tests/test_prompts_generate_endpoint.py`

## Running the Tests

```bash
# Run the test file directly
python tests/test_prompts_generate_endpoint.py

# Or use pytest
pytest tests/test_prompts_generate_endpoint.py -v
```

## Test Coverage

### ✅ 9 Comprehensive Tests

1. **test_prompts_generate_basic**
   - Tests basic prompt generation with single_model format
   - Verifies all 4 prompts are generated (discovery, clustering, dedup, outlier)
   - Checks PromptsMetadata structure
   - Validates task-specific content (airline keywords)
   - Verifies expanded task description is longer than original

2. **test_prompts_generate_side_by_side**
   - Tests side_by_side format detection
   - Verifies prompts contain comparison-related keywords
   - Validates model comparison context

3. **test_prompts_generate_no_task_description**
   - Tests fallback behavior when no task_description provided
   - Verifies system uses default task description
   - Checks for "behavioral patterns" in default

4. **test_prompts_generate_minimal_data**
   - Edge case: Only 1 conversation in dataset
   - Verifies system handles small datasets gracefully
   - Should still generate valid prompts

5. **test_prompts_generate_sample_size_parameter**
   - Tests num_samples parameter (1, 3, 5)
   - Verifies different sample sizes all work
   - Creates 10 rows, samples subsets

6. **test_prompts_generate_error_no_rows**
   - Error handling: Empty rows array
   - Expects 400 status code
   - Validates error message

7. **test_prompts_generate_error_invalid_method**
   - Error handling: Invalid column structure
   - Expects 422 status code
   - Tests method detection failure

8. **test_prompts_generate_deterministic_with_seed**
   - Tests deterministic behavior with fixed seed
   - Calls endpoint twice with seed=42
   - Verifies identical prompts (via caching)

9. **test_prompts_generate_with_airline_data**
   - Integration test with real data/airline_data_demo.jsonl
   - Tests full pipeline with realistic data
   - Validates domain-specific content in prompts

## Expected Test Output

```
================================================================================
RUNNING /prompts/generate ENDPOINT TESTS
================================================================================

================================================================================
TEST: POST /prompts/generate (basic single_model)
================================================================================
✓ Test passed: Generated all 4 prompts
✓ Discovery prompt: 1847 chars
✓ Clustering prompt: 523 chars
✓ Dedup prompt: 487 chars
✓ Outlier prompt: 456 chars
✓ Generation time: 12.34s
✓ Verification passed: True

[... 8 more tests ...]

================================================================================
TEST SUMMARY: 9 passed, 0 failed
================================================================================

Passed tests:
  ✓ prompts_generate_basic
  ✓ prompts_generate_side_by_side
  ✓ prompts_generate_no_task_description
  ✓ prompts_generate_minimal_data
  ✓ prompts_generate_sample_size_parameter
  ✓ prompts_generate_error_no_rows
  ✓ prompts_generate_error_invalid_method
  ✓ prompts_generate_deterministic_with_seed
  ✓ prompts_generate_with_airline_data

================================================================================
ALL TESTS PASSED! ✓
================================================================================
```

## Test Data

### Single Model Format
```python
{
    "question_id": "test_001",
    "prompt": "I need to book a flight from NYC to LAX for next week",
    "model": "gpt-4",
    "model_response": "I'd be happy to help you book a flight! ..."
}
```

### Side-by-Side Format
```python
{
    "question_id": "test_001",
    "prompt": "I need to book a flight from NYC to LAX for next week",
    "model_a": "gpt-4",
    "model_b": "claude-3",
    "model_a_response": "I'd be happy to help you book a flight! ...",
    "model_b_response": "I can assist with that. Let me check ..."
}
```

### Task Description
```python
"This is a customer service chatbot for an airline. The chatbot helps customers "
"with booking flights, checking flight status, and managing reservations."
```

## API Contract Validation

Each test validates the response structure:

```typescript
{
  prompts: {
    discovery_prompt: string,           // Main extraction prompt
    clustering_prompt: string,          // Clustering prompt
    dedup_prompt: string,               // Deduplication prompt
    outlier_prompt: string,             // Outlier detection prompt
    task_description_original: string,  // User's original task
    expanded_task_description: string,  // System-expanded task
    dynamic_prompts_used: boolean,      // Always true for this endpoint
    verification_passed?: boolean,      // Optional: prompt validation
    reflection_attempts?: number        // Optional: # of LLM reflections
  },
  generation_time_seconds: number       // Time taken
}
```

## Assertions Checked

### Required Fields
- ✅ All 4 prompts present (discovery, clustering, dedup, outlier)
- ✅ All prompts non-null and substantial (>50-100 chars)
- ✅ task_description_original preserved
- ✅ expanded_task_description exists and is longer
- ✅ dynamic_prompts_used = true
- ✅ generation_time_seconds > 0

### Content Quality
- ✅ Discovery prompt contains task-specific keywords
- ✅ Clustering prompt contains task-specific keywords
- ✅ Expanded task longer than original
- ✅ Side-by-side prompts mention comparison/models
- ✅ Prompts are substantial (not truncated)

### Error Handling
- ✅ 400 for empty rows
- ✅ 422 for invalid data structure
- ✅ Proper error messages in response

### Determinism
- ✅ Same seed produces identical prompts (via caching)
- ✅ Caching works correctly

## Prerequisites

### Required Environment Variables
```bash
export OPENAI_API_KEY="your-key"  # Required for prompt generation
```

### Optional Test Data
- `data/data/airline_data_demo.jsonl` - For integration test
- Skipped gracefully if not present

## Debugging Failed Tests

If a test fails, check:

1. **API Key:** Ensure OPENAI_API_KEY is set
2. **Model Availability:** gpt-4.1 must be accessible
3. **Network:** Tests make real API calls (not mocked)
4. **Timeout:** Generation can take 10-30 seconds per test
5. **Cache:** LMDB cache at `~/.stringsight/cache.lmdb`

## Performance Notes

- Each test makes real LLM calls (expensive!)
- Total test suite: ~2-3 minutes with caching
- Without caching: ~5-10 minutes
- Use gpt-4.1 (not gpt-4.1-mini) for better prompt quality

## Next Steps

After tests pass:

1. ✅ Backend endpoint verified working
2. 🔄 Add frontend API client method
3. 🔄 Create frontend UI components
4. 🔄 Integration testing end-to-end

## Related Files

- **Endpoint:** `stringsight/routers/prompts.py` (line 102)
- **Schema:** `stringsight/schemas.py` (GeneratePromptsRequest)
- **Prompt Generation:** `stringsight/prompt_generation.py`
- **Dynamic System:** `stringsight/prompts/dynamic/`
