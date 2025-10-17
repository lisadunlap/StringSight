"""Basic integration test for OpenAIExtractor + LLMJsonParser.

Runs the two stages together with a monkey-patched extractor so no real
OpenAI call is made.  Executable with `pytest -q` or by running the file
directly.
"""

import json
from stringsight.core.data_objects import ConversationRecord, PropertyDataset
from stringsight.extractors.openai import OpenAIExtractor
from stringsight.postprocess.parser import LLMJsonParser


# ---------------------------------------------------------------------------
# Prepare synthetic data
# ---------------------------------------------------------------------------

conv = ConversationRecord(
    question_id="q1",
    prompt="What is the capital of France?",
    responses={
        "gpt-4o": "Paris is the capital of France.",
        "claude-3": "France's capital city is Paris.",
    },
    scores={},
    meta={},
)

dataset = PropertyDataset(conversations=[conv])

# ---------------------------------------------------------------------------
# Mock extractor – avoid external API calls
# ---------------------------------------------------------------------------

sample_json = {
    "properties": [
        {
            "property_description": "Provides correct factual answer",
            "category": "Knowledge",
            "type": "General",
            "impact": "High",
            "reason": "Answer is correct and concise",
            "evidence": "Says Paris is capital of France",
            "model": "gpt-4o",
        },
    ]
}
raw_response = json.dumps(sample_json)

# Monkey-patch the private batch method
OpenAIExtractor._extract_properties_batch = lambda self, msgs: [raw_response for _ in msgs]  # type: ignore

# ---------------------------------------------------------------------------
# Run stages
# ---------------------------------------------------------------------------

extractor = OpenAIExtractor(verbose=False, use_wandb=False)
parser = LLMJsonParser(verbose=False, use_wandb=False)

dataset_after_extract = extractor(dataset)
dataset_after_parse = parser(dataset_after_extract)

print(f"Dataset after extract: {dataset_after_extract}")
print(f"Dataset after parse: {dataset_after_parse}")

# ---------------------------------------------------------------------------
# Assertions (simple)
# ---------------------------------------------------------------------------

prop_list = dataset_after_parse.properties
assert len(prop_list) == 1, "Expected exactly one parsed property"
prop = prop_list[0]
assert prop.property_description == "Provides correct factual answer"
assert prop.model == "gpt-4o"

print("✅ Property extraction + parsing integration test passed!") 