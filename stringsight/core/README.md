s# Core Data Objects

This module defines the core data contracts for the StringSight pipeline. These objects ensure consistency across extraction, clustering, and UI visualization.

## Data Objects

### ConversationRecord
Represents a single conversation trace (or pair of traces for side-by-side).
- **question_id** (`str`): Unique identifier for the question/prompt. MUST be a string.
- **prompt** (`str`): The user prompt.
- **model** (`str | List[str]`): Model name(s).
- **responses** (`str | List[str]`): Model response(s).
- **scores** (`Dict | List[Dict]`): Evaluation scores.

### Property
Represents an extracted behavioral property.
- **id** (`str`): Unique identifier for the property (UUID). MUST be a string.
- **question_id** (`str`): Links back to `ConversationRecord.question_id`. MUST be a string.
- **model** (`str`): The model that exhibited the behavior.
- **property_description** (`str`): Text description of the behavior.
- **raw_response** (`str`): The raw JSON string from the LLM.

### Cluster
Represents a group of similar properties.
- **id** (`str`): Cluster identifier. MUST be a string.
- **question_ids** (`List[str]`): List of `question_id`s in this cluster. All elements MUST be strings.
- **property_ids** (`List[str]`): List of `property.id`s in this cluster. All elements MUST be strings.

## Type Enforcement
All ID fields (`id`, `question_id`, `question_ids`, `property_ids`) are strictly enforced as strings in `__post_init__`. This prevents mismatches between numeric IDs (e.g. `1`) and string IDs (e.g. `"1"`) across the pipeline, specifically when moving between Python backend and JavaScript frontend.

## JSON Serialization
When using `PropertyDataset.to_serializable_dict()`, all objects are converted to JSON-safe types. ID fields will be serialized as strings.

## DataFrame Conversion
`PropertyDataset.to_dataframe()` ensures `question_id` columns are cast to `str` before merging conversations and properties, ensuring reliable joins even if input data had mixed types.

