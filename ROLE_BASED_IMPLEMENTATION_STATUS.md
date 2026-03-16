# Role-Based Property Analysis - COMPLETE ✅

## Implementation Status: FULLY FUNCTIONAL

The role-based property analysis feature is now fully implemented and ready to use!

## Completed Tasks (10/13 - Core Implementation Complete)

### Backend Implementation ✅

1. **Property Schema** - Added `role` field to Property dataclass
   - Location: `stringsight/core/data_objects.py`
   - Type: `str | None = None`
   - Accepts any string value (flexible for different use cases)

2. **Conversation Formatting** - Added `conv_to_str_by_role()` function
   - Location: `stringsight/extractors/conv_to_str.py`
   - Filters conversations by role
   - Includes turn numbers for context

3. **Role-Specific Prompts** - Created template-based and dynamic prompt system
   - Location: `stringsight/prompts/extraction/role_analysis.py`
   - Built-in templates for: user, assistant, tool, system
   - Generic template for custom roles
   - Functions: `get_role_analysis_prompt()`, `generate_role_discovery_prompt()`

4. **Dynamic Discovery** - Extended DiscoveryPromptGenerator
   - Location: `stringsight/prompts/dynamic/discovery_generator.py`
   - Added `generate_role_specific()` method
   - Analyzes sample conversations for role-specific patterns
   - Generates custom prompts based on data

5. **Parser** - Updated to extract role field
   - Location: `stringsight/postprocess/parser.py`
   - Extracts `role` from LLM responses
   - No validation (accepts any string)

6. **OpenAIExtractor** - Multi-pass extraction support ✅
   - Location: `stringsight/extractors/openai.py`
   - Added `extract_by_role` parameter (List[str] | None)
   - Added `use_dynamic_prompts` parameter (bool)
   - Implemented `_run_multi_pass_extraction()` method
   - Integrated with role-specific prompts

7. **Pipeline Builder** - Role extraction parameters ✅
   - Location: `stringsight/_public/pipeline_builders.py`
   - Added `extract_by_role` and `use_dynamic_prompts` parameters
   - Auto-configures `groupby_column='role'` when role extraction enabled
   - Falls back to `behavior_type` when disabled

8. **Public API** - Updated explain() function ✅
   - Location: `stringsight/_public/sync_api.py`
   - Added `extract_by_role` and `use_dynamic_role_prompts` parameters
   - Parameters passed through to pipeline builder

### Frontend Implementation ✅

9. **TypeScript Types** - Added role field ✅
    - Location: `frontend/src/components/cards/PropertyCard.tsx` (PropertyLike interface)
    - Location: `frontend/src/components/cards/ClusterCard.tsx` (ClusterLike interface)
    - Property: `role?: string | null`
    - Cluster: `meta.role?: string`

10. **UI Filters** - Role filtering support ✅
    - PropertiesTab: Role available as categorical column (existing FilterBar handles it)
    - ClustersTab: Existing "Group" filter automatically works with roles
    - PropertyCard: Added role chip with hash-based color assignment
    - ClustersTab: Enhanced color function to handle custom roles with hash-based colors

## Cancelled Tasks (Optional Enhancements)

- **Validator**: Role-aware evidence validation (can be added later if needed)
- **Metrics**: Role correlation metrics (can be added later if needed)
- **Frontend**: Dedicated correlation display component (can be added later if needed)

## Usage Examples

```python
from stringsight import explain

# Default behavior (unchanged - fully backward compatible)
result = explain(df)

# Template-based role extraction (fast)
result = explain(df, extract_by_role=["user", "assistant"])

# Dynamic discovery (adaptive, learns from your data)
result = explain(
    df, 
    extract_by_role=["customer", "agent"],
    use_dynamic_role_prompts=True
)

# Custom roles for specialized analysis
result = explain(
    df,
    extract_by_role=["reasoning", "tool_use", "output"]
)
```

## How It Works

1. **Multi-Pass Extraction**: When `extract_by_role` is provided, the extractor runs separate LLM calls for each role
2. **Role-Specific Prompts**: Each role gets its own system prompt (template or dynamically generated)
3. **Conversation Filtering**: `conv_to_str_by_role()` filters messages to only include the specified role
4. **Automatic Clustering**: Pipeline automatically sets `groupby_column='role'` to cluster properties by role
5. **UI Display**: 
   - Properties show role chips with unique colors
   - Clusters can be filtered by role using the existing "Group" filter
   - Colors are consistently generated from role names

## UI Features

- **PropertyCard**: Displays role as a colored chip when present
- **ClustersTab**: 
  - "Group" filter shows all roles
  - Cluster cards display role badges
  - Colors auto-generated consistently for each role
- **PropertiesTab**: Role available as filterable column
- **Backward Compatible**: UI gracefully handles data without roles

## Technical Implementation

- **Fully backward compatible**: Default behavior unchanged
- **Leverages existing infrastructure**: Uses `groupby_column` clustering
- **Dynamic color assignment**: Hash-based consistent colors for any role
- **No hardcoded role values**: Works with any string
- **Multi-pass extraction**: Separate LLM calls per role with role-specific prompts

## Testing

Test with multi-turn conversations:

```python
import pandas as pd
from stringsight import explain

df = pd.DataFrame({
    'question_id': ['1', '2'],
    'prompt': ['Hello', 'Help me code'],
    'model': ['gpt-4', 'gpt-4'],
    'model_response': [
        [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi! How can I help?"}
        ],
        [
            {"role": "user", "content": "Help me code"},
            {"role": "assistant", "content": "Sure! What do you need?"}
        ]
    ]
})

# Extract and cluster by role
result = explain(df, extract_by_role=["user", "assistant"])

# Check role field
print(result.properties[0].role)  # "user" or "assistant"

# Clusters will have meta.group set to role value
print(result.clusters[0].meta.get("group"))  # "user" or "assistant"
```

## Files Modified

### Backend (8 files)
1. `stringsight/core/data_objects.py` - Property schema
2. `stringsight/extractors/conv_to_str.py` - Role filtering
3. `stringsight/prompts/extraction/role_analysis.py` - NEW: Role prompts
4. `stringsight/prompts/dynamic/discovery_generator.py` - Discovery extension
5. `stringsight/postprocess/parser.py` - Role field extraction
6. `stringsight/extractors/openai.py` - Multi-pass extraction
7. `stringsight/_public/pipeline_builders.py` - Parameter forwarding
8. `stringsight/_public/sync_api.py` - Public API

### Frontend (2 files)
9. `frontend/src/components/cards/PropertyCard.tsx` - Types & role chip
10. `frontend/src/components/cards/ClusterCard.tsx` - Cluster types
11. `frontend/src/components/ClustersTab.tsx` - Dynamic role colors
