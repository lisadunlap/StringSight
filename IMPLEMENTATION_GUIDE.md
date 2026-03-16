# Remaining Implementation Guide: Role-Based Property Analysis

This document provides detailed instructions for completing the role-based property analysis feature.

## What's Been Completed ✅

1. **Property Schema** - `role` field added to Property dataclass
2. **Conversation Formatting** - `conv_to_str_by_role()` function created
3. **Role-Specific Prompts** - Template and dynamic prompt system created
4. **Dynamic Discovery** - DiscoveryPromptGenerator extended with `generate_role_specific()`
5. **Parser** - Updated to extract `role` field from LLM responses

## What Needs to Be Completed

### 1. OpenAIExtractor Multi-Pass Implementation (CRITICAL)

**File**: `stringsight/extractors/openai.py`

**Changes Needed**:

a) Update `__init__` to add new parameters:
```python
def __init__(
    self,
    # ... existing parameters ...
    extract_by_role: Optional[List[str]] = None,
    use_dynamic_prompts: bool = False,
    **kwargs
):
    # ... existing init code ...
    self.extract_by_role = extract_by_role
    self.use_dynamic_prompts = use_dynamic_prompts
```

b) Modify `run()` method to handle multi-pass extraction:
```python
async def run(self, data: PropertyDataset, progress_callback: Any = None, **kwargs: Any) -> PropertyDataset:
    # If extract_by_role is None, use existing logic (backward compatible)
    if self.extract_by_role is None:
        return await self._run_single_pass(data, progress_callback, **kwargs)
    
    # Otherwise, run multi-pass extraction
    return await self._run_multi_pass(data, progress_callback, **kwargs)
```

c) Implement `_run_multi_pass()` method:
```python
async def _run_multi_pass(self, data: PropertyDataset, progress_callback: Any = None, **kwargs: Any) -> PropertyDataset:
    """Run extraction separately for each role."""
    all_properties = []
    
    for role in self.extract_by_role:
        self.log(f"Extracting properties for role: {role}")
        
        # Get role-specific system prompt
        if self.use_dynamic_prompts:
            from ..prompts.dynamic.discovery_generator import DiscoveryPromptGenerator
            from ..core.caching import UnifiedCache
            generator = DiscoveryPromptGenerator(UnifiedCache())
            # Get sample conversations for discovery
            sample_convs = [conv.responses for conv in data.conversations[:10]]
            role_prompt = generator.generate_role_specific(
                role=role,
                expanded_task_description="Extract properties from conversations",
                sample_conversations=sample_convs,
                model=self.model
            )
        else:
            from ..prompts.extraction.role_analysis import get_role_analysis_prompt
            role_prompt = get_role_analysis_prompt(role, task_description="")
        
        # Build prompts using role-specific filtering
        user_messages = []
        for conv in data.conversations:
            prompt = self._build_role_prompt(conv, role)
            user_messages.append(prompt)
        
        # Call LLM for this role
        raw_responses = await parallel_completions_async(
            user_messages,
            model=self.model,
            system_prompt=role_prompt,  # Use role-specific prompt
            max_workers=self.max_workers,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            show_progress=True,
            progress_desc=f"Extracting {role} properties",
            progress_callback=progress_callback
        )
        
        # Wrap responses in Property objects
        for conv, raw in zip(data.conversations, raw_responses):
            if raw is None:
                continue
            prop = Property(
                id=str(uuid.uuid4()),
                question_id=conv.question_id,
                model=conv.model,
                raw_response=raw,
                role=role  # Tag with role
            )
            all_properties.append(prop)
    
    # Return dataset with all properties from all roles
    return PropertyDataset(
        conversations=data.conversations,
        all_models=data.all_models,
        properties=all_properties,
        clusters=data.clusters,
        model_stats=data.model_stats,
    )
```

d) Add helper method for role-specific prompt building:
```python
def _build_role_prompt(self, conversation, role: str) -> Union[str, List[Dict[str, Any]]]:
    """Build prompt focusing on specific role messages."""
    from .conv_to_str import conv_to_str_by_role
    
    # Convert conversation to string, filtering by role
    if isinstance(conversation.responses, list):
        response = conv_to_str_by_role(conversation.responses, role)
    else:
        response = str(conversation.responses)
    
    return response
```

e) Rename existing `run()` to `_run_single_pass()` and keep all existing logic there.

### 2. Pipeline Builder Updates

**File**: `stringsight/_public/pipeline_builders.py`

**Changes**:

a) Add parameters to `build_default_pipeline()`:
```python
def build_default_pipeline(
    # ... existing parameters ...
    extract_by_role: List[str] | None = None,
    use_dynamic_prompts: bool = False,
    **kwargs
) -> Pipeline:
```

b) Pass to extractor:
```python
extractor_kwargs = {
    # ... existing kwargs ...
    'extract_by_role': extract_by_role,
    'use_dynamic_prompts': use_dynamic_prompts,
}
```

c) Auto-configure clustering by role:
```python
# If role extraction is enabled, cluster by role instead of behavior_type
if extract_by_role and not kwargs.get('groupby_column'):
    clusterer_kwargs['groupby_column'] = 'role'
elif not kwargs or 'groupby_column' not in kwargs:
    clusterer_kwargs['groupby_column'] = 'behavior_type'
```

### 3. Public API Updates

**File**: `stringsight/_public/sync_api.py`

**Changes**:

Add parameters to `explain()`:
```python
def explain(
    data: pd.DataFrame | PropertyDataset,
    # ... existing parameters ...
    extract_by_role: List[str] | None = None,
    use_dynamic_prompts: bool = False,
    **kwargs
) -> PropertyDataset:
    """
    ... existing docstring ...
    
    Args:
        extract_by_role: Optional list of roles to extract separately.
            When provided, runs separate extraction passes for each role.
            Examples: ["user", "assistant"], ["customer", "agent"]
        use_dynamic_prompts: If True, use dynamic discovery to generate
            role-specific prompts from sample data. Only applies when
            extract_by_role is provided.
    
    Examples:
        # Standard extraction (unchanged)
        >>> result = explain(df)
        
        # Role-based extraction with templates
        >>> result = explain(df, extract_by_role=["user", "assistant"])
        
        # Role-based with dynamic discovery
        >>> result = explain(
        ...     df, 
        ...     extract_by_role=["customer", "agent"],
        ...     use_dynamic_prompts=True
        ... )
    """
    # Pass to pipeline builder
    pipeline = build_default_pipeline(
        # ... existing arguments ...
        extract_by_role=extract_by_role,
        use_dynamic_prompts=use_dynamic_prompts,
        **kwargs
    )
```

### 4. Optional: Validator Updates

**File**: `stringsight/postprocess/validator.py`

**Changes**:

Add optional role-aware evidence validation:
```python
def __init__(
    self,
    # ... existing parameters ...
    validate_role_evidence: bool = False,
    **kwargs
):
    self.validate_role_evidence = validate_role_evidence
```

Add validation logic in `validate()` method to check that evidence comes from the correct role's messages (if role is set and validation is enabled).

### 5. Optional: Metrics Updates

**File**: `stringsight/metrics/functional_metrics.py`

**Changes**:

Add method for role correlation:
```python
def compute_role_correlation_metrics(
    self,
    role: str | None = None
) -> pd.DataFrame:
    """
    Compute correlation between role-specific property clusters and scores.
    
    Args:
        role: Specific role to analyze (e.g., "user"). If None, analyzes all roles.
        
    Returns:
        DataFrame with columns: cluster_id, role, avg_score, score_delta, prevalence
    """
    # Filter properties by role
    if role:
        role_props = [p for p in self.data.properties if p.role == role]
    else:
        role_props = self.data.properties
    
    # Compute average score for conversations containing each cluster
    # Similar to existing metrics but focused on role patterns
    # Return correlation analysis
```

## Frontend Implementation Summary

### 1. TypeScript Types

**File**: `frontend/src/types/index.ts`

Add `role` field to Property and Cluster interfaces.

### 2. UI Filters

**Files**: `frontend/src/components/PropertiesTab.tsx`, `frontend/src/components/ClustersTab.tsx`

Add multi-select dropdowns for role filtering:
- Extract unique role values from data
- Filter display based on selection
- Auto-generate colors for each role
- Only show filter when role data exists

### 3. Optional: Correlation Display

**File**: `frontend/src/components/metrics/RoleCorrelationMetrics.tsx`

Create component to display which role patterns correlate with high/low scores.

## Testing

Once implemented, test with:

```python
import pandas as pd
from stringsight import explain

# Create test data with multi-turn conversations
df = pd.DataFrame({
    'question_id': ['1', '2'],
    'prompt': ['Hello', 'How are you?'],
    'model': ['gpt-4', 'gpt-4'],
    'model_response': [
        [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ],
        [
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm doing well!"}
        ]
    ]
})

# Test role-based extraction
result = explain(df, extract_by_role=["user", "assistant"])

# Check that properties have role field
print(result.properties[0].role)  # Should be "user" or "assistant"

# Check that clustering grouped by role
print(result.clusters[0].meta.get("group"))  # Should be role value
```

## Backward Compatibility Checklist

- [ ] `extract_by_role=None` behaves exactly like before
- [ ] Existing code works without changes
- [ ] Properties without `role` field load correctly
- [ ] UI handles missing role data gracefully
- [ ] All existing tests pass

## Integration Checklist

- [ ] OpenAIExtractor multi-pass implementation
- [ ] Pipeline builder parameter forwarding
- [ ] Public API parameter forwarding
- [ ] Frontend type definitions
- [ ] Frontend role filters (basic)
- [ ] Documentation updates
- [ ] Example notebooks/scripts
- [ ] Unit tests for role extraction
- [ ] Integration tests end-to-end

## Priority Order

1. **Must Have** (Core Functionality):
   - OpenAIExtractor multi-pass
   - Pipeline builder updates
   - Public API updates

2. **Should Have** (Usability):
   - Frontend type definitions
   - Basic frontend filters
   - Documentation

3. **Nice to Have** (Advanced):
   - Role-aware evidence validation
   - Correlation metrics
   - Advanced UI components
