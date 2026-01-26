# Frontend Changes for Optimized Dynamic Prompts

## Overview

With the backend optimization that now skips verification by default, we should update the frontend to:
1. Add options to control the optimization settings
2. Better communicate what "Generate from Traces" actually does
3. Optionally expose skip_verification and customize_clustering flags

## Recommended Changes

### Option 1: Simple (Recommended) - Just Update Text

**Minimal changes, best UX**

Update the description text in [PromptModeComponents.tsx](stringsight-frontend/src/components/sidebar-sections/PromptModeComponents.tsx) to reflect the new, faster behavior:

```tsx
// Line 55-59: Update the description
<MenuItem value="dynamic">
  <Box>
    <Typography variant="body2" fontWeight="medium">Generate from Traces (Fast)</Typography>
    <Typography variant="caption" color="text.secondary">
      AI expands task description from sample conversations
    </Typography>
  </Box>
</MenuItem>
```

**Why this works:**
- The backend now defaults to `skip_verification=True`, which is fast and reliable
- Users don't need to understand the technical details
- The simplified flow (expand task → use with template) is easy to explain

---

### Option 2: Add Advanced Options (Optional)

If you want to give power users control over the optimization:

#### 2.1. Add Schema Properties

Update `GeneratePromptsRequest` in [schemas.py](stringsight/schemas.py):

```python
class GeneratePromptsRequest(BaseModel):
    """Request for generating dynamic prompts without extraction."""
    rows: List[Dict[str, Any]]
    method: Optional[Literal["single_model", "side_by_side"]] = None
    task_description: Optional[str] = None
    num_samples: Optional[int] = 5
    model: Optional[str] = "gpt-4.1"
    output_dir: Optional[str] = None
    seed: Optional[int] = 42
    skip_verification: Optional[bool] = True  # NEW
    customize_clustering: Optional[bool] = True  # NEW
```

#### 2.2. Pass Through in API

Update [prompts.py](stringsight/routers/prompts.py) endpoint (line ~142):

```python
# Generate prompts (discovery + clustering)
discovery_prompt, custom_clustering_prompts, prompts_metadata = _generate_prompts(
    task_description=req.task_description,
    dataset=dataset,
    method=method,
    use_dynamic_prompts=True,
    dynamic_prompt_samples=req.num_samples or 5,
    model=req.model or "gpt-4.1",
    system_prompt_override=None,
    output_dir=req.output_dir,
    seed=req.seed or 42,
    skip_verification=req.skip_verification if req.skip_verification is not None else True,  # NEW
    customize_clustering=req.customize_clustering if req.customize_clustering is not None else True,  # NEW
)
```

Wait - I need to check if `_generate_prompts` accepts these parameters...

Actually, looking at [prompt_generation.py](stringsight/prompt_generation.py), the `generate_prompts()` function calls `generator.generate_all_prompts()` internally. We'd need to pass the parameters through. Let me check that flow.

Actually, a simpler approach:

#### 2.3. Add to Frontend UI (Advanced Section)

Add an "Advanced Options" accordion in `DynamicPromptsSection`:

```tsx
// After the task description field, before the Generate button:

{/* Advanced Options (Collapsible) */}
<Accordion sx={{ mb: 2 }}>
  <AccordionSummary expandIcon={<ExpandMoreIcon />}>
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
      <Typography variant="subtitle2">Advanced Options</Typography>
      <Tooltip title="Fine-tune prompt generation behavior">
        <InfoOutlinedIcon fontSize="small" sx={{ color: 'text.secondary' }} />
      </Tooltip>
    </Box>
  </AccordionSummary>
  <AccordionDetails>
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      {/* Skip Verification Toggle */}
      <FormControlLabel
        control={
          <Switch
            checked={skipVerification}
            onChange={(e) => setSkipVerification(e.target.checked)}
            disabled={disabled || isGenerating}
          />
        }
        label={
          <Box>
            <Typography variant="body2">Skip Verification (Recommended)</Typography>
            <Typography variant="caption" color="text.secondary">
              Use template with expanded task instead of custom generation
            </Typography>
          </Box>
        }
      />

      {/* Customize Clustering Toggle */}
      <FormControlLabel
        control={
          <Switch
            checked={customizeClustering}
            onChange={(e) => setCustomizeClustering(e.target.checked)}
            disabled={disabled || isGenerating}
          />
        }
        label={
          <Box>
            <Typography variant="body2">Customize Clustering Prompts</Typography>
            <Typography variant="caption" color="text.secondary">
              Generate task-specific clustering prompts (adds 3 LLM calls)
            </Typography>
          </Box>
        }
      />

      {/* Cost Estimate */}
      <Alert severity="info" icon={<InfoOutlinedIcon />}>
        <Typography variant="caption">
          {skipVerification && customizeClustering && "Fast mode: ~4 LLM calls"}
          {skipVerification && !customizeClustering && "Fastest mode: ~1 LLM call"}
          {!skipVerification && "Full mode: ~6-9 LLM calls (slower, may fail)"}
        </Typography>
      </Alert>
    </Box>
  </AccordionDetails>
</Accordion>
```

---

## Comparison of Options

| Aspect | Option 1: Simple | Option 2: Advanced |
|--------|------------------|-------------------|
| User complexity | Low (no decisions) | Medium (optional tuning) |
| Implementation effort | 5 minutes | 1-2 hours |
| Maintenance | Minimal | More code to maintain |
| User benefit | Best for 95% of users | Good for power users |
| Backend changes | None | Add params to schema & router |

---

## Recommendation: Option 1 (Simple)

**Why:**
1. **The defaults are now optimal** - `skip_verification=True, customize_clustering=True` works great for everyone
2. **Simpler UX** - Users don't need to understand verification or clustering customization
3. **Less maintenance** - No new UI controls or backend plumbing
4. **Faster to ship** - Just update a few text strings

**The only change needed:**

Update the menu item text in [PromptModeComponents.tsx](stringsight-frontend/src/components/sidebar-sections/PromptModeComponents.tsx) line 54-59:

```tsx
<MenuItem value="dynamic">
  <Box>
    <Typography variant="body2" fontWeight="medium">
      Generate from Traces
    </Typography>
    <Typography variant="caption" color="text.secondary">
      AI expands task from sample conversations (fast)
    </Typography>
  </Box>
</MenuItem>
```

And optionally update the tooltip on line 189:

```tsx
<Tooltip title="AI will expand your task description by analyzing sample conversations to identify what to look for.">
  <InfoOutlinedIcon fontSize="small" sx={{ color: 'text.secondary' }} />
</Tooltip>
```

---

## Optional: Add Success Metrics Display

Since verification is skipped, remove the "reflection attempts" chip display (line 230-236) or make it conditional:

```tsx
{generatedPrompts.reflection_attempts !== undefined &&
 generatedPrompts.reflection_attempts > 0 && (
  <Chip
    label={`${generatedPrompts.reflection_attempts} reflections`}
    size="small"
    color="info"
  />
)}
```

This chip will no longer show up (reflection_attempts will be 0 with skip_verification=True).

---

## Testing Frontend Changes

After making Option 1 changes:

1. **Visual check**: The "Generate from Traces" option should say "(fast)" or similar
2. **Functional check**: Click "Generate Prompts from Traces" and verify:
   - Completes faster than before
   - Shows "Prompts generated successfully"
   - No reflection attempts chip (or shows 0)
   - Shows expanded task description
   - Shows generated discovery prompt
   - Shows generated clustering prompts

3. **Backend logs**: Should show:
   ```
   Step 1: Expanding task description...
   Step 2: Using base template with expanded task (skipping custom generation and verification)...
   Step 3: Customizing clustering prompts in parallel...
   ```

---

## If You Want Option 2 Later

You can always add the advanced options later if users request more control. The backend already supports both modes via the parameters.

To add Option 2:
1. Update `GeneratePromptsRequest` schema
2. Pass parameters through in router
3. Add UI switches in frontend
4. Update API call in frontend to include the parameters

But start with Option 1 - it's the right default for most users.
# Frontend Changes Applied

## Summary

Updated the frontend to reflect the optimized dynamic prompt generation flow that now skips verification by default.

## Changes Made

### File: `stringsight-frontend/src/components/sidebar-sections/PromptModeComponents.tsx`

#### 1. Updated Menu Item Text (Line 55-58)

**Before:**
```tsx
<Typography variant="body2" fontWeight="medium">Generate from Traces</Typography>
<Typography variant="caption" color="text.secondary">
  AI generates prompts from sample conversations
</Typography>
```

**After:**
```tsx
<Typography variant="body2" fontWeight="medium">Generate from Traces (Fast)</Typography>
<Typography variant="caption" color="text.secondary">
  AI expands task from sample conversations
</Typography>
```

**Why:**
- Added "(Fast)" to indicate the optimized performance
- Changed "generates prompts" to "expands task" to better reflect what actually happens (task expansion rather than custom prompt generation)

#### 2. Updated Tooltip Text (Line 189)

**Before:**
```tsx
<Tooltip title="Provide context to guide prompt generation. If omitted, AI will analyze general behavioral patterns.">
```

**After:**
```tsx
<Tooltip title="AI will expand your task description by analyzing sample conversations to identify what to look for.">
```

**Why:**
- More accurately describes the new flow: task expansion → template
- Clearer about what the AI does (expands task description, not generates custom prompt)

#### 3. Reflection Attempts Display (Line 230-236)

**No change needed** - already correctly only shows when `reflection_attempts > 0`:
```tsx
{generatedPrompts.reflection_attempts !== undefined &&
 generatedPrompts.reflection_attempts > 0 && (
  <Chip label={`${generatedPrompts.reflection_attempts} reflection...`} />
)}
```

With `skip_verification=True` (the new default), this chip will simply not appear since `reflection_attempts` will be 0.

## What Users Will See

### Before Optimization:
- Menu: "Generate from Traces" → "AI generates prompts from sample conversations"
- Sometimes shows reflection attempts chip
- Takes 6-9 LLM calls (slower)

### After Optimization:
- Menu: "Generate from Traces (Fast)" → "AI expands task from sample conversations"
- No reflection attempts chip (always 0 with skip_verification=True)
- Takes 4 LLM calls (much faster)

## What Didn't Change

✅ **Discovery Prompt Display** - Still shows the full system prompt (just uses standard template)
✅ **Expanded Task Description** - Still shows the expanded task (this is what gets generated)
✅ **Clustering Prompts** - Still shows customized clustering prompts
✅ **Success Messages** - All work the same way
✅ **Error Handling** - No changes needed

## Testing

To verify the changes work correctly:

1. **Visual Check:**
   - Prompt mode dropdown should show "Generate from Traces (Fast)"
   - Description should say "AI expands task from sample conversations"
   - Tooltip should mention "expand your task description"

2. **Functional Check:**
   - Click "Generate Prompts from Traces"
   - Should complete faster than before
   - Should show "Prompts generated successfully"
   - Should NOT show reflection attempts chip
   - Should show expanded task description
   - Should show full discovery prompt (with expanded task in TASK CONTEXT section)
   - Should show customized clustering prompts

3. **Backend Logs:**
   ```
   Step 1: Expanding task description...
   Step 2: Using base template with expanded task (skipping custom generation and verification)...
   Step 3: Customizing clustering prompts in parallel...
   ```

## Future Improvements (Optional)

If you want to add advanced controls later, you could add:
- Switch to toggle `skip_verification` (on by default)
- Switch to toggle `customize_clustering` (on by default)
- Cost estimate showing number of LLM calls

But the current defaults work great for 95% of users, so these are optional.

## Backend Compatibility

No backend changes were required for these frontend updates. The backend already:
- ✅ Defaults to `skip_verification=True`
- ✅ Defaults to `customize_clustering=True`
- ✅ Returns all the same data in the response
- ✅ Works with existing API contracts

The frontend changes are purely cosmetic to better communicate what's happening.
# Frontend Layout Unified

## Summary

Both "Template (Default)" and "Generate from Traces (Fast)" modes now have **identical layouts** with a consistent user experience.

## Changes Made

### Before:
- **Template mode**: Task input + "Full System Prompt" accordion
- **Dynamic mode**: Task input + Generate button → Then showed 3 separate accordions (Discovery Prompt, Expanded Task, Clustering Prompts)

### After:
- **Template mode**: Task input + "Full System Prompt" accordion
- **Dynamic mode**: Task input + "Full System Prompt" accordion + Generate button → Simple success message

## Layout Structure (Both Modes)

```
┌─────────────────────────────────────────┐
│ Task Description Input                  │
│ (text area)                             │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ ▼ Full System Prompt                    │
│   (accordion - click to expand)         │
│   - Shows complete prompt with task     │
│     description inserted                │
└─────────────────────────────────────────┘

[DYNAMIC MODE ONLY:]
┌─────────────────────────────────────────┐
│ [Generate Prompts from Traces] Button   │
└─────────────────────────────────────────┘

[AFTER GENERATION - DYNAMIC MODE ONLY:]
┌─────────────────────────────────────────┐
│ ✓ Prompts generated successfully.       │
│   The full system prompt above has      │
│   been updated with your expanded       │
│   task description.                     │
└─────────────────────────────────────────┘
```

## Key Differences Between Modes

| Aspect | Template Mode | Dynamic Mode |
|--------|---------------|--------------|
| Task input | Required | Optional |
| System prompt | Shows immediately | Shows after generation |
| Generate button | No | Yes |
| Task expansion | No (uses as-is) | Yes (AI expands) |
| Success message | No | Yes (after generation) |

## User Flow

### Template Mode:
1. User enters task description
2. Full system prompt accordion updates automatically
3. User can expand to see the complete prompt
4. Ready to run extraction

### Dynamic Mode:
1. User enters task description (optional)
2. System prompt accordion shows placeholder
3. User clicks "Generate Prompts from Traces"
4. AI expands task description in background
5. System prompt accordion updates with expanded task
6. Success message appears
7. Ready to run extraction

## What Changed in Code

### `DynamicPromptsSection` Component:

**Added props:**
- `fullSystemPrompt?: string` - The complete system prompt to display
- `isLoadingPrompt?: boolean` - Loading state for the prompt

**Removed sections:**
- ❌ "Generated Discovery Prompt" accordion
- ❌ "Expanded Task Description" accordion
- ❌ "Generated Clustering Prompts" accordion
- ❌ Reflection attempts chip

**Added section:**
- ✅ "Full System Prompt" accordion (same as Template mode)
- ✅ Simple success message

## Benefits

1. **Consistent UX**: Both modes look and feel the same
2. **Simpler**: No overwhelming accordions with technical details
3. **Clearer**: Users see the final prompt that matters, not internals
4. **Unified**: Same "Full System Prompt" accordion in both modes
5. **Less cognitive load**: Users don't need to understand "discovery" vs "clustering" prompts

## Backend Integration

The parent component (in `App.tsx` or similar) needs to:

1. **For Template mode**:
   - Pass `fullSystemPrompt` with task inserted into template

2. **For Dynamic mode**:
   - Initially pass empty/placeholder `fullSystemPrompt`
   - On "Generate" click, call generation endpoint
   - When generation completes, extract `discovery_prompt` from response
   - Pass `discovery_prompt` as `fullSystemPrompt` prop
   - This updates the accordion automatically

## Example Integration Pattern

```tsx
// In parent component
const [dynamicFullPrompt, setDynamicFullPrompt] = useState<string>('');

const handleGenerate = async () => {
  const result = await generatePrompts(...);
  // Extract the discovery prompt (which IS the full system prompt)
  setDynamicFullPrompt(result.prompts.discovery_prompt);
};

// Render
<DynamicPromptsSection
  taskDescription={taskDescription}
  onTaskDescriptionChange={setTaskDescription}
  onGenerateClick={handleGenerate}
  generatedPrompts={generatedPrompts}
  fullSystemPrompt={dynamicFullPrompt}  // Pass the discovery prompt here
  isLoadingPrompt={isGenerating}
  ...
/>
```

## What Users See

### Both Modes Show:
```
Full System Prompt
├── Introduction
├── TASK CONTEXT
│   └── [User's task description - plain in Template, expanded in Dynamic]
├── Goal Instructions
├── Analysis Process
├── Output Schema
├── Behavior Types
├── Writing Properties
├── Field Requirements
└── Critical Constraints
```

The **only difference** is the task description section:
- **Template**: Uses task as entered
- **Dynamic**: Uses AI-expanded task with rich details

Everything else is identical!

## Migration Notes

This is a **breaking change** for the parent component - it must now:
1. Pass `fullSystemPrompt` to `DynamicPromptsSection`
2. Update `fullSystemPrompt` with `discovery_prompt` after generation
3. Remove any code that was handling the separate accordions

But the **user experience is much better** and the **code is simpler**.
