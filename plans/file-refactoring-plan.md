# File Refactoring Plan for StringSight Large Files

## Executive Summary

This plan breaks down the 5 largest files in StringSight (total: 5,220 lines) into smaller, more maintainable modules. The refactorings are **prioritized by risk/reward ratio** and designed to fit within a **1-2 week timeline**.

**Target Outcome**: Reduce large file sizes by 30-50% while improving code organization and testability.

**Critical Constraints**:
- ❌ **No backward compatibility** - imports will need to be updated throughout the codebase
- ✅ **Output files unchanged** - file formats, names, JSON structure, and column names must remain identical
- ✅ **API behavior unchanged** - all public functions return same data structures

---

## Priority Order (Risk/Reward)

| File | Current Lines | Target Lines | Priority | Risk | Value |
|------|--------------|--------------|----------|------|-------|
| clustering_utils.py | 1,117 | ~350 | **P0** | LOW | HIGH |
| functional_metrics.py | 1,332 | ~600 | **P1** | MEDIUM | HIGH |
| extraction.py | 847 | ~750 | **P2** | LOW | MEDIUM |
| data_objects.py | 909 | ~900 | **P3** | LOW | LOW |
| api.py | 1,015 | ~1,000 | **P4** | MEDIUM | LOW |

---

## P0: clustering_utils.py (1,117 lines → ~350 lines)

**Timeline**: Days 1-4
**Risk**: LOW
**Value**: HIGH

### Current Structure

19 functions organized into:
- Embedding generation (4 functions, ~300 lines)
- LLM-based clustering (6 functions, ~250 lines)
- Persistence/WandB (5 functions, ~350 lines)
- Legacy/deprecated (2 functions, ~100 lines)
- Utilities (2 functions, ~117 lines with massive duplication)

### Refactoring Steps

#### Step 1: Extract `embeddings.py` (300 lines)

**Functions to move**:
```python
def _get_openai_embeddings_batch(...)
def _get_openai_embeddings(...)
def _get_embeddings(...)
def _setup_embeddings(...)
```

**New file**: `stringsight/clusterers/embeddings.py`

**Dependencies**:
```python
from stringsight.core.llm_utils import _normalize_embedding_model_name
from stringsight.core.caching import UnifiedCache
from stringsight.constants import DEFAULT_MAX_WORKERS
import numpy as np
```

**Update imports across codebase**:
```bash
# Find all files that import from clustering_utils
grep -r "from stringsight.clusterers.clustering_utils import" stringsight/

# Update to new imports:
# OLD: from stringsight.clusterers.clustering_utils import _get_embeddings
# NEW: from stringsight.clusterers.embeddings import _get_embeddings
```

#### Step 2: Extract `llm_clustering.py` (250 lines)

**Functions to move**:
```python
def _clean_list_item(...)
def generate_coarse_labels(...)
async def assign_fine_to_coarse(...)
async def llm_coarse_cluster_with_centers(...)
def embedding_match(...)
def match_label_names(...)
async def llm_match(...)
```

**New file**: `stringsight/clusterers/llm_clustering.py`

**Dependencies**:
```python
from stringsight.prompts.clustering.prompts import deduplication_clustering_systems_prompt
from stringsight.core.llm_utils import parallel_completions_async
from stringsight.core.caching import UnifiedCache
from .embeddings import _get_embeddings  # Internal import
```

#### Step 3: Extract `persistence.py` (350 lines)

**Functions to move**:
```python
def load_clustered_results(...)
def save_clustered_results(...)
def log_results_to_wandb(...)
def initialize_wandb(...)
def load_precomputed_embeddings(...)
```

**New file**: `stringsight/clusterers/persistence.py`

**Dependencies**:
```python
import pandas as pd
import pickle
import numpy as np
from pathlib import Path
```

#### Step 4: Delete Legacy Functions (NO RISK)

**Delete** (after verifying not imported anywhere):
```python
def _get_openai_embeddings_batch_litellm(...)  # Marked "kept for reference"
def _get_openai_embeddings_litellm(...)        # Not used by pipeline
```

**Verification command**:
```bash
grep -r "_get_openai_embeddings_batch_litellm\|_get_openai_embeddings_litellm" stringsight/
```

#### Step 5: Fix Massive Code Duplication in `create_summary_table`

**Issue**: Lines 963-1118 contain **the same 50-line block repeated 6 times** (likely copy-paste error).

**Fix**: Remove duplicate blocks, keep only lines 963-993.

**Before** (1118 lines):
```python
# Lines 963-993: First block (KEEP)
# Lines 994-1043: Duplicate (DELETE)
# Lines 1044-1093: Duplicate (DELETE)
# Lines 1094-1118: Duplicate (DELETE)
```

**After** (~993 lines).

### Final Structure

```
stringsight/clusterers/
├── clustering_utils.py          # DELETED (functions moved to new modules)
├── embeddings.py                # NEW: Embedding generation (300 lines)
├── llm_clustering.py           # NEW: LLM-based clustering (250 lines)
├── persistence.py              # NEW: Save/load/wandb (350 lines)
├── hdbscan.py                  # Existing
└── __init__.py                 # Export commonly used functions
```

**Note**: `clustering_utils.py` is DELETED entirely. All imports must be updated.

### Testing Checklist

- [ ] Find and update ALL imports from clustering_utils
- [ ] Run `pytest tests/` - all tests pass
- [ ] **CRITICAL: Verify output files unchanged**:
  - [ ] `clustered_results.jsonl` - same format and keys
  - [ ] `clustered_results.parquet` - same schema
  - [ ] All file names remain the same
- [ ] Verify cache hit rates unchanged
- [ ] Run full pipeline on airline demo dataset
- [ ] Check no duplicate code in create_summary_table
- [ ] Delete clustering_utils.py after confirming all imports updated

### Implementation Files

- Source: [stringsight/clusterers/clustering_utils.py](../stringsight/clusterers/clustering_utils.py)
- Update: [stringsight/clusterers/__init__.py](../stringsight/clusterers/__init__.py)
- Consumers: Check imports in routers/clustering.py, workers/tasks.py

---

## P1: functional_metrics.py (1,332 lines → ~600 lines)

**Timeline**: Days 5-9
**Risk**: MEDIUM
**Value**: HIGH

### Current Structure

Single large class `FunctionalMetrics(PipelineStage)` with 30+ methods:
- Data preparation (1 method, ~115 lines)
- Core metrics computation (10 methods, ~300 lines)
- **Bootstrap analysis** (10 methods, ~600 lines) ← **PRIMARY TARGET**
- Output formatting (5 methods, ~200 lines)
- Helper methods (5 methods, ~100 lines)

### Refactoring Steps

#### Step 1: Extract `bootstrap.py` (600 lines) ⚠️ MOST COMPLEX

**Create new class**:
```python
class BootstrapAnalyzer:
    """Bootstrap statistical analysis for metrics confidence intervals."""

    def __init__(self, samples: int = 100, seed: int | None = None):
        self.samples = samples
        self.seed = seed

    @staticmethod
    def _bootstrap_prepare(df, cluster_names, model_names) -> Dict[str, Any]:
        """Prepare data structures for bootstrap resampling."""

    @staticmethod
    def _bootstrap_allocate_arrays(S, n_models, n_clusters, n_metrics) -> Dict[str, Any]:
        """Allocate numpy arrays for bootstrap samples."""

    @staticmethod
    def _bootstrap_compute_one_replicate(prep, arrays, i, conv_weights) -> None:
        """Compute metrics for one bootstrap replicate."""

    @staticmethod
    def _bootstrap_attach_results(prep, arrays, model_cluster_scores,
                                    cluster_scores, model_scores) -> None:
        """Attach confidence intervals to metric dictionaries."""

    def _bootstrap_salience_from_proportions(self, proportions, n_models) -> Any:
        """Compute salience from proportion array."""

    @staticmethod
    def _compute_weighted_means_1d(group_idx, n_groups, weights, values) -> Any:
        """Compute weighted means per group (NaN-safe)."""

    @staticmethod
    def _compute_weighted_means_2d(row_idx, col_idx, n_rows, n_cols,
                                     weights, values) -> Any:
        """Compute weighted means for 2D grouping."""

    @staticmethod
    def _bootstrap_scores_to_matrix(scores_series, metric_to_idx, n_metrics) -> Any:
        """Convert score dicts to numpy matrix."""

    @staticmethod
    def _is_significant(lower: float, upper: float, contains: float = 0) -> bool:
        """Test if confidence interval excludes a value."""

    def add_bootstrap_analysis(
        self,
        df: pd.DataFrame,
        model_cluster_scores: Dict,
        cluster_scores: Dict,
        model_scores: Dict,
        cluster_names: List[str],
        model_names: List[str],
        progress_callback: Any = None
    ) -> Tuple[Dict, Dict, Dict]:
        """Add bootstrap confidence intervals to all metrics."""
```

**New file**: `stringsight/metrics/bootstrap.py`

**Dependencies**:
```python
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
```

**Integration in FunctionalMetrics.run()**:
```python
# Old code (lines 280-305):
if self.compute_bootstrap:
    model_cluster_scores, cluster_scores, model_scores = self._add_bootstrap_analysis(
        df, model_cluster_scores, cluster_scores, model_scores,
        cluster_names, model_names, progress_callback=progress_callback
    )

# New code:
if self.compute_bootstrap:
    from .bootstrap import BootstrapAnalyzer
    analyzer = BootstrapAnalyzer(
        samples=self.bootstrap_samples,
        seed=self.bootstrap_seed
    )
    model_cluster_scores, cluster_scores, model_scores = analyzer.add_bootstrap_analysis(
        df, model_cluster_scores, cluster_scores, model_scores,
        cluster_names, model_names, progress_callback=progress_callback
    )
```

**Testing Strategy** (CRITICAL):
1. Create unit tests with synthetic data (3 models, 5 clusters, 2 metrics)
2. Test bootstrap with fixed seed - verify deterministic results
3. **Compare output files before/after refactor** - must be byte-for-byte identical:
   - `model_cluster_scores.json`
   - `cluster_scores.json`
   - `model_scores.json`
   - All JSON keys and structure unchanged
4. Compare bootstrap CIs before/after refactor with same seed (must be identical)
5. Test edge cases:
   - Single model (bootstrap should handle gracefully)
   - Single cluster
   - Empty metrics
   - All NaN values in a metric
6. Performance test: Verify no slowdown

#### Step 2: Extract `data_preparation.py` (150 lines)

**Functions to extract**:
```python
def prepare_metrics_dataframe(data: PropertyDataset) -> pd.DataFrame:
    """Convert PropertyDataset to metrics dataframe.

    Extracted from FunctionalMetrics._prepare_data() (lines 306-420).
    """

def infer_metric_keys(df: pd.DataFrame) -> List[str]:
    """Infer metric column names from dataframe.

    Extracted from FunctionalMetrics._infer_metric_keys() (lines 547-563).
    """
```

**New file**: `stringsight/metrics/data_preparation.py`

**Integration**:
```python
# In FunctionalMetrics.run()
from .data_preparation import prepare_metrics_dataframe
df = prepare_metrics_dataframe(data)
```

#### Step 3: Extract `salience.py` (50 lines)

**Function to extract**:
```python
def compute_salience(
    model_cluster_scores: Dict[str, Dict[str, Dict[str, Any]]]
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Compute salience (proportion deviation from other models).

    Extracted from FunctionalMetrics._compute_salience() (lines 656-688).

    For each model-cluster combination, computes how much that model
    over/under-represents in that cluster compared to other models.

    Salience = model_proportion - avg_proportion_of_other_models
    """
```

**New file**: `stringsight/metrics/salience.py`

### Final Structure

```
stringsight/metrics/
├── functional_metrics.py         # Core class (~600 lines, down from 1332)
├── bootstrap.py                  # NEW: Bootstrap analysis (600 lines)
├── data_preparation.py          # NEW: Data prep helpers (150 lines)
├── salience.py                  # NEW: Salience computation (50 lines)
├── data_transformers.py         # Existing (465 lines)
├── plotting.py                  # Existing (969 lines)
├── frontend_adapters.py         # Existing
└── __init__.py
```

### Testing Checklist

- [ ] Unit tests for BootstrapAnalyzer with synthetic data
- [ ] **CRITICAL: Compare output files before/after** - must be identical:
  - [ ] `model_cluster_scores.json` - byte-for-byte identical
  - [ ] `cluster_scores.json` - byte-for-byte identical
  - [ ] `model_scores.json` - byte-for-byte identical
  - [ ] All JSON keys unchanged (no renames, no additions, no deletions)
- [ ] Compare bootstrap results before/after (identical with same seed)
- [ ] Test all edge cases (single model, single cluster, NaN values)
- [ ] Run full metrics pipeline on airline demo
- [ ] Verify WandB logging still works (same table structure)
- [ ] Performance test (should be no slower)
- [ ] Check all imports: `from stringsight.metrics.functional_metrics import FunctionalMetrics`

### Implementation Files

- Source: [stringsight/metrics/functional_metrics.py](../stringsight/metrics/functional_metrics.py)
- Update: [stringsight/metrics/__init__.py](../stringsight/metrics/__init__.py)
- Test: Create [tests/unit/test_bootstrap.py](../tests/unit/test_bootstrap.py)
- Consumers: Check imports in workers/tasks.py, routers/clustering.py

---

## P2: routers/extraction.py (847 lines → ~750 lines)

**Timeline**: Day 10
**Risk**: LOW
**Value**: MEDIUM

### Current Structure

- Schemas (1 dataclass, ~15 lines)
- Helper functions (1 function, ~40 lines)
- API endpoints (10 endpoints, ~790 lines)

### Refactoring Steps

#### Step 1: Extract `extraction_helpers.py` (100 lines)

**Functions to create**:
```python
def enrich_properties_with_row_index(
    properties: List[Dict[str, Any]],
    df: pd.DataFrame,
    method: Literal["single_model", "side_by_side"]
) -> List[Dict[str, Any]]:
    """Add UI row indices to properties using vectorized pandas operations.

    Extracted from _enrich_properties_with_row_index (lines 64-106).
    """

def build_index_map(
    df: pd.DataFrame,
    method: Literal["single_model", "side_by_side"]
) -> Dict[Tuple[str, str], int]:
    """Build mapping from (question_id, model) to row index.

    Consolidates logic from multiple endpoints (lines 76-94).
    """

def validate_extraction_request(
    df: pd.DataFrame,
    method: Literal["single_model", "side_by_side"]
) -> None:
    """Validate dataframe has required columns for extraction.

    Raises:
        ValueError: If required columns are missing
    """
```

**New file**: `stringsight/routers/extraction_helpers.py`

**Benefits**:
- Reduces duplication (index mapping logic appears in 3+ endpoints)
- Makes endpoints more readable
- Easier to test helper logic independently

#### Step 2: Update imports in extraction.py

```python
from .extraction_helpers import (
    enrich_properties_with_row_index,
    build_index_map,
    validate_extraction_request
)
```

### Final Structure

```
stringsight/routers/
├── extraction.py                 # Main endpoints (~750 lines)
├── extraction_helpers.py         # NEW: Shared helpers (100 lines)
├── clustering.py                 # Existing
├── jobs.py                       # Existing
└── ...
```

### Testing Checklist

- [ ] Test helper functions with edge cases
- [ ] Run API integration tests
- [ ] Verify all extraction endpoints work
- [ ] Test both single_model and side_by_side modes
- [ ] Check error handling

---

## P3: data_objects.py (909 lines → ~900 lines)

**Timeline**: OPTIONAL
**Risk**: LOW
**Value**: LOW

### Recommendation: **DEFER OR SKIP**

This file is **already well-organized** with clear class boundaries:
- Helper functions (2 functions, ~70 lines)
- Dataclasses (5 classes, ~500 lines)
- PropertyDataset (1 class, ~340 lines)

### Optional Improvement

If time permits, extract utility functions:

**Create**: `stringsight/core/format_utils.py` (~70 lines)
```python
def simple_to_oai_format(prompt: str, response: str) -> list:
    """Convert simple prompt-response to OpenAI format."""

def check_and_convert_to_oai_format(
    prompt: str,
    response: str
) -> Tuple[list, bool]:
    """Check if response is OAI format, convert if needed."""
```

**Benefit**: Minimal - these are closely tied to ConversationRecord.

**Recommendation**: Focus on P0-P2 instead.

---

## P4: api.py (1,015 lines → ~1,000 lines)

**Timeline**: OPTIONAL
**Risk**: MEDIUM
**Value**: LOW

### Recommendation: **DEFER**

Most refactoring already done via routers/. Remaining code is:
- Cache management (~130 lines) - recently improved with bounded LRU
- Background job system (~500 lines) - complex, tightly coupled to FastAPI
- Helper functions (~180 lines) - varied functionality

### Optional Improvements

If time permits:

#### Option 1: Extract `workers/cluster_job.py` (~500 lines)

Move background clustering job system to workers module.

**Concern**: Requires access to FastAPI app instance, may increase complexity.

#### Option 2: Extract `cache/jsonl_cache.py` (~130 lines)

Move JSONL cache to separate module.

**Benefit**: Makes cache reusable, easier to test.

**Recommendation**: Only do if P0-P2 complete early.

---

## Implementation Roadmap

### Week 1: High-Impact, Low-Risk

**Days 1-2: clustering_utils.py - Embeddings & LLM**
- [ ] Extract embeddings.py (300 lines)
- [ ] Extract llm_clustering.py (250 lines)
- [ ] Update imports with backward compatibility
- [ ] Run tests

**Days 3-4: clustering_utils.py - Persistence & Cleanup**
- [ ] Extract persistence.py (350 lines)
- [ ] Delete legacy litellm functions
- [ ] Fix create_summary_table duplication (remove ~125 lines)
- [ ] Full test suite
- [ ] Verify backward compatibility

**Day 5: routers/extraction.py**
- [ ] Extract extraction_helpers.py (100 lines)
- [ ] Update imports
- [ ] Run integration tests

### Week 2: Medium-Risk, High-Value

**Days 1-3: functional_metrics.py - Bootstrap (MOST COMPLEX)**
- [ ] Extract bootstrap.py (600 lines)
- [ ] Create comprehensive unit tests
- [ ] Test with fixed seeds - verify deterministic results
- [ ] Compare outputs before/after (must be identical)
- [ ] Test edge cases (single model, NaN values, etc.)

**Days 4-5: functional_metrics.py - Cleanup**
- [ ] Extract data_preparation.py (150 lines)
- [ ] Extract salience.py (50 lines)
- [ ] Final integration testing
- [ ] Update documentation
- [ ] Performance testing

---

## Testing Strategy

### 1. Unit Tests (Per Module)

**embeddings.py**:
```python
def test_get_embeddings_batch():
    # Test batch embedding generation
    # Verify cache behavior

def test_setup_embeddings():
    # Test model normalization
    # Test OpenAI vs local embeddings
```

**bootstrap.py** (CRITICAL):
```python
def test_bootstrap_with_fixed_seed():
    # Bootstrap with seed=42 should give same CIs every time

def test_bootstrap_single_model():
    # Edge case: only one model

def test_bootstrap_nan_handling():
    # Metrics with NaN values

def test_bootstrap_ci_coverage():
    # Verify CIs contain true means (synthetic data)
```

### 2. Integration Tests

```python
def test_full_pipeline_after_refactor():
    """Run complete pipeline on airline demo."""
    import json
    import filecmp

    # Run pipeline and save to temp directory
    result_before = run_pipeline_before_refactor()  # Baseline
    result_after = run_pipeline_after_refactor()    # After refactor

    # CRITICAL: Output files must be identical
    assert filecmp.cmp("before/model_cluster_scores.json", "after/model_cluster_scores.json")
    assert filecmp.cmp("before/cluster_scores.json", "after/cluster_scores.json")
    assert filecmp.cmp("before/model_scores.json", "after/model_scores.json")

    # Verify JSON structure unchanged
    with open("after/model_cluster_scores.json") as f:
        data = json.load(f)
        assert set(data.keys()) == expected_model_names
        for model in data.values():
            for cluster in model.values():
                # All expected keys present
                assert "size" in cluster
                assert "proportion" in cluster
                assert "quality" in cluster
                # No new keys added
                assert set(cluster.keys()).issubset(expected_keys)

def test_new_import_paths():
    """Verify new import paths work."""
    from stringsight.clusterers.embeddings import _get_embeddings
    from stringsight.clusterers.llm_clustering import generate_coarse_labels
    from stringsight.clusterers.persistence import save_clustered_results
    # Should not raise ImportError
```

### 3. Performance Tests

```python
def test_no_performance_regression():
    # Time full pipeline before/after
    # Should be within 5% (allow for variance)
```

---

## Import Update Strategy

**Approach**: Break first, fix imports second. No backward compatibility.

### Step 1: Find All Affected Imports

```bash
# Find all files importing from clustering_utils
grep -r "from stringsight.clusterers.clustering_utils" stringsight/ routers/ workers/ tests/

# Find all files importing from functional_metrics internals (if any)
grep -r "from stringsight.metrics.functional_metrics import.*bootstrap" stringsight/
```

### Step 2: Create Import Mapping

**clustering_utils.py → New locations**:
```python
# Embeddings
_get_embeddings → stringsight.clusterers.embeddings
_setup_embeddings → stringsight.clusterers.embeddings

# LLM clustering
generate_coarse_labels → stringsight.clusterers.llm_clustering
assign_fine_to_coarse → stringsight.clusterers.llm_clustering

# Persistence
save_clustered_results → stringsight.clusterers.persistence
load_clustered_results → stringsight.clusterers.persistence
log_results_to_wandb → stringsight.clusterers.persistence
```

### Step 3: Batch Update with sed/awk

```bash
# Example: Update _get_embeddings imports
find stringsight -name "*.py" -exec sed -i 's/from stringsight.clusterers.clustering_utils import _get_embeddings/from stringsight.clusterers.embeddings import _get_embeddings/g' {} +
```

### Step 4: Delete Old Files

Once all imports updated and tests pass:
```bash
rm stringsight/clusterers/clustering_utils.py
```

---

## Summary

This plan provides a **concrete, actionable roadmap** for breaking up StringSight's largest files. The refactorings are:

✅ **Practical**: Focus on high-value extractions, not perfect architecture
✅ **Clean**: No backward compatibility - imports updated throughout
✅ **Testable**: Clear testing strategy for each change
✅ **Incremental**: Can be done module by module
✅ **Time-bound**: Fits 1-2 week timeline

**Recommended execution order**: P0 (clustering_utils) → P1 (functional_metrics) → P2 (extraction) → DEFER P3-P4.

The highest-value work is **clustering_utils.py** (DELETE entire file, 70% line reduction) and **functional_metrics.py** (55% size reduction, makes bootstrap reusable).

## Key Simplifications (No Backward Compatibility)

✅ **Simpler**: Delete old files entirely, no re-export boilerplate
✅ **Cleaner imports**: Direct imports from new modules
✅ **Less code**: No need for compatibility shims
✅ **Faster**: One-time import updates, then done

**Trade-off**: Requires updating ~10-20 import statements across the codebase (easily done with grep + sed).

## Output File Compatibility Guarantee

**CRITICAL**: All refactorings are internal code reorganization only. External behavior is unchanged:

### What Stays the Same

✅ **File names**: `clustered_results.jsonl`, `model_cluster_scores.json`, etc.
✅ **File formats**: JSONL, JSON, Parquet schemas unchanged
✅ **JSON keys**: All dictionary keys remain identical
✅ **JSON structure**: Nesting, data types, arrays all unchanged
✅ **DataFrame columns**: Column names and data types unchanged
✅ **Numeric precision**: All calculations produce identical results
✅ **WandB tables**: Same column structure and data

### What Can Change

✅ **Internal imports**: Module paths change (e.g., `clustering_utils` → `embeddings`)
✅ **Code organization**: Functions move between files
✅ **File sizes**: Individual source files get smaller

### Verification Strategy

Every refactoring must pass this test:
```bash
# Before refactor
python -m stringsight explain data.jsonl --output before/

# After refactor (with updated imports)
python -m stringsight explain data.jsonl --output after/

# Files must be identical
diff before/model_cluster_scores.json after/model_cluster_scores.json
# Should output: (no differences)

# Or use json comparison for robustness to formatting
python -c "
import json
with open('before/model_cluster_scores.json') as f1, open('after/model_cluster_scores.json') as f2:
    assert json.load(f1) == json.load(f2), 'JSON content differs!'
"
```

This ensures users can upgrade without any changes to their downstream processing, analysis scripts, or frontend display logic.
