# StringSight Metrics System

The metrics system provides comprehensive model performance analysis across clusters of properties, supporting both pre-computed results analysis and dynamic UI-driven clustering workflows.

## Core Data Flow

```
Conversations (battles) → Properties → Clusters → Metrics
     ↓                      ↓            ↓         ↓
  question_id         property_id    cluster.label  model_cluster_scores
  (1:many)            (many:1)       (many:1)       cluster_scores  
                                                    model_scores
```

**Key Relationships:**
- **Battle/Conversation**: Single question-response interaction (identified by `question_id`)
- **Property**: Extracted behavioral characteristic (≥1 per conversation, has unique `property_id`)  
- **Cluster**: Group of similar properties with descriptive `label` (**clustering happens at property level**)
- **Metrics**: Performance analysis computed from property-cluster assignments using `cluster.label` as identifier

**Cluster Object Structure** (`stringsight.core.data_objects.Cluster`):
```python
@dataclass
class Cluster:
    id: str                           # fine cluster id (internal identifier)
    label: str                        # cluster name/description (used in metrics files)
    size: int                         # number of properties in cluster
    property_descriptions: List[str]  # property descriptions in the cluster
    property_ids: List[str]          # property ids in the cluster
    question_ids: List[str]          # conversation ids that have properties in cluster
    meta: Dict[str, Any]             # cluster metadata (e.g., {"group": "Positive"})
```

## Output File Structure

The metrics system produces 3 core data structures, available in both nested JSON and flattened JSONL formats:

### 1. MODEL_CLUSTER_SCORES - Per Model-Cluster Performance
- **Purpose**: Detailed metrics for each (model, cluster.label) combination
- **Frontend Usage**: Frequency plots, quality plots, model cards
- **Files**: `model_cluster_scores.json`, `model_cluster_scores_df.jsonl`

**JSON Structure:**
```json
{
  "model_name": {
    "cluster.label": {                               // Descriptive cluster name from Cluster.label
      "size": 42,                                    // Number of PROPERTIES in this model-cluster combo
      "proportion": 0.15,                            // Fraction of this model's properties in cluster
      "quality": {
        "helpfulness (1-5)": 3.2,                   // Raw quality scores averaged across properties
        "accuracy (0/1)": 0.85
      },
      "quality_delta": {
        "helpfulness (1-5)": 0.3,                   // Relative to model's overall average
        "accuracy (0/1)": 0.12
      },
      "proportion_delta": 0.08,                      // Salience: over/under-representation vs cross-model avg
      "metadata": {"group": "Positive (helpful)"},   // From Cluster.meta
      "examples": [                                  // Sample from Cluster.property_descriptions & question_ids
        ["conversation_id", conversation_metadata, property_metadata]
      ],
      
      // Optional: Bootstrap confidence intervals
      "proportion_ci": {"lower": 0.12, "upper": 0.18, "mean": 0.15},
      "quality_ci": {"helpfulness (1-5)": {"lower": 3.0, "upper": 3.4, "mean": 3.2}},
      "quality_delta_ci": {"helpfulness (1-5)": {"lower": 0.1, "upper": 0.5, "mean": 0.3}},
      "proportion_delta_ci": {"lower": 0.02, "upper": 0.14, "mean": 0.08},
      
      // Optional: Statistical significance flags
      "quality_delta_significant": {"helpfulness (1-5)": true, "accuracy (0/1)": false},
      "proportion_delta_significant": true
    }
  }
}
```

**JSONL Structure (Flattened):**
```json
{
  "model": "gpt-4-turbo",
  "cluster": "Provides comprehensive step-by-step solutions",  // Cluster.label
  "size": 42,
  "proportion": 0.15,
  "proportion_delta": 0.08,
  "quality_helpfulness_1_5": 3.2,                   // Sanitized metric names
  "quality_delta_helpfulness_1_5": 0.3,
  "proportion_ci_lower": 0.12,
  "quality_delta_helpfulness_1_5_significant": true
}
```

### 2. CLUSTER_SCORES - Per Cluster Aggregates
- **Purpose**: Cluster performance across all models
- **Frontend Usage**: Cluster analysis, cross-model comparison
- **Files**: `cluster_scores.json`, `cluster_scores_df.jsonl`

**Structure:**
```json
{
  "cluster.label": {                                 // Descriptive cluster name
    "size": 156,                                     // Total properties across all models in cluster
    "proportion": 0.08,                              // Fraction of all properties in this cluster
    "quality": {"helpfulness (1-5)": 3.5},          // Cross-model average quality for cluster
    "quality_delta": {"helpfulness (1-5)": 0.2},    // Relative to global average across all clusters
    "metadata": {"group": "Positive (helpful)"},     // From Cluster.meta
    "examples": [...]                                // Sample properties from all models in cluster
  }
}
```

### 3. MODEL_SCORES - Per Model Benchmarks  
- **Purpose**: Model performance across all clusters
- **Frontend Usage**: Benchmark table, model comparison
- **Files**: `model_scores.json`, `model_scores_df.jsonl`

**Structure:**
```json
{
  "model_name": {
    "size": 280,                                     // Total properties for model across all clusters
    "proportion": 1.0,                               // Always 1.0 (100% of model's properties)
    "quality": {"helpfulness (1-5)": 3.1},          // Average quality across all clusters
    "quality_delta": {"helpfulness (1-5)": -0.1},   // Relative to cross-model average
    "examples": [...]                                // Sample properties for model across all clusters
  }
}
```

## Key Metric Definitions

| Metric | Definition | Range | Interpretation |
|--------|------------|-------|----------------|
| **size** | Number of properties (NOT conversations) | ≥ 0 | Volume at property level |
| **proportion** | Fraction of parent set in this subset | [0, 1] | Frequency/coverage |
| **quality** | Raw quality scores (averaged across properties) | Varies | Absolute performance |
| **quality_delta** | Relative quality vs baseline | ℝ | +good, -bad, 0=average |
| **proportion_delta** | Salience (over/under-representation) | ℝ | +frequent, -rare, 0=average |
| ***_significant** | Statistical significance flag | Boolean | True if CI ∄ 0 |
| **unique_conversations** | Count of unique question_ids | ≥ 0 | Actual "battle" count |

**Important**: `size` in metrics = `len(Cluster.property_descriptions)`, while unique conversations = `len(set(Cluster.question_ids))`

## Current Issues & Bugs

### 🔴 CRITICAL - UI Clustering Metrics Gap
**Problem**: When users perform clustering in UI, no metrics are computed  
**Impact**: Metrics tab shows empty/incorrect data for UI-generated clusters  
**Root Cause**: Metrics system only works with pre-computed results files  
**Solution**: Add dynamic metrics computation endpoint that takes operational data  

**Code Location**: Need new `/metrics/compute-live` API endpoint  
**Priority**: P0 - Blocks UI metrics validation  

**Required Endpoint Structure**:
```python
@app.post("/metrics/compute-live")
def compute_live_metrics(
    conversations: List[ConversationRecord],
    properties: List[Property], 
    clusters: List[Cluster]
) -> Dict[str, Any]:
    # Returns: model_cluster_scores, cluster_scores, model_scores
```

### 🟡 HIGH - Battle Count Confusion  
**Problem**: `size` field represents property counts, but UI displays as "battles"  
**Impact**: Misleading conversation/battle counts in frontend displays  
**Root Cause**: Property-level clustering vs conversation-level reporting mismatch  
**Solution**: Add `unique_conversations` field and fix frontend labels  

**Code Locations**:
- `frontend_adapters.py:328` - `_calculate_total_battles()` tries to extract from examples
- Frontend metrics components - need to distinguish property counts vs battle counts
- Should use `len(set(Cluster.question_ids))` for actual battle counts

### 🟡 MEDIUM - Inconsistent Cluster Name Usage  
**Problem**: Code may inconsistently use `Cluster.id` vs `Cluster.label` as cluster identifier  
**Impact**: Potential mismatches between UI state and metrics computation  
**Solution**: Audit codebase to ensure consistent use of `Cluster.label` in metrics  

**Code Location**: Verify in `functional_metrics.py` data preparation logic

### 🟡 MEDIUM - Missing Bootstrap for UI  
**Problem**: Dynamic computation doesn't include confidence intervals  
**Impact**: Less statistical rigor for UI-generated results  
**Solution**: Extend live computation with optional bootstrap parameter  

### 🟢 LOW - Metric Name Sanitization  
**Problem**: Quality metric names with special chars get sanitized inconsistently  
**Impact**: Frontend key mismatches between JSON and JSONL formats  
**Code Location**: `data_transformers.py:23` - `sanitize_metric_name()`

## Fix Implementation Plan

### Phase 1: Critical UI Support (P0)
1. **Add Live Metrics Computation Endpoint**
   - Reuse `FunctionalMetrics.run()` logic with runtime data
   - Take `conversations`, `properties`, `clusters` from UI state
   - Return same 3-structure format as pre-computed results

2. **Frontend Integration**
   - Update `useMetricsData` hook to detect UI vs pre-computed data
   - Call live computation when metrics files don't exist

### Phase 2: Data Accuracy (P1)  
3. **Fix Battle Counting**
   - Use `len(set(cluster.question_ids))` for conversation counts
   - Keep `cluster.size` for property counts
   - Update frontend labels to distinguish property vs conversation metrics

4. **Validate Metric Calculations**
   - Ensure `cluster.label` used consistently as cluster identifier
   - Cross-check live computation vs saved results
   - Add integration tests for property → conversation aggregation

### Phase 3: Enhanced Features (P2)
5. **Bootstrap Support for Live Computation**  
   - Add optional `compute_bootstrap` parameter to live endpoint
   - Include confidence intervals and significance testing

6. **Performance Optimization**
   - Cache live metrics computation results
   - Add progress indicators for long computations

## Testing Strategy for UI Validation

### Quick Validation Checklist (Prioritized for Speed)
1. **✅ Load Pre-computed Results** 
   - Load `/results/omni_math_low/` → Navigate to metrics tab
   - Verify all 3 data structures render correctly
   - Check that cluster names match `Cluster.label` values

2. **✅ Battle Count Accuracy**
   - Compare displayed "battles" vs actual unique `question_ids`
   - Verify `size` shows property counts, separate field shows conversation counts

3. **✅ UI Clustering Flow** (Currently Broken - Priority Fix)
   - Upload data → Cluster in UI → Navigate to metrics tab
   - Should trigger live metrics computation and display results

4. **✅ Metric Value Validation**
   - Cross-check computed values with manual calculations
   - Verify `quality_delta` and `proportion_delta` formulas

### Test Data Requirements
- Use existing `/results/omni_math_low/` for pre-computed validation
- Multi-model dataset with known cluster assignments for UI testing
- Properties with quality scores for computation verification

## Development Notes

### Data Preparation for Live Metrics
```python
# From UI state (conversations, properties, clusters) to metrics input
df = prepare_metrics_dataframe(conversations, properties, clusters)
# Where cluster column = cluster.label, property info includes question_id mapping
```

### Bootstrap Configuration
```python
FunctionalMetrics(
    compute_bootstrap=True,
    bootstrap_samples=100,  # Balance accuracy vs speed
    log_to_wandb=True
)
```

### Frontend Data Flow
```
Pre-computed: Results Files → API → useMetricsData → MetricsTab
UI-generated: Operational Data → Live API → useMetricsData → MetricsTab
```

## Module Reference

| Module | Purpose | Key Classes |
|--------|---------|-------------|
| `functional_metrics.py` | Core metrics computation | `FunctionalMetrics` |
| `frontend_adapters.py` | UI data loading/formatting | `MetricsDataAdapter` | 
| `data_transformers.py` | Format conversion utilities | `flatten_*`, `sanitize_*` |
| `plotting.py` | Visualization generation | `generate_all_plots` |
| `cluster_subset.py` | Dynamic metrics helpers | `compute_subset_metrics` |

---

**Last Updated**: 2025-01-10  
**Status**: Under active development - UI metrics integration in progress