# Custom Pipelines

Learn how to build custom analysis pipelines with StringSight's modular architecture.

## Pipeline Architecture

StringSight uses a 4-stage pipeline where each stage operates on a `PropertyDataset` object:

```
PropertyDataset → Extraction → Post-processing → Clustering → Metrics → PropertyDataset
```

Each stage:
- Inherits from `PipelineStage`
- Implements a `run(dataset: PropertyDataset) -> PropertyDataset` method
- Can be configured independently
- Supports caching to avoid recomputation

## Basic Custom Pipeline

### Using PipelineBuilder

```python
from stringsight.pipeline import PipelineBuilder
from stringsight.extractors import OpenAIExtractor
from stringsight.postprocess import LLMJsonParser, PropertyValidator
from stringsight.clusterers import HDBSCANClusterer
from stringsight.metrics import SingleModelMetrics

# Build custom pipeline
pipeline = (PipelineBuilder("My Custom Pipeline")
    .extract_properties(
        OpenAIExtractor(
            model="gpt-4o-mini",
            temperature=0.5
        )
    )
    .parse_properties(LLMJsonParser())
    .validate_properties(PropertyValidator())
    .cluster_properties(
        HDBSCANClusterer(
            min_cluster_size=5,
            embedding_model="all-MiniLM-L6-v2"
        )
    )
    .compute_metrics(SingleModelMetrics())
    .configure(
        use_wandb=True,
        wandb_project="custom-analysis"
    )
    .build())

# Use with explain()
from stringsight import explain

clustered_df, model_stats = explain(
    df,
    custom_pipeline=pipeline
)
```

### Manual Pipeline Construction

```python
from stringsight.pipeline import Pipeline
from stringsight.core import PropertyDataset

# Create dataset from DataFrame
dataset = PropertyDataset.from_dataframe(df, method="single_model")

# Initialize pipeline
pipeline = Pipeline("Manual Pipeline")

# Add stages
pipeline.add_stage(OpenAIExtractor(model="gpt-4.1"))
pipeline.add_stage(LLMJsonParser())
pipeline.add_stage(PropertyValidator())
pipeline.add_stage(HDBSCANClusterer(min_cluster_size=15))
pipeline.add_stage(SingleModelMetrics())

# Run pipeline
result_dataset = pipeline.run(dataset)

# Extract results
clustered_df = result_dataset.to_dataframe()
model_stats = result_dataset.model_stats
```

## Custom Extractors

Create custom property extractors by inheriting from `PipelineStage`:

```python
from stringsight.core.stage import PipelineStage
from stringsight.core.data_objects import PropertyDataset, Property
from typing import List
import anthropic

class ClaudeExtractor(PipelineStage):
    """Custom extractor using Anthropic's Claude API."""

    def __init__(self, model: str = "claude-3-opus-20240229", **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.client = anthropic.Anthropic()

    def run(self, data: PropertyDataset) -> PropertyDataset:
        """Extract properties using Claude."""
        properties = []

        for conv in data.conversations:
            # Build prompt
            prompt = f"Analyze this response and identify key behavioral properties:\n\n{conv.responses}"

            # Call Claude
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )

            # Parse response and create Property objects
            prop = Property(
                id=f"{conv.question_id}_prop",
                question_id=conv.question_id,
                model=conv.model,
                property_description=response.content[0].text,
                raw_response=response.content[0].text
            )
            properties.append(prop)

        data.properties = properties
        return data

# Use custom extractor
pipeline = (PipelineBuilder("Claude Pipeline")
    .extract_properties(ClaudeExtractor(model="claude-3-sonnet-20240229"))
    .cluster_properties(HDBSCANClusterer())
    .build())
```

## Custom Clusterers

Create custom clustering strategies:

```python
from stringsight.clusterers.base import BaseClusterer
from sklearn.cluster import DBSCAN
import numpy as np

class DBSCANClusterer(BaseClusterer):
    """Custom clusterer using DBSCAN algorithm."""

    def __init__(self, eps: float = 0.5, min_samples: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.min_samples = min_samples

    def run(self, data: PropertyDataset) -> PropertyDataset:
        """Cluster properties using DBSCAN."""
        # Get embeddings
        embeddings = self.generate_embeddings(data)

        # Apply DBSCAN
        clusterer = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = clusterer.fit_predict(embeddings)

        # Create clusters
        data.clusters = self.create_clusters_from_labels(
            data,
            labels,
            cluster_name_prefix="DBSCAN"
        )

        return data

# Use custom clusterer
pipeline = (PipelineBuilder("DBSCAN Pipeline")
    .extract_properties(OpenAIExtractor())
    .cluster_properties(DBSCANClusterer(eps=0.3, min_samples=10))
    .build())
```

## Custom Metrics

Implement custom metric calculations:

```python
from stringsight.metrics.base import BaseMetrics
from collections import defaultdict

class CustomMetrics(BaseMetrics):
    """Custom metrics calculator."""

    def run(self, data: PropertyDataset) -> PropertyDataset:
        """Calculate custom metrics."""
        metrics = defaultdict(dict)

        for cluster in data.clusters:
            for model in data.all_models:
                # Get properties for this model-cluster combination
                model_props = [
                    p for p in cluster.property_ids
                    if data.get_property(p).model == model
                ]

                # Calculate custom metrics
                metrics[model][cluster.label] = {
                    "count": len(model_props),
                    "proportion": len(model_props) / len(cluster.property_ids),
                    "custom_score": self.calculate_custom_score(model_props)
                }

        data.model_stats = dict(metrics)
        return data

    def calculate_custom_score(self, properties):
        """Your custom scoring logic."""
        # Example: average property description length
        return sum(len(p.property_description) for p in properties) / len(properties)

# Use custom metrics
pipeline = (PipelineBuilder("Custom Metrics Pipeline")
    .extract_properties(OpenAIExtractor())
    .cluster_properties(HDBSCANClusterer())
    .compute_metrics(CustomMetrics())
    .build())
```

## Advanced Configurations

### Multi-Stage Extraction

Combine multiple extraction strategies:

```python
class MultiStageExtractor(PipelineStage):
    """Run multiple extractors in sequence."""

    def __init__(self, extractors: List[PipelineStage]):
        super().__init__()
        self.extractors = extractors

    def run(self, data: PropertyDataset) -> PropertyDataset:
        """Run all extractors and combine results."""
        all_properties = []

        for extractor in self.extractors:
            # Run extractor
            result = extractor.run(data)
            all_properties.extend(result.properties)

        data.properties = all_properties
        return data

# Use multi-stage extraction
pipeline = (PipelineBuilder("Multi-Stage Pipeline")
    .extract_properties(
        MultiStageExtractor([
            OpenAIExtractor(model="gpt-4.1", temperature=0.3),
            OpenAIExtractor(model="gpt-4o-mini", temperature=0.7)
        ])
    )
    .cluster_properties(HDBSCANClusterer())
    .build())
```

### Conditional Processing

Add conditional logic to pipeline stages:

```python
class ConditionalStage(PipelineStage):
    """Apply different processing based on conditions."""

    def __init__(self, condition_fn, true_stage, false_stage):
        super().__init__()
        self.condition_fn = condition_fn
        self.true_stage = true_stage
        self.false_stage = false_stage

    def run(self, data: PropertyDataset) -> PropertyDataset:
        """Run stage based on condition."""
        if self.condition_fn(data):
            return self.true_stage.run(data)
        else:
            return self.false_stage.run(data)

# Example: Use different clusterers based on dataset size
def is_large_dataset(data):
    return len(data.conversations) > 1000

pipeline = Pipeline("Conditional Pipeline")
pipeline.add_stage(OpenAIExtractor())
pipeline.add_stage(
    ConditionalStage(
        condition_fn=is_large_dataset,
        true_stage=HDBSCANClusterer(min_cluster_size=50),
        false_stage=HDBSCANClusterer(min_cluster_size=10)
    )
)
```

### Caching & Checkpoints

Enable caching to save intermediate results:

```python
# Built-in caching
clustered_df, model_stats = explain(
    df,
    extraction_cache_dir=".cache/extraction",
    clustering_cache_dir=".cache/clustering",
    metrics_cache_dir=".cache/metrics"
)

# Manual checkpointing
dataset = PropertyDataset.from_dataframe(df, method="single_model")

# Run extraction
extractor = OpenAIExtractor(cache_dir=".cache/extraction")
dataset = extractor.run(dataset)

# Save checkpoint
dataset.save("checkpoint_after_extraction.json")

# Later: Load checkpoint
dataset = PropertyDataset.load("checkpoint_after_extraction.json")

# Continue pipeline
clusterer = HDBSCANClusterer()
dataset = clusterer.run(dataset)
```

## Example: Domain-Specific Pipeline

Build a pipeline for analyzing customer support conversations:

```python
from stringsight import explain
from stringsight.extractors import FixedAxesLabeler

# Define customer support taxonomy
SUPPORT_TAXONOMY = {
    "empathetic_response": "Response shows empathy and emotional intelligence",
    "policy_adherence": "Response correctly follows company policies",
    "problem_resolution": "Response effectively solves the customer's issue",
    "clear_communication": "Response is easy to understand and well-structured",
    "proactive_assistance": "Response anticipates and addresses related concerns"
}

# Create pipeline
from stringsight.pipeline import PipelineBuilder

pipeline = (PipelineBuilder("Customer Support Analysis")
    .extract_properties(
        FixedAxesLabeler(
            taxonomy=SUPPORT_TAXONOMY,
            model="gpt-4.1"
        )
    )
    .configure(
        use_wandb=True,
        wandb_project="customer-support-analysis"
    )
    .build())

# Run analysis
clustered_df, model_stats = explain(
    support_df,
    custom_pipeline=pipeline,
    output_dir="results/customer_support"
)

# Analyze results by taxonomy category
for category, details in model_stats.items():
    print(f"\n{category}:")
    print(f"  Coverage: {details['proportion']:.2%}")
    print(f"  Quality: {details['quality']}")
```

## Testing Custom Stages

Write tests for custom pipeline stages:

```python
import pytest
from stringsight.core.data_objects import PropertyDataset, ConversationRecord

def test_custom_extractor():
    """Test custom extractor."""
    # Create test data
    conv = ConversationRecord(
        question_id="test_1",
        prompt="Test prompt",
        model="gpt-4",
        responses="Test response",
        scores={},
        meta={}
    )
    dataset = PropertyDataset(
        conversations=[conv],
        all_models=["gpt-4"],
        properties=[],
        clusters=[],
        model_stats={}
    )

    # Run extractor
    extractor = ClaudeExtractor()
    result = extractor.run(dataset)

    # Assertions
    assert len(result.properties) > 0
    assert result.properties[0].question_id == "test_1"
    assert result.properties[0].property_description is not None

def test_custom_clusterer():
    """Test custom clusterer."""
    # Setup test data with properties
    # ... create dataset with properties

    # Run clusterer
    clusterer = DBSCANClusterer(eps=0.3, min_samples=5)
    result = clusterer.run(dataset)

    # Assertions
    assert len(result.clusters) > 0
    assert all(c.size >= 5 for c in result.clusters)

# Run tests
pytest.main([__file__, "-v"])
```

## Best Practices

1. **Inherit from base classes** - Use `PipelineStage`, `BaseClusterer`, `BaseMetrics`
2. **Implement `run()` method** - Main entry point for your stage
3. **Return PropertyDataset** - Always return the dataset object (modified or unchanged)
4. **Add logging** - Use `self.logger` for debugging
5. **Support caching** - Implement cache_dir parameter for expensive operations
6. **Document parameters** - Add docstrings explaining all configuration options
7. **Test thoroughly** - Write unit tests for custom stages
8. **Handle errors gracefully** - Add try/except with informative error messages

## Next Steps

- **[Performance Tuning](performance.md)** - Optimize your custom pipelines
- **[API Reference](../api/reference.md)** - Detailed API documentation
- **[Contributing](../development/contributing.md)** - Share your custom stages
