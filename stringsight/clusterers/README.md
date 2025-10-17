# Clusterers

This package contains clustering stages that group properties into coherent categories.

## BaseClusterer (contract)

`BaseClusterer` defines a minimal interface for clustering implementations:

- `cluster(data: PropertyDataset, column_name: str) -> pd.DataFrame`
  - Must return a DataFrame with standardized columns:
    - `question_id`
    - `{column_name}` (default: `property_description`)
    - `{column_name}_cluster_id`
    - `{column_name}_cluster_label`
- `postprocess_clustered_df(df, column_name) -> pd.DataFrame` (optional)
  - Hook to adjust labels (e.g., move tiny clusters to `Outliers`).
- `get_config() -> Any`
  - Returns a config object with fields used for saving/logging.
- `run(data, column_name="property_description") -> PropertyDataset`
  - Orchestrates the pipeline: cluster → postprocess → build clusters → add "No properties" → attach attributes → save.
- `save(df, clusters) -> Dict[str, str]`
  - Persists artifacts via shared utilities.
- `add_no_properties_cluster(data, clusters) -> None`
  - Appends a synthetic "No properties" cluster **only when _no_ properties were extracted for the entire dataset**.  
    If at least one property exists, the function is a no-op, so conversations for which extraction failed remain unclustered rather than being grouped into the global bucket.

## Implementing a new clusterer

Create a subclass in this folder and implement the `cluster` method.

```python
from __future__ import annotations

import pandas as pd
from stringsight.clusterers.base import BaseClusterer
from stringsight.core.data_objects import PropertyDataset

class MyClusterer(BaseClusterer):
    def __init__(self, *, output_dir: str | None = None, **kwargs):
        super().__init__(output_dir=output_dir, **kwargs)
        # Optionally set self.config with fields used in saving/logging
        self.config = type("Config", (), {
            "min_cluster_size": 10,
            "embedding_model": "my-model",
            "assign_outliers": False,
            "use_wandb": False,
            "wandb_project": None,
            "disable_dim_reduction": False,
            "min_samples": 5,
            "cluster_selection_epsilon": 0.0,
        })()

    def cluster(self, data: PropertyDataset, column_name: str) -> pd.DataFrame:
        # Build and return a DataFrame with the standardized columns listed above
        raise NotImplementedError
```

### Notes

- Keep `column_name` defaulting to `"property_description"` to integrate with the rest of the pipeline.
- Use `postprocess_clustered_df` for small, policy-like tweaks instead of embedding them in `cluster`.
- The base class will handle building `Cluster` objects, backfilling `cluster_*` onto properties, saving artifacts, and (optionally) adding the "No properties" cluster when applicable. 