import pathlib
import pytest
import re

import pandas as pd

from stringsight.core.data_objects import PropertyDataset
from stringsight.metrics import SideBySideMetrics

DATASET_PATH = pathlib.Path("tests/outputs/hdbscan_clustered_results.parquet")


def _load_clustered_dataset() -> PropertyDataset:
    if not DATASET_PATH.exists():
        pytest.skip(f"Clustered dataset not found at {DATASET_PATH}. Run tests/test_hdbscan_clusterer.py first.")
    try:
        return PropertyDataset.load(DATASET_PATH, format="parquet")
    except Exception as e:
        pytest.skip(f"Failed to load clustered dataset: {e}")


def test_side_by_side_metrics_basic(tmp_path):
    # Ensure tmp_path is a pathlib.Path (pytest fixture may pass Path, __main__ passes str)
    from pathlib import Path as _Path
    if isinstance(tmp_path, str):
        tmp_path = _Path(tmp_path)

    dataset = _load_clustered_dataset()

    # Run metrics stage
    metrics_stage = SideBySideMetrics(output_dir=tmp_path)
    result = metrics_stage(dataset)

    assert result.model_stats, "model_stats should not be empty"

    # Pick first model
    model_name = next(iter(result.model_stats))
    stats_fine = result.model_stats[model_name]["fine"]
    assert stats_fine, "Fine metrics list should not be empty"

    # Verify schema of first ModelStats
    first = stats_fine[0]
    assert hasattr(first, "score") and isinstance(first.score, float)
    assert hasattr(first, "proportion")
    assert first.property_description, "Cluster label should not be empty"

    # Ensure score is finite
    assert first.score >= 0

    # Check that output files were written
    model_safe = re.sub(r"[^A-Za-z0-9._-]", "_", str(model_name).strip("()[]"))
    out_file = tmp_path / f"{model_safe}_fine_metrics.jsonl"
    assert out_file.exists(), "Metrics JSONL file should be written"

    # --- Debug prints ---------------------------------------------------
    print("\n--- Metrics summary (all models) ---")
    for model_name in result.model_stats:
        stats_fine = result.model_stats[model_name]["fine"]
        print(f"Model: {model_name}  |  fine clusters: {len(stats_fine)}")
        for ms in stats_fine[:3]:
            print(
                f"  • {ms.property_description[:60]:<60} | size={ms.size:<3} | prop={ms.proportion:.3f} | score={ms.score:.2f}"
            )
        if "coarse" in result.model_stats[model_name]:
            stats_coarse = result.model_stats[model_name]["coarse"]
            print(f"Coarse clusters: {len(stats_coarse)} (showing up to 3)")
            for ms in stats_coarse[:3]:
                print(
                    f"  • {ms.property_description[:60]:<60} | size={ms.size:<3} | prop={ms.proportion:.3f} | score={ms.score:.2f}"
                )
        print("\nMetrics files written to:", tmp_path) 

if __name__ == "__main__":
    test_side_by_side_metrics_basic(tmp_path="tests/outputs/side_by_side_metrics")