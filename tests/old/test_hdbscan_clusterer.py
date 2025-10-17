import pytest
import pathlib
import pandas as pd

from stringsight.clusterers.hdbscan import HDBSCANClusterer
from stringsightght.core.data_objects import PropertyDataset


DATASET_PATH = pathlib.Path("tests/outputs/arena_first50_properties.json")


def _load_saved_dataset() -> PropertyDataset:
    """Load the PropertyDataset produced by the property-extraction test."""
    if not DATASET_PATH.exists():
        pytest.skip(f"Expected dataset not found at {DATASET_PATH}. Run tests/test_arena_subset.py first.")
    try:
        return PropertyDataset.load(DATASET_PATH, format="json")
    except Exception as e:
        pytest.skip(f"Failed to load dataset ({e}); skipping clustering test.")


def test_hdbscan_clusterer_basic():
    dataset = _load_saved_dataset()

    # Ensure we have some properties with descriptions; otherwise skip
    valid_props = [p for p in dataset.properties if p.property_description]
    if len(valid_props) < 3:
        pytest.skip("Not enough parsed properties to cluster.")

    clusterer = HDBSCANClusterer(min_cluster_size=3, hierarchical=True, include_embeddings=False, max_coarse_clusters=10)
    result = clusterer(dataset)

    # Save clustering results to file
    result.save("tests/outputs/hdbscan_clustered_results.parquet", format="parquet")

    # Convert clustered PropertyDataset to DataFrame
    df = result.to_dataframe(type="clusters")
    print("results cols ", df.columns)

    # Save the enriched DataFrame
    df.to_json("tests/outputs/property_with_clusters.jsonl", orient="records", lines=True)
    print("\n--- Property DataFrame with Clusters (first 5 rows) ---")
    print(df.head())

    # Each property now has a cluster id
    for p in result.properties:
        # only check properties that had descriptions
        if p.property_description:
            assert hasattr(p, "cluster_id")

    # --- Debug: print a summary of the clusters (first 5) ---
    print("\n--- Fine-grained clusters (up to 3 shown) ---")
    # for c in result.clusters[:3]:
    #     print(f"Cluster {c.id} | size={c.size} | label='{c.label}' | parent_id={c.parent_id}")

    # print 3 examples of each cluster
    for c in result.clusters[:3]:
        print(f"Cluster {c.id} | size={c.size} | label='{c.label}' | parent_id={c.parent_id}")
        for p in c.property_descriptions:
            print(f"  {p}")

if __name__ == "__main__":
    test_hdbscan_clusterer_basic()