import pandas as pd
import json

from stringsight.core.data_objects import PropertyDataset
from lmmvibes import explain, label

fixed_labels = {
  "sarcastic": "response contains sarcasm",
  "safety-conscious": "response is safety-conscious",
  "nuclear world domination": "response mentions nuclear world domination",
}

########################################################
# Test explain
########################################################
df = pd.read_json("data/test_data.jsonl", lines=True)
clustered_df, model_stats = explain(
    df,
    method="single_model",
    min_cluster_size=10,
    output_dir="results/test"
)

########################################################
# Test label with fixed labels
########################################################

df = pd.read_json("data/test_data.jsonl", lines=True)
clustered_df, model_stats = label(
    df,
    taxonomy=fixed_labels,
    output_dir="results/test_fixed_labels"
)