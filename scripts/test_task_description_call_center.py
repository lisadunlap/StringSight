#!/usr/bin/env python3
"""
Quick test runner for task_description-enabled pipeline on a 20-row sample of
data/demo_data/call_center.jsonl with bootstrap disabled.
"""

import os
from pathlib import Path

from run_full_pipeline import run_pipeline


def main():
    data_path = str(Path("data/demo_data/call_center.jsonl").resolve())
    output_dir = str(Path("results/test_task_description_call_center").resolve())

    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    task_description = (
        "The task is summarizing call-center conversations for IT support. Please focus on if any crucial information is missing, if any information is made up, and what specific aspects of the call are being focused on in the summary."
    )

    # Disable bootstrap via metrics_kwargs
    metrics_kwargs = {"compute_confidence_intervals": False}

    run_pipeline(
        data_path=data_path,
        output_dir=output_dir,
        method="single_model",
        system_prompt=None,  # allow explain() to build from task_description
        task_description=task_description,
        clusterer="hdbscan",
        min_cluster_size=2,
        max_coarse_clusters=12,
        embedding_model="text-embedding-3-small",
        hierarchical=False,
        max_workers=16,
        use_wandb=False,
        verbose=True,
        sample_size=20,
        metrics_kwargs=metrics_kwargs,
    )


if __name__ == "__main__":
    main()


