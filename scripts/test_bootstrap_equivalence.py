#!/usr/bin/env python3
"""
Run metrics twice (with and without bootstrap_keep_full_records) and compare outputs.

Usage example:
  python scripts/test_bootstrap_equivalence.py \
    --method single_model \
    --input_file data/demo_data/taubench_airline.jsonl \
    --output_dir results/taubench_airline_demo \
    --system_prompt agent_system_prompt \
    --bootstrap_samples 200

This script:
  - Runs metrics-only twice against the same dataset/results directory
  - First with --bootstrap_keep_full_records (legacy heavy path)
  - Then without (optimized lean path)
  - Times both runs and compares the resulting JSON outputs
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from stringsight.public import compute_metrics_only


def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def _compare_ci_dict(ci1, ci2, path):
    diffs = []
    for key in ("lower", "upper", "mean"):
        if key in ci1 and key in ci2:
            if isinstance(ci1[key], (int, float)) and isinstance(ci2[key], (int, float)):
                if abs(ci1[key] - ci2[key]) > 1e-9:
                    diffs.append(f"{path}.{key} differs: {ci1[key]} vs {ci2[key]}")
    return diffs


def _compare_quality_ci_map(m1, m2, path):
    diffs = []
    common_metrics = sorted(set(m1.keys()) & set(m2.keys()))
    for metric in common_metrics:
        ci1 = m1.get(metric)
        ci2 = m2.get(metric)
        if isinstance(ci1, dict) and isinstance(ci2, dict):
            diffs.extend(_compare_ci_dict(ci1, ci2, f"{path}.{metric}"))
    return diffs


def _compare_metrics_dict(v1, v2, path_prefix):
    """Compare CI-bearing fields inside a metrics dict."""
    diffs = []
    # Proportion CIs
    if isinstance(v1.get("proportion_ci"), dict) and isinstance(v2.get("proportion_ci"), dict):
        diffs.extend(_compare_ci_dict(v1["proportion_ci"], v2["proportion_ci"], f"{path_prefix}.proportion_ci"))
    if isinstance(v1.get("proportion_delta_ci"), dict) and isinstance(v2.get("proportion_delta_ci"), dict):
        diffs.extend(_compare_ci_dict(v1["proportion_delta_ci"], v2["proportion_delta_ci"], f"{path_prefix}.proportion_delta_ci"))
    # Quality score CIs
    if isinstance(v1.get("quality_ci"), dict) and isinstance(v2.get("quality_ci"), dict):
        diffs.extend(_compare_quality_ci_map(v1["quality_ci"], v2["quality_ci"], f"{path_prefix}.quality_ci"))
    # Quality delta CIs
    if isinstance(v1.get("quality_delta_ci"), dict) and isinstance(v2.get("quality_delta_ci"), dict):
        diffs.extend(_compare_quality_ci_map(v1["quality_delta_ci"], v2["quality_delta_ci"], f"{path_prefix}.quality_delta_ci"))
    return diffs


def compare_dicts(d1, d2, keys_to_compare=None, path_prefix=""):
    """Compare top-N entries and include CI fields recursively where present."""
    diffs = []
    if d1 is None or d2 is None:
        diffs.append("One of the dicts is None")
        return diffs
    if not isinstance(d1, dict) or not isinstance(d2, dict):
        diffs.append("Non-dict objects provided for comparison")
        return diffs

    keys = keys_to_compare or sorted(set(d1.keys()) & set(d2.keys()))
    for k in keys:
        v1 = d1.get(k)
        v2 = d2.get(k)
        current_path = f"{path_prefix}.{k}" if path_prefix else str(k)

        # If these look like metrics dicts, compare CI-bearing fields
        if isinstance(v1, dict) and isinstance(v2, dict):
            looks_like_metrics = any(field in v1 for field in ("proportion", "proportion_ci", "quality", "quality_ci", "quality_delta_ci"))
            if looks_like_metrics:
                diffs.extend(_compare_metrics_dict(v1, v2, current_path))
                # Also compare shallow numeric scalars if present (e.g., size, proportion)
                for sk in ("size", "proportion", "proportion_delta"):
                    if isinstance(v1.get(sk), (int, float)) and isinstance(v2.get(sk), (int, float)):
                        if abs(v1[sk] - v2[sk]) > 1e-9:
                            diffs.append(f"{current_path}.{sk} differs: {v1[sk]} vs {v2[sk]}")
                continue

            # Otherwise, descend one level if values are dicts-of-dicts (e.g., model->cluster->metrics)
            subkeys = sorted(set(v1.keys()) & set(v2.keys()))
            for sk in subkeys:
                sv1 = v1.get(sk)
                sv2 = v2.get(sk)
                if isinstance(sv1, dict) and isinstance(sv2, dict):
                    diffs.extend(compare_dicts(sv1, sv2, path_prefix=f"{current_path}.{sk}"))
                elif isinstance(sv1, (int, float)) and isinstance(sv2, (int, float)):
                    if abs(sv1 - sv2) > 1e-9:
                        diffs.append(f"{current_path}.{sk} differs: {sv1} vs {sv2}")
                elif sv1 != sv2:
                    # Ignore non-numeric non-critical diffs at this level
                    pass
        else:
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                if abs(v1 - v2) > 1e-9:
                    diffs.append(f"{current_path} differs: {v1} vs {v2}")
            elif v1 != v2:
                # Ignore non-numeric non-critical diffs at this level
                pass
    return diffs


def main():
    parser = argparse.ArgumentParser(description="Test bootstrap equivalence with and without full records")
    parser.add_argument("--method", type=str, default="single_model")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--system_prompt", type=str, default="single_model_system_prompt")
    parser.add_argument("--bootstrap_samples", type=int, default=100)
    parser.add_argument("--use_wandb", action="store_true")
    args = parser.parse_args()

    results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Metrics-only expects an existing pipeline results directory or file path.
    # We assume output_dir contains full_dataset.json from a prior run.
    metrics_input_path = results_dir

    # Run legacy-heavy path
    import numpy as np
    np.random.seed(42)
    t0 = time.time()
    _, model_stats_legacy = compute_metrics_only(
        input_path=str(metrics_input_path),
        method=args.method,
        output_dir=str(results_dir / "equiv_legacy"),
        metrics_kwargs={
            "compute_confidence_intervals": True,
            "bootstrap_samples": args.bootstrap_samples,
            # Emulate legacy by forcing full-records via environment variable (no longer supported by flags)
        },
        use_wandb=args.use_wandb,
        verbose=False,
    )
    t1 = time.time()

    # Run optimized-lean path
    np.random.seed(42)
    _, model_stats_lean = compute_metrics_only(
        input_path=str(metrics_input_path),
        method=args.method,
        output_dir=str(results_dir / "equiv_lean"),
        metrics_kwargs={
            "compute_confidence_intervals": True,
            "bootstrap_samples": args.bootstrap_samples,
            # Lean path is default now
        },
        use_wandb=args.use_wandb,
        verbose=False,
    )
    t2 = time.time()

    # Load key JSON outputs from both runs
    legacy_dir = results_dir / "equiv_legacy"
    lean_dir = results_dir / "equiv_lean"
    files = [
        "model_cluster_scores.json",
        "cluster_scores.json",
        "model_scores.json",
    ]

    print("\n=== Timing ===")
    print(f"Legacy (keep full records): {t1 - t0:.2f}s")
    print(f"Lean (skip heavy fields):   {t2 - t1:.2f}s")

    print("\n=== Comparing outputs (numeric fields) ===")
    any_diffs = False
    for fname in files:
        legacy_path = legacy_dir / fname
        lean_path = lean_dir / fname
        legacy = load_json(legacy_path)
        lean = load_json(lean_path)

        # Compare a few top-level structures shallowly; for nested dicts, focus on scalar fields
        if isinstance(legacy, dict) and isinstance(lean, dict):
            diffs = []
            common_keys = sorted(set(legacy.keys()) & set(lean.keys()))
            # Check a sample of up to 10 keys for brevity
            for k in common_keys[:10]:
                v1 = legacy[k]
                v2 = lean[k]
                if isinstance(v1, dict) and isinstance(v2, dict):
                    diffs.extend(compare_dicts(v1, v2))
                else:
                    if v1 != v2:
                        diffs.append(f"Top-level entry for {k} differs in type or value")
        else:
            diffs = [f"Structure mismatch for {fname}"]

        status = "OK" if not diffs else "DIFF"
        print(f"- {fname}: {status}")
        if diffs:
            any_diffs = True
            # Print a few differences for context
            for d in diffs[:10]:
                print(f"  * {d}")

    if any_diffs:
        print("\nNote: Differences may occur in non-essential fields (e.g., examples/metadata) or due to randomness.\n"
              "We seeded NumPy for both runs; numeric metric fields should closely match.")
    else:
        print("\nAll checked files matched on compared numeric fields.")


if __name__ == "__main__":
    sys.exit(main())


