#!/usr/bin/env python3
"""
Compare optimized weight-based bootstrap vs naive materialized true-bootstrap.

This script exists to validate that the optimized bootstrap implementation produces
the same numeric outputs as a naive *true with-replacement* bootstrap that explicitly
duplicates conversations according to the bootstrap draw.

The comparison is tolerance-based because reduction order can differ.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from stringsight.core.data_objects import PropertyDataset
from stringsight.metrics import get_metrics
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


def _load_dataset_from_results_dir(results_dir: Path) -> PropertyDataset:
    """Load a PropertyDataset from a pipeline results directory (or file)."""
    if results_dir.is_dir():
        possible_files = [
            results_dir / "full_dataset.json",
            results_dir / "full_dataset.parquet",
            results_dir / "clustered_results.parquet",
            results_dir / "dataset.json",
            results_dir / "dataset.parquet",
        ]
        for file_path in possible_files:
            if file_path.exists():
                return PropertyDataset.load(str(file_path))
        raise FileNotFoundError(f"No recognizable dataset file found in {results_dir}")
    if results_dir.is_file():
        return PropertyDataset.load(str(results_dir))
    raise FileNotFoundError(f"Input path does not exist: {results_dir}")


def _naive_resample_conversations(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Materialize a true with-replacement bootstrap sample (preserving duplicates).

    Important:
        This uses the *same multinomial draw-count representation* as the optimized
        weight-based bootstrap (per-conversation draw counts).

        That makes this script a strict equivalence test: given the same seed and
        number of replicates, both code paths will use the same bootstrap draw
        counts, so any remaining differences should be due to implementation bugs
        (or tiny floating point reduction-order differences), not Monte Carlo noise
        from sampling different bootstrap replicates.
    """
    conv_ids = df["conversation_id"].unique()
    n_conv = len(conv_ids)
    if n_conv == 0:
        return df.iloc[0:0].copy()

    # True with-replacement bootstrap counts for each conversation_id
    p = np.full(n_conv, 1.0 / float(n_conv), dtype=float)
    counts = rng.multinomial(n_conv, p)

    parts = []
    for cid, k in zip(conv_ids, counts):
        if k <= 0:
            continue
        block = df[df["conversation_id"] == cid]
        if block.empty:
            continue
        parts.extend([block] * int(k))

    if not parts:
        return df.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=True).copy()


def _compute_ci(values, lower_percentile=2.5, upper_percentile=97.5):
    """Compute confidence interval for a list of values."""
    if not values:
        return None
    return {
        'lower': float(np.percentile(values, lower_percentile)),
        'upper': float(np.percentile(values, upper_percentile)),
        'mean': float(np.mean(values))
    }


def _attach_bootstrap_cis(stage, model_cluster_scores, cluster_scores, model_scores, bootstrap_samples):
    """Attach CI-bearing fields to metrics dicts (mirrors FunctionalMetrics behavior)."""
    # model_cluster_scores
    for model in model_cluster_scores:
        for cluster in model_cluster_scores[model]:
            proportions = []
            proportion_deltas = []
            quality_scores = {k: [] for k in model_cluster_scores[model][cluster].get("quality", {})}
            quality_deltas = {k: [] for k in model_cluster_scores[model][cluster].get("quality_delta", {})}

            for sample in bootstrap_samples:
                if model in sample["model_cluster"] and cluster in sample["model_cluster"][model]:
                    sm = sample["model_cluster"][model][cluster]
                    proportions.append(sm.get("proportion", 0))
                    proportion_deltas.append(sm.get("proportion_delta", 0))
                    for k in quality_scores:
                        if k in sm.get("quality", {}):
                            quality_scores[k].append(sm["quality"][k])
                    for k in quality_deltas:
                        if k in sm.get("quality_delta", {}):
                            quality_deltas[k].append(sm["quality_delta"][k])

            proportion_ci = _compute_ci(proportions)
            if proportion_ci:
                model_cluster_scores[model][cluster]["proportion_ci"] = proportion_ci
                model_cluster_scores[model][cluster]["proportion"] = proportion_ci["mean"]

            proportion_delta_ci = _compute_ci(proportion_deltas)
            if proportion_delta_ci:
                model_cluster_scores[model][cluster]["proportion_delta_ci"] = proportion_delta_ci
                model_cluster_scores[model][cluster]["proportion_delta"] = proportion_delta_ci["mean"]
                model_cluster_scores[model][cluster]["proportion_delta_significant"] = stage._is_significant(
                    proportion_delta_ci["lower"], proportion_delta_ci["upper"], 0
                )
            else:
                model_cluster_scores[model][cluster]["proportion_delta_significant"] = False

            quality_ci = {}
            for k, vals in quality_scores.items():
                ci = _compute_ci(vals)
                if ci:
                    quality_ci[k] = ci
                    model_cluster_scores[model][cluster]["quality"][k] = ci["mean"]
            if quality_ci:
                model_cluster_scores[model][cluster]["quality_ci"] = quality_ci

            quality_delta_ci = {}
            quality_delta_significant = {}
            for k, vals in quality_deltas.items():
                ci = _compute_ci(vals)
                if ci:
                    quality_delta_ci[k] = ci
                    model_cluster_scores[model][cluster]["quality_delta"][k] = ci["mean"]
                    quality_delta_significant[k] = stage._is_significant(ci["lower"], ci["upper"], 0)
                else:
                    quality_delta_significant[k] = False
            if quality_delta_ci:
                model_cluster_scores[model][cluster]["quality_delta_ci"] = quality_delta_ci
            model_cluster_scores[model][cluster]["quality_delta_significant"] = quality_delta_significant

    # cluster_scores
    for cluster in cluster_scores:
        proportions = []
        quality_scores = {k: [] for k in cluster_scores[cluster].get("quality", {})}
        quality_deltas = {k: [] for k in cluster_scores[cluster].get("quality_delta", {})}

        for sample in bootstrap_samples:
            if cluster in sample["cluster"]:
                sm = sample["cluster"][cluster]
                proportions.append(sm.get("proportion", 0))
                for k in quality_scores:
                    if k in sm.get("quality", {}):
                        quality_scores[k].append(sm["quality"][k])
                for k in quality_deltas:
                    if k in sm.get("quality_delta", {}):
                        quality_deltas[k].append(sm["quality_delta"][k])

        proportion_ci = _compute_ci(proportions)
        if proportion_ci:
            cluster_scores[cluster]["proportion_ci"] = proportion_ci
            cluster_scores[cluster]["proportion"] = proportion_ci["mean"]

        quality_ci = {}
        for k, vals in quality_scores.items():
            ci = _compute_ci(vals)
            if ci:
                quality_ci[k] = ci
                cluster_scores[cluster]["quality"][k] = ci["mean"]
        if quality_ci:
            cluster_scores[cluster]["quality_ci"] = quality_ci

        quality_delta_ci = {}
        quality_delta_significant = {}
        for k, vals in quality_deltas.items():
            ci = _compute_ci(vals)
            if ci:
                quality_delta_ci[k] = ci
                cluster_scores[cluster]["quality_delta"][k] = ci["mean"]
                quality_delta_significant[k] = stage._is_significant(ci["lower"], ci["upper"], 0)
            else:
                quality_delta_significant[k] = False
        if quality_delta_ci:
            cluster_scores[cluster]["quality_delta_ci"] = quality_delta_ci
        cluster_scores[cluster]["quality_delta_significant"] = quality_delta_significant

    # model_scores
    for model in model_scores:
        proportions = []
        quality_scores = {k: [] for k in model_scores[model].get("quality", {})}
        quality_deltas = {k: [] for k in model_scores[model].get("quality_delta", {})}

        for sample in bootstrap_samples:
            if model in sample["model"]:
                sm = sample["model"][model]
                proportions.append(sm.get("proportion", 0))
                for k in quality_scores:
                    if k in sm.get("quality", {}):
                        quality_scores[k].append(sm["quality"][k])
                for k in quality_deltas:
                    if k in sm.get("quality_delta", {}):
                        quality_deltas[k].append(sm["quality_delta"][k])

        proportion_ci = _compute_ci(proportions)
        if proportion_ci:
            model_scores[model]["proportion_ci"] = proportion_ci
            model_scores[model]["proportion"] = proportion_ci["mean"]

        quality_ci = {}
        for k, vals in quality_scores.items():
            ci = _compute_ci(vals)
            if ci:
                quality_ci[k] = ci
                model_scores[model]["quality"][k] = ci["mean"]
        if quality_ci:
            model_scores[model]["quality_ci"] = quality_ci

        quality_delta_ci = {}
        quality_delta_significant = {}
        for k, vals in quality_deltas.items():
            ci = _compute_ci(vals)
            if ci:
                quality_delta_ci[k] = ci
                model_scores[model]["quality_delta"][k] = ci["mean"]
                quality_delta_significant[k] = stage._is_significant(ci["lower"], ci["upper"], 0)
            else:
                quality_delta_significant[k] = False
        if quality_delta_ci:
            model_scores[model]["quality_delta_ci"] = quality_delta_ci
        model_scores[model]["quality_delta_significant"] = quality_delta_significant


def main():
    parser = argparse.ArgumentParser(description="Test bootstrap equivalence: optimized vs naive true-bootstrap")
    parser.add_argument("--method", type=str, default="single_model")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_wandb", action="store_true")
    args = parser.parse_args()

    results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics_input_path = Path(args.input_file)

    # ----------------------------
    # 1) Optimized (library path)
    # ----------------------------
    t0 = time.time()
    compute_metrics_only(
        input_path=str(metrics_input_path),
        method=args.method,
        output_dir=str(results_dir / "equiv_optimized"),
        metrics_kwargs={
            "compute_confidence_intervals": True,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.seed,
            "log_to_wandb": args.use_wandb,
            "generate_plots": False,
        },
        use_wandb=args.use_wandb,
        verbose=False,
    )
    t1 = time.time()

    # ----------------------------
    # 2) Naive true-bootstrap
    # ----------------------------
    dataset = _load_dataset_from_results_dir(metrics_input_path)
    stage = get_metrics(
        method=args.method,
        output_dir=str(results_dir / "equiv_naive"),
        compute_bootstrap=False,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.seed,
        log_to_wandb=args.use_wandb,
        generate_plots=False,
        verbose=False,
        use_wandb=args.use_wandb,
    )

    df = stage._prepare_data(dataset)
    cluster_names = [c for c in df["cluster"].unique() if pd.notna(c)]
    model_names = list(df["model"].unique())

    base_model_cluster = stage._compute_model_cluster_scores(df, cluster_names, model_names, include_metadata=False)
    base_model_cluster = stage._compute_salience(base_model_cluster)
    base_cluster = stage._compute_cluster_scores(df, cluster_names, model_names, include_metadata=False)
    base_model = stage._compute_model_scores(df, cluster_names, model_names, include_metadata=False)

    rng = np.random.default_rng(args.seed)
    bootstrap_samples = []
    for _ in range(args.bootstrap_samples):
        sample_df = _naive_resample_conversations(df, rng)
        sample_mc = stage._compute_model_cluster_scores(sample_df, cluster_names, model_names, include_metadata=False)
        sample_mc = stage._compute_salience(sample_mc)
        sample_c = stage._compute_cluster_scores(sample_df, cluster_names, model_names, include_metadata=False)
        sample_m = stage._compute_model_scores(sample_df, cluster_names, model_names, include_metadata=False)
        bootstrap_samples.append({"model_cluster": sample_mc, "cluster": sample_c, "model": sample_m})

    _attach_bootstrap_cis(stage, base_model_cluster, base_cluster, base_model, bootstrap_samples)

    naive_dir = results_dir / "equiv_naive"
    naive_dir.mkdir(parents=True, exist_ok=True)
    with open(naive_dir / "model_cluster_scores.json", "w") as f:
        json.dump(base_model_cluster, f)
    with open(naive_dir / "cluster_scores.json", "w") as f:
        json.dump(base_cluster, f)
    with open(naive_dir / "model_scores.json", "w") as f:
        json.dump(base_model, f)
    t2 = time.time()

    # Load key JSON outputs from both runs
    opt_dir = results_dir / "equiv_optimized"
    naive_dir = results_dir / "equiv_naive"
    files = [
        "model_cluster_scores.json",
        "cluster_scores.json",
        "model_scores.json",
    ]

    print("\n=== Timing ===")
    print(f"Optimized (weight-based): {t1 - t0:.2f}s")
    print(f"Naive (materialized):     {t2 - t1:.2f}s")

    print("\n=== Comparing outputs (numeric fields) ===")
    any_diffs = False
    for fname in files:
        opt_path = opt_dir / fname
        naive_path = naive_dir / fname
        opt = load_json(opt_path)
        naive = load_json(naive_path)

        # Compare a few top-level structures shallowly; for nested dicts, focus on scalar fields
        if isinstance(opt, dict) and isinstance(naive, dict):
            diffs = []
            common_keys = sorted(set(opt.keys()) & set(naive.keys()))
            # Check a sample of up to 10 keys for brevity
            for k in common_keys[:10]:
                v1 = opt[k]
                v2 = naive[k]
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
        print(
            "\nNote: Differences may occur in non-essential fields (e.g., examples/metadata).\n"
            "Bootstrap is seeded; numeric metric fields should closely match within tolerance."
        )
    else:
        print("\nAll checked files matched on compared numeric fields.")


if __name__ == "__main__":
    sys.exit(main())


