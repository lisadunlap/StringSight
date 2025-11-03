#!/usr/bin/env python3
"""
Run StringSight pipeline from a YAML configuration.

This script loads a configuration file that specifies dataset path, output
directory, task description, and other pipeline arguments, and then delegates
to the shared `run_pipeline` function used by all dataset runners.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

# Add parent directory to path to import the shared runner
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf

from scripts.run_full_pipeline import run_pipeline


def _load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        config_path: Absolute or relative path to a YAML configuration file.

    Returns:
        A plain Python dictionary with configuration entries. Expected keys:
        - data_path: str, path to input file (.json, .jsonl, or .csv)
        - output_dir: str, directory for results
        - task_description: Optional[str]
        - method: Optional[str] in {"single_model", "side_by_side"}
        - min_cluster_size: Optional[int]
        - embedding_model: Optional[str]
        - extraction_model: Optional[str]
        - max_workers: Optional[int]
        - disable_wandb: Optional[bool]
        - quiet: Optional[bool]
        - sample_size: Optional[int]
        - groupby_column: Optional[str]
        - assign_outliers: Optional[bool]
        - model_a: Optional[str]
        - model_b: Optional[str]
        - models: Optional[List[str]] list of model names to include (filters input)
        - score_columns: Optional[List[str]] list of column names containing scores
    """
    conf = OmegaConf.load(config_path)
    return OmegaConf.to_container(conf, resolve=True)  # type: ignore[return-value]


def _merge_overrides(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Return a new dict where non-None overrides take precedence.

    Args:
        base: Baseline configuration.
        overrides: Potential overrides; keys with value None are ignored.

    Returns:
        A merged dictionary.
    """
    merged: Dict[str, Any] = dict(base)
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value
    return merged


def _bool_flag(value: bool) -> Optional[bool]:
    """Helper to convert CLI boolean flags to Optional[bool]."""
    return bool(value) if value else None


def main() -> Tuple[Any, Any]:
    """Main entrypoint to run the pipeline from a YAML configuration.

    Returns:
        A tuple of (clustered_df, model_stats) from the underlying pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Run StringSight from YAML configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_from_config.py \
      --config scripts/dataset_configs/safety.yaml

  # Override the data path and sample size at runtime
  python scripts/run_from_config.py \
      --config-name medi_qa \
      --data_path data/medhelm/medi_qa.jsonl \
      --sample_size 200
        """,
    )

    # Config selection: by explicit path or by name under scripts/dataset_configs
    config_group = parser.add_mutually_exclusive_group(required=False)
    config_group.add_argument("--config", type=str, help="Path to a YAML config file")
    config_group.add_argument(
        "--config-name",
        type=str,
        help="Name of a config YAML in scripts/dataset_configs (e.g., 'safety', 'medi_qa')",
    )
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="List available config names from scripts/dataset_configs and exit",
    )

    # Optional overrides mirror run_full_pipeline with a subset of common args
    parser.add_argument("--data_path", type=str, default=None, help="Override: dataset path (.json, .jsonl, or .csv)")
    parser.add_argument("--output_dir", type=str, default=None, help="Override: output directory")
    parser.add_argument(
        "--method",
        type=str,
        choices=["single_model", "side_by_side"],
        default=None,
        help="Override: analysis method",
    )
    parser.add_argument("--task_description", type=str, default=None, help="Override: task description")
    parser.add_argument("--no_task_description", action="store_true", help="Disable task description")
    parser.add_argument("--min_cluster_size", type=int, default=None, help="Override: min cluster size")
    parser.add_argument("--embedding_model", type=str, default=None, help="Override: embedding model")
    parser.add_argument("--extraction_model", type=str, default=None, help="Override: extraction model (default: gpt-4.1)")
    parser.add_argument("--max_workers", type=int, default=None, help="Override: parallel workers")
    parser.add_argument("--sample_size", type=int, default=None, help="Override: sample size")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable Weights & Biases logging (default: enabled)")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    parser.add_argument("--groupby_column", type=str, default=None, help="Override: stratified clustering column")
    parser.add_argument("--assign_outliers", action="store_true", help="Assign outliers to nearest clusters")
    parser.add_argument("--model_a", type=str, default=None, help="Model A when using tidy side-by-side input")
    parser.add_argument("--model_b", type=str, default=None, help="Model B when using tidy side-by-side input")
    parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        default=None,
        help=(
            "Optional list of model names to include. When provided and the input is in long/tidy format "
            "(column 'model' exists), the dataset will be filtered to only these models before analysis."
        ),
    )
    parser.add_argument(
        "--score_columns",
        nargs="+",
        type=str,
        default=None,
        help="Optional list of column names containing score metrics (e.g., accuracy, helpfulness)",
    )

    args = parser.parse_args()

    # Handle listing available configs and exit early
    if args.list_configs:
        configs_dir = Path(__file__).parent / "dataset_configs"
        if not configs_dir.exists():
            print("No configs directory found at:", configs_dir)
            sys.exit(0)
        names: List[str] = []
        for p in configs_dir.iterdir():
            if p.suffix in {".yaml", ".yml"}:
                names.append(p.stem)
        names = sorted(set(names))
        print("Available configs:")
        for name in names:
            print(f"  - {name}")
        sys.exit(0)

    # Resolve configuration path
    config_path: Optional[str] = None
    if args.config:
        config_path = args.config
    elif args.config_name:
        base_dir = Path(__file__).parent / "dataset_configs"
        candidate_yaml = base_dir / f"{args.config_name}.yaml"
        candidate_yml = base_dir / f"{args.config_name}.yml"
        if candidate_yaml.exists():
            config_path = str(candidate_yaml)
        elif candidate_yml.exists():
            config_path = str(candidate_yml)
        else:
            raise FileNotFoundError(
                f"Config name '{args.config_name}' not found in {base_dir}. "
                "Use --list-configs to see available options or provide --config with a path."
            )
    else:
        raise ValueError("One of --config or --config-name is required. Use --list-configs to discover names.")

    base_cfg = _load_config(config_path)

    # Determine final task description toggle
    task_desc: Optional[str]
    if args.no_task_description:
        task_desc = None
    else:
        task_desc = args.task_description if args.task_description is not None else base_cfg.get("task_description")

    overrides: Dict[str, Any] = {
        "data_path": args.data_path,
        "output_dir": args.output_dir,
        "method": args.method,
        "task_description": task_desc,
        "min_cluster_size": args.min_cluster_size,
        "embedding_model": args.embedding_model,
        "extraction_model": args.extraction_model,
        "max_workers": args.max_workers,
        "sample_size": args.sample_size,
        "disable_wandb": _bool_flag(args.disable_wandb),
        "quiet": _bool_flag(args.quiet),
        "groupby_column": args.groupby_column,
        "assign_outliers": _bool_flag(args.assign_outliers),
        "model_a": args.model_a,
        "model_b": args.model_b,
        "models": args.models,
        "score_columns": args.score_columns,
    }

    cfg = _merge_overrides(base_cfg, overrides)

    # Required fields validation
    data_path = cfg.get("data_path")
    output_dir = cfg.get("output_dir")
    if not data_path or not output_dir:
        raise ValueError("Both 'data_path' and 'output_dir' must be provided either in the YAML or via CLI.")

    # Map quiet flag to verbose for run_pipeline
    verbose = not bool(cfg.get("quiet", False))

    # Determine wandb toggle: default ON unless explicitly disabled via CLI/YAML
    use_wandb_flag = not bool(cfg.get("disable_wandb", False))

    clustered_df, model_stats = run_pipeline(
        data_path=data_path,
        output_dir=output_dir,
        method=cfg.get("method", "single_model"),
        system_prompt=None,
        task_description=cfg.get("task_description"),
        clusterer=cfg.get("clusterer", "hdbscan"),
        min_cluster_size=cfg.get("min_cluster_size", 15),
        embedding_model=cfg.get("embedding_model", "text-embedding-3-small"),
        extraction_model=cfg.get("extraction_model"),
        max_workers=cfg.get("max_workers", 64),
        use_wandb=use_wandb_flag,
        verbose=verbose,
        sample_size=cfg.get("sample_size"),
        groupby_column=cfg.get("groupby_column", "behavior_type"),
        assign_outliers=bool(cfg.get("assign_outliers", False)),
        model_a=cfg.get("model_a"),
        model_b=cfg.get("model_b"),
        filter_models=cfg.get("models"),
        score_columns=cfg.get("score_columns"),
    )

    return clustered_df, model_stats


if __name__ == "__main__":
    main()


