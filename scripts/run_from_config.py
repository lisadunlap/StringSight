#!/usr/bin/env python3
"""
Run StringSight pipeline from a YAML configuration.

This script loads a configuration file that specifies dataset path, output
directory, task description, and other pipeline arguments, and then delegates
to the shared `run_pipeline` function used by all dataset runners.
"""

import argparse
import sys
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

# Add parent directory to path to import the shared runner
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf
import pandas as pd

from scripts.run_full_pipeline import run_pipeline, load_dataset
from stringsight import label


def _load_taxonomy(taxonomy_spec: Any) -> Dict[str, str]:
    """Load taxonomy from a file path or dictionary.
    
    Args:
        taxonomy_spec: Either a string path to a JSON file, or a dictionary mapping
            label names to descriptions.
    
    Returns:
        Dictionary mapping label names to descriptions.
    """
    if isinstance(taxonomy_spec, str):
        # Assume it's a file path
        taxonomy_path = Path(taxonomy_spec)
        if not taxonomy_path.exists():
            raise FileNotFoundError(f"Taxonomy file not found: {taxonomy_path}")
        
        with open(taxonomy_path, 'r') as f:
            taxonomy = json.load(f)
        
        if not isinstance(taxonomy, dict):
            raise ValueError(f"Taxonomy file must contain a JSON object/dictionary, got {type(taxonomy)}")
        
        return taxonomy
    elif isinstance(taxonomy_spec, dict):
        # Already a dictionary
        return taxonomy_spec
    else:
        raise ValueError(f"Taxonomy must be a file path (str) or dictionary, got {type(taxonomy_spec)}")


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
        - taxonomy: Optional[str | Dict[str, str]] - path to JSON file or inline dict (enables label() mode)
        - min_cluster_size: Optional[int]
        - embedding_model: Optional[str]
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
        - extraction_model: Optional[str] model for property extraction
        - summary_model: Optional[str] model for cluster summarization
        - cluster_assignment_model: Optional[str] model for cluster matching
        - prompt_column: Optional[str] name of the prompt column (default: "prompt")
        - model_column: Optional[str] name of the model column for single_model (default: "model" if None)
        - model_response_column: Optional[str] name of the model response column for single_model (default: "model_response")
        - question_id_column: Optional[str] name of the question_id column (default: "question_id" if column exists)
        - model_a_column: Optional[str] name of the model_a column for side_by_side (default: "model_a")
        - model_b_column: Optional[str] name of the model_b column for side_by_side (default: "model_b")
        - model_a_response_column: Optional[str] name of the model_a_response column for side_by_side (default: "model_a_response")
        - model_b_response_column: Optional[str] name of the model_b_response column for side_by_side (default: "model_b_response")
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


def run_label_pipeline(
    data_path: str,
    output_dir: str,
    taxonomy: Dict[str, str],
    model_name: str = "gpt-4.1",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 2048,
    max_workers: int = 64,
    use_wandb: bool = True,
    verbose: bool = False,
    sample_size: Optional[int] = None,
    metrics_kwargs: Optional[Dict[str, Any]] = None,
    score_columns: Optional[List[str]] = None,
    prompt_column: str = "prompt",
    model_column: Optional[str] = None,
    model_response_column: Optional[str] = None,
    question_id_column: Optional[str] = None,
    extraction_cache_dir: Optional[str] = None,
    metrics_cache_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Run the label pipeline (fixed-taxonomy analysis).
    
    Args:
        data_path: Path to input dataset
        output_dir: Output directory for results
        taxonomy: Dictionary mapping label names to descriptions
        model_name: LLM model for labeling
        temperature: Temperature for LLM
        top_p: Top-p for LLM
        max_tokens: Max tokens for LLM
        max_workers: Max parallel workers
        use_wandb: Whether to log to wandb
        verbose: Whether to print progress
        sample_size: Optional sample size
        metrics_kwargs: Additional metrics configuration
        score_columns: Optional list of score column names
        prompt_column: Name of prompt column
        model_column: Name of model column
        model_response_column: Name of model response column
        question_id_column: Name of question_id column
        extraction_cache_dir: Cache directory for extraction
        metrics_cache_dir: Cache directory for metrics
    
    Returns:
        Tuple of (clustered_df, model_stats)
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    df = load_dataset(data_path, method="single_model")
    
    # Attach filename for wandb naming
    df.name = os.path.basename(data_path)
    
    if verbose:
        print(f"Loaded dataset with {len(df)} rows")
        if sample_size:
            print(f"Will sample to {sample_size} rows (sample_size type: {type(sample_size)}, value: {sample_size})")
        else:
            print(f"No sampling requested (sample_size: {sample_size})")
    
    # Run label pipeline (label() will handle sampling via validate_and_prepare_dataframe)
    if verbose:
        print(f"Calling label() with sample_size={sample_size} (type: {type(sample_size)})")
    
    clustered_df, model_stats = label(
        df,
        taxonomy=taxonomy,
        model_name=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        max_workers=max_workers,
        use_wandb=use_wandb,
        verbose=verbose,
        output_dir=str(output_path),
        sample_size=sample_size,  # Pass through to label() which handles sampling
        score_columns=score_columns,
        prompt_column=prompt_column,
        model_column=model_column,
        model_response_column=model_response_column,
        question_id_column=question_id_column,
        metrics_kwargs=metrics_kwargs,
        extraction_cache_dir=extraction_cache_dir,
        metrics_cache_dir=metrics_cache_dir,
    )
    
    return clustered_df, model_stats


def main() -> Tuple[Any, Any]:
    """Main entrypoint to run the pipeline from a YAML configuration.

    Supports both explain() and label() modes. If 'taxonomy' is provided in the config
    or via CLI, runs label() mode (fixed-taxonomy analysis). Otherwise runs explain()
    mode (clustering-based analysis).

    Returns:
        A tuple of (clustered_df, model_stats) from the underlying pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Run StringSight from YAML configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run explain() mode (clustering-based analysis)
  python scripts/run_from_config.py \
      --config scripts/dataset_configs/safety.yaml

  # Override the data path and sample size at runtime
  python scripts/run_from_config.py \
      --config-name medi_qa \
      --data_path data/medhelm/medi_qa.jsonl \
      --sample_size 200

  # Run label() mode (fixed-taxonomy analysis) with taxonomy from JSON file
  python scripts/run_from_config.py \
      --config scripts/dataset_configs/my_config.yaml \
      --taxonomy path/to/taxonomy.json

  # Run label() mode with inline taxonomy in YAML config
  python scripts/run_from_config.py \
      --config scripts/dataset_configs/my_config.yaml \
      --taxonomy inline
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
    parser.add_argument("--extraction_model", type=str, default=None, help="Override: model for property extraction (e.g., gpt-4.1)")
    parser.add_argument("--summary_model", type=str, default=None, help="Override: model for cluster summarization (e.g., gpt-4.1)")
    parser.add_argument("--cluster_assignment_model", type=str, default=None, help="Override: model for cluster matching (e.g., gpt-4.1-mini)")
    parser.add_argument("--prompt_column", type=str, default=None, help="Override: name of the prompt column (default: 'prompt')")
    parser.add_argument("--model_column", type=str, default=None, help="Override: name of the model column for single_model (default: 'model' if None)")
    parser.add_argument("--model_response_column", type=str, default=None, help="Override: name of the model response column for single_model (default: 'model_response')")
    parser.add_argument("--question_id_column", type=str, default=None, help="Override: name of the question_id column (default: 'question_id' if column exists)")
    parser.add_argument("--model_a_column", type=str, default=None, help="Override: name of the model_a column for side_by_side (default: 'model_a')")
    parser.add_argument("--model_b_column", type=str, default=None, help="Override: name of the model_b column for side_by_side (default: 'model_b')")
    parser.add_argument("--model_a_response_column", type=str, default=None, help="Override: name of the model_a_response column for side_by_side (default: 'model_a_response')")
    parser.add_argument("--model_b_response_column", type=str, default=None, help="Override: name of the model_b_response column for side_by_side (default: 'model_b_response')")
    
    # Label function parameters
    parser.add_argument(
        "--taxonomy",
        type=str,
        default=None,
        help=(
            "Path to JSON file containing taxonomy (dict mapping label names to descriptions), "
            "or 'inline' to use taxonomy from YAML config. When provided, enables label() mode instead of explain()."
        ),
    )
    parser.add_argument("--label_temperature", type=float, default=None, help="Override: temperature for label() LLM (default: 0.0)")
    parser.add_argument("--label_top_p", type=float, default=None, help="Override: top_p for label() LLM (default: 1.0)")
    parser.add_argument("--label_max_tokens", type=int, default=None, help="Override: max_tokens for label() LLM (default: 2048)")

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

    # Handle taxonomy: can come from CLI or config
    taxonomy_spec = args.taxonomy if args.taxonomy is not None else base_cfg.get("taxonomy")
    
    # Determine if we're in label() mode
    use_label_mode = taxonomy_spec is not None
    
    overrides: Dict[str, Any] = {
        "data_path": args.data_path,
        "output_dir": args.output_dir,
        "method": args.method,
        "task_description": task_desc,
        "min_cluster_size": args.min_cluster_size,
        "embedding_model": args.embedding_model,
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
        "extraction_model": args.extraction_model,
        "summary_model": args.summary_model,
        "cluster_assignment_model": args.cluster_assignment_model,
        "prompt_column": args.prompt_column,
        "model_column": args.model_column,
        "model_response_column": args.model_response_column,
        "question_id_column": args.question_id_column,
        "model_a_column": args.model_a_column,
        "model_b_column": args.model_b_column,
        "model_a_response_column": args.model_a_response_column,
        "model_b_response_column": args.model_b_response_column,
        "taxonomy": taxonomy_spec,
        "label_temperature": args.label_temperature,
        "label_top_p": args.label_top_p,
        "label_max_tokens": args.label_max_tokens,
    }

    cfg = _merge_overrides(base_cfg, overrides)

    # Handle legacy 'response_column' alias -> 'model_response_column'
    if "response_column" in cfg and "model_response_column" not in cfg:
        cfg["model_response_column"] = cfg.pop("response_column")

    # Required fields validation
    data_path = cfg.get("data_path")
    output_dir = cfg.get("output_dir")
    if not data_path or not output_dir:
        raise ValueError("Both 'data_path' and 'output_dir' must be provided either in the YAML or via CLI.")

    # Map quiet flag to verbose
    verbose = not bool(cfg.get("quiet", False))

    # Determine wandb toggle: default ON unless explicitly disabled via CLI/YAML
    use_wandb_flag = not bool(cfg.get("disable_wandb", False))


    # Route to label() or explain() based on taxonomy presence
    if use_label_mode:
        # Load taxonomy
        taxonomy_spec = cfg.get("taxonomy")
        if isinstance(taxonomy_spec, dict):
            # Taxonomy is already a dict (from YAML config)
            taxonomy = taxonomy_spec
        elif taxonomy_spec == "inline":
            # User specified --taxonomy inline, so taxonomy should be in base config as dict
            taxonomy = base_cfg.get("taxonomy")
            if not isinstance(taxonomy, dict):
                raise ValueError("When --taxonomy=inline, taxonomy must be provided as a dictionary in the YAML config")
        else:
            # Assume it's a file path (string)
            taxonomy = _load_taxonomy(taxonomy_spec)
        
        if verbose:
            print(f"Running label() mode with taxonomy: {list(taxonomy.keys())}")
        
        # Extract label-specific parameters
        model_name = cfg.get("extraction_model") or cfg.get("model_name", "gpt-4.1")
        temperature = cfg.get("label_temperature", 0.0)
        top_p = cfg.get("label_top_p", 1.0)
        max_tokens = cfg.get("label_max_tokens", 2048)
        
        if verbose:
            print(f"Label model (model_name): {model_name}")

        # Extract metrics_kwargs if provided
        metrics_kwargs = cfg.get("metrics_kwargs")
        
        # Extract and convert sample_size to int if provided
        sample_size_raw = cfg.get("sample_size")
        sample_size = int(sample_size_raw) if sample_size_raw is not None else None
        
        if verbose and sample_size:
            print(f"Sample size from config: {sample_size_raw} (type: {type(sample_size_raw)}) -> converted to: {sample_size}")
        
        clustered_df, model_stats = run_label_pipeline(
            data_path=data_path,
            output_dir=output_dir,
            taxonomy=taxonomy,
            model_name=model_name,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            max_workers=cfg.get("max_workers", 64),
            use_wandb=use_wandb_flag,
            verbose=verbose,
            sample_size=sample_size,
            metrics_kwargs=metrics_kwargs,
            score_columns=cfg.get("score_columns"),
            prompt_column=cfg.get("prompt_column", "prompt"),
            model_column=cfg.get("model_column"),
            model_response_column=cfg.get("model_response_column"),
            question_id_column=cfg.get("question_id_column"),
            extraction_cache_dir=cfg.get("extraction_cache_dir"),
            metrics_cache_dir=cfg.get("metrics_cache_dir"),
        )
    else:
        # Standard explain() mode
        if verbose:
            print("Running explain() mode (clustering-based analysis)")
            print("Effective model configuration:")
            print(f"  - extraction_model: {cfg.get('extraction_model')}")
            print(f"  - summary_model: {cfg.get('summary_model')}")
            print(f"  - cluster_assignment_model: {cfg.get('cluster_assignment_model')}")
        
        clustered_df, model_stats = run_pipeline(
            data_path=data_path,
            output_dir=output_dir,
            method=cfg.get("method", "single_model"),
            system_prompt=None,
            task_description=cfg.get("task_description"),
            clusterer=cfg.get("clusterer", "hdbscan"),
            min_cluster_size=cfg.get("min_cluster_size", 15),
            embedding_model=cfg.get("embedding_model", "text-embedding-3-large"),
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
            extraction_model=cfg.get("extraction_model"),
            summary_model=cfg.get("summary_model"),
            cluster_assignment_model=cfg.get("cluster_assignment_model"),
            prompt_column=cfg.get("prompt_column", "prompt"),
            model_column=cfg.get("model_column"),
            model_response_column=cfg.get("model_response_column"),
            question_id_column=cfg.get("question_id_column"),
            model_a_column=cfg.get("model_a_column"),
            model_b_column=cfg.get("model_b_column"),
            model_a_response_column=cfg.get("model_a_response_column"),
            model_b_response_column=cfg.get("model_b_response_column"),
        )

    return clustered_df, model_stats


if __name__ == "__main__":
    main()


