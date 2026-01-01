"""
stringsight.metrics.single_model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Functional metrics wrapper for single-model datasets.

This class now delegates all computation to the functional metrics pipeline
(`stringsight.metrics.functional_metrics.FunctionalMetrics`) and exists mainly as
an ergonomic/compatibility alias so existing code importing
`SingleModelMetrics` continues to work.
"""

from __future__ import annotations

from typing import Any

from .functional_metrics import FunctionalMetrics


class SingleModelMetrics(FunctionalMetrics):
    """Metrics stage for single-model data using functional metrics.

    Notes:
    - Uses the functional outputs: `model_cluster_scores.json`, `cluster_scores.json`, `model_scores.json`
    - Confidence intervals and significance are provided via the functional bootstrap when enabled
    - Plots and wandb logging behavior are inherited from the functional base
    """

    def __init__(
        self,
        output_dir: str | None = None,
        # Backward-compat arg name for users of the old BaseMetrics
        compute_confidence_intervals: bool | None = None,
        # Functional name (preferred)
        compute_bootstrap: bool | None = None,
        bootstrap_samples: int = 100,
        bootstrap_seed: int | None = None,
        log_to_wandb: bool = True,
        generate_plots: bool = True,
        **kwargs: Any,
    ) -> None:
        # Map legacy arg name to functional if provided
        if compute_bootstrap is None and compute_confidence_intervals is not None:
            compute_bootstrap = compute_confidence_intervals
        # Default to True (functional pipeline default)
        if compute_bootstrap is None:
            compute_bootstrap = True

        super().__init__(
            output_dir=output_dir,
            compute_bootstrap=compute_bootstrap,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            log_to_wandb=log_to_wandb,
            generate_plots=generate_plots,
            **kwargs,
        ) 