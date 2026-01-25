"""
bootstrap.py

Bootstrap statistical analysis for metrics confidence intervals.
Provides fast with-replacement bootstrap over conversation IDs using weight-based resampling.
"""

from __future__ import annotations

from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np


class BootstrapAnalyzer:
    """Bootstrap statistical analysis for metrics confidence intervals.

    This class implements a true with-replacement bootstrap over conversation IDs,
    using per-conversation weights (draw counts) for performance instead of
    materializing duplicated DataFrames.
    """

    def __init__(self, samples: int = 100, seed: int | None = None):
        """Initialize bootstrap analyzer.

        Args:
            samples: Number of bootstrap samples to generate
            seed: Random seed for reproducibility (optional)
        """
        self.samples = samples
        self.seed = seed

    @staticmethod
    def _bootstrap_salience_from_proportions(
        proportions: "Any", *, n_models: int
    ) -> "Any":
        """Compute per-model proportion deltas (salience) from proportions.

        This is the bootstrap analogue of `_compute_salience()`, but operates on a dense
        array instead of nested dicts.

        Args:
            proportions: Array of shape (n_models, n_clusters) with proportions in [0, 1].
            n_models: Number of models (first dimension).

        Returns:
            Array of shape (n_models, n_clusters) with `proportion_delta`.
        """
        if n_models <= 1:
            return np.zeros_like(proportions, dtype=float)

        totals = np.sum(proportions, axis=0, keepdims=True)
        avg_others = (totals - proportions) / float(n_models - 1)
        return proportions - avg_others

    @staticmethod
    def _compute_weighted_means_1d(
        *,
        group_idx: "Any",
        n_groups: int,
        weights: "Any",
        values: "Any",
    ) -> "Any":
        """Compute weighted means per group with NaN-safe handling.

        Args:
            group_idx: int array of shape (n_rows,) mapping each row to a group in [0, n_groups).
            n_groups: number of groups.
            weights: float array of shape (n_rows,) (non-negative weights).
            values: float array of shape (n_rows, n_metrics) with NaN for missing values.

        Returns:
            means: float array of shape (n_groups, n_metrics). If a group has no valid entries
                   for a metric (denominator 0), the mean is NaN (distinguishing missing from real zero).
        """
        n_rows, n_metrics = values.shape
        if n_rows == 0:
            return np.full((n_groups, n_metrics), np.nan, dtype=float)

        means = np.full((n_groups, n_metrics), np.nan, dtype=float)
        for j in range(n_metrics):
            col = values[:, j]
            valid = ~np.isnan(col)
            if not np.any(valid):
                continue
            num = np.bincount(
                group_idx[valid],
                weights=(weights[valid] * col[valid]),
                minlength=n_groups,
            )
            den = np.bincount(group_idx[valid], weights=weights[valid], minlength=n_groups)
            means[:, j] = np.divide(num, den, out=np.full_like(num, np.nan, dtype=float), where=den > 0)
        return means

    @staticmethod
    def _compute_weighted_means_2d(
        *,
        row_idx: "Any",
        col_idx: "Any",
        n_rows: int,
        n_cols: int,
        weights: "Any",
        values: "Any",
    ) -> "Any":
        """Compute weighted means for a 2D grouping (row_idx, col_idx) with NaN-safe handling."""
        if len(values) == 0:
            return np.zeros((n_rows, n_cols, values.shape[1]), dtype=float)

        flat = row_idx * n_cols + col_idx
        n_groups = n_rows * n_cols
        means_flat = BootstrapAnalyzer._compute_weighted_means_1d(
            group_idx=flat,
            n_groups=n_groups,
            weights=weights,
            values=values,
        )
        return means_flat.reshape((n_rows, n_cols, values.shape[1]))

    @staticmethod
    def _bootstrap_scores_to_matrix(
        *,
        scores_series: pd.Series,
        metric_to_idx: Dict[str, int],
        n_metrics: int,
    ) -> "Any":
        """Convert a `scores` series into a dense matrix.

        Expected input format:
            - `scores_series` entries are `dict[str, number]` or non-dict / empty.
            - `metric_to_idx` maps metric keys to column indices.

        Output:
            - `np.ndarray` of shape (n_rows, n_metrics) with float values.
            - Missing metrics are encoded as NaN (so denominators ignore them).
        """
        mat = np.full((len(scores_series), n_metrics), np.nan, dtype=float)
        for i, s in enumerate(scores_series):
            if not isinstance(s, dict):
                continue
            for k, v in s.items():
                j = metric_to_idx.get(k)
                if j is None:
                    continue
                if isinstance(v, (int, float)):
                    mat[i, j] = float(v)
        return mat

    @staticmethod
    def _bootstrap_prepare(
        *,
        df: pd.DataFrame,
        cluster_names: List[str],
        model_names: List[str],
        metric_keys: List[str],
    ) -> Dict[str, Any]:
        """Prepare stable indices and dense score matrices for fast bootstrap.

        Returns a dict with:
            - model_names, cluster_names, metric_keys, and mapping dicts
            - conversation index and per-row conversation/model/cluster indices
            - deduplicated frames and dense score matrices
        """
        model_names = list(model_names)
        cluster_names = list(cluster_names)
        n_models = len(model_names)
        n_clusters = len(cluster_names)

        metric_to_idx = {k: i for i, k in enumerate(metric_keys)}
        model_to_idx = {m: i for i, m in enumerate(model_names)}
        cluster_to_idx = {c: i for i, c in enumerate(cluster_names)}

        # De-duplicate at the same levels the metric computations implicitly use.
        # - Denominators and model/global scores: unique per (conversation_id, model)
        # - Cluster numerators: unique per (conversation_id, model, cluster)
        df_cm = df.drop_duplicates(subset=["conversation_id", "model"]).copy()
        df_cmc = df.drop_duplicates(subset=["conversation_id", "model", "cluster"]).copy()

        conv_index = pd.Index(df["conversation_id"].unique())
        n_conv = len(conv_index)

        # Precompute row -> conversation/model/cluster indices
        cm_conv_idx = conv_index.get_indexer(df_cm["conversation_id"])
        cm_model_idx = np.array([model_to_idx.get(m, -1) for m in df_cm["model"]], dtype=int)

        cmc_conv_idx = conv_index.get_indexer(df_cmc["conversation_id"])
        cmc_model_idx = np.array([model_to_idx.get(m, -1) for m in df_cmc["model"]], dtype=int)
        cmc_cluster_idx = np.array([cluster_to_idx.get(c, -1) for c in df_cmc["cluster"]], dtype=int)

        # Filter unknown model/cluster rows (should not happen, but keep arrays aligned)
        cm_keep = cm_model_idx >= 0
        df_cm = df_cm.loc[cm_keep].reset_index(drop=True)
        cm_conv_idx = cm_conv_idx[cm_keep]
        cm_model_idx = cm_model_idx[cm_keep]

        cmc_keep = (cmc_model_idx >= 0) & (cmc_cluster_idx >= 0)
        df_cmc = df_cmc.loc[cmc_keep].reset_index(drop=True)
        cmc_conv_idx = cmc_conv_idx[cmc_keep]
        cmc_model_idx = cmc_model_idx[cmc_keep]
        cmc_cluster_idx = cmc_cluster_idx[cmc_keep]

        n_metrics = len(metric_keys)
        cm_scores = BootstrapAnalyzer._bootstrap_scores_to_matrix(
            scores_series=df_cm["scores"],
            metric_to_idx=metric_to_idx,
            n_metrics=n_metrics,
        )
        cmc_scores = BootstrapAnalyzer._bootstrap_scores_to_matrix(
            scores_series=df_cmc["scores"],
            metric_to_idx=metric_to_idx,
            n_metrics=n_metrics,
        )

        return {
            "model_names": model_names,
            "cluster_names": cluster_names,
            "metric_keys": metric_keys,
            "metric_to_idx": metric_to_idx,
            "model_to_idx": model_to_idx,
            "cluster_to_idx": cluster_to_idx,
            "n_models": n_models,
            "n_clusters": n_clusters,
            "n_metrics": n_metrics,
            "conv_index": conv_index,
            "n_conv": n_conv,
            "cm_conv_idx": cm_conv_idx,
            "cm_model_idx": cm_model_idx,
            "cmc_conv_idx": cmc_conv_idx,
            "cmc_model_idx": cmc_model_idx,
            "cmc_cluster_idx": cmc_cluster_idx,
            "cm_scores": cm_scores,
            "cmc_scores": cmc_scores,
        }

    @staticmethod
    def _bootstrap_allocate_arrays(*, S: int, n_models: int, n_clusters: int, n_metrics: int) -> Dict[str, "Any"]:
        """Allocate bootstrap result arrays."""
        return {
            "mc_prop": np.zeros((S, n_models, n_clusters), dtype=float),
            "mc_prop_delta": np.zeros((S, n_models, n_clusters), dtype=float),
            "mc_quality": np.zeros((S, n_models, n_clusters, n_metrics), dtype=float),
            "mc_quality_delta": np.zeros((S, n_models, n_clusters, n_metrics), dtype=float),
            "c_prop": np.zeros((S, n_clusters), dtype=float),
            "c_quality": np.zeros((S, n_clusters, n_metrics), dtype=float),
            "c_quality_delta": np.zeros((S, n_clusters, n_metrics), dtype=float),
            "m_prop": np.zeros((S, n_models), dtype=float),
            "m_quality": np.zeros((S, n_models, n_metrics), dtype=float),
            "m_quality_delta": np.zeros((S, n_models, n_metrics), dtype=float),
        }

    def _bootstrap_compute_one_replicate(
        self,
        *,
        prep: Dict[str, Any],
        arrays: Dict[str, Any],
        i: int,
        conv_weights: "Any",
    ) -> None:
        """Compute all bootstrap metrics for a single replicate and store into `arrays`."""
        n_models = prep["n_models"]
        n_clusters = prep["n_clusters"]
        n_metrics = prep["n_metrics"]

        # Row weights (by conversation)
        w_cm = conv_weights[prep["cm_conv_idx"]]
        w_cmc = conv_weights[prep["cmc_conv_idx"]]

        cm_model_idx = prep["cm_model_idx"]
        cmc_model_idx = prep["cmc_model_idx"]
        cmc_cluster_idx = prep["cmc_cluster_idx"]

        # ---- Denominators: by model, and global ----
        model_sizes = np.bincount(cm_model_idx, weights=w_cm, minlength=n_models)
        global_size = float(np.sum(w_cm))

        # Weighted mean scores per model and global
        model_means = self._compute_weighted_means_1d(
            group_idx=cm_model_idx,
            n_groups=n_models,
            weights=w_cm,
            values=prep["cm_scores"],
        )
        global_means = self._compute_weighted_means_1d(
            group_idx=np.zeros(len(prep["cm_scores"]), dtype=int),
            n_groups=1,
            weights=w_cm,
            values=prep["cm_scores"],
        )[0]

        # ---- Cluster-level numerators (across all models) ----
        cluster_sizes = np.bincount(cmc_cluster_idx, weights=w_cmc, minlength=n_clusters)
        cluster_means = self._compute_weighted_means_1d(
            group_idx=cmc_cluster_idx,
            n_groups=n_clusters,
            weights=w_cmc,
            values=prep["cmc_scores"],
        )

        # ---- Model-cluster numerators ----
        flat_mc = cmc_model_idx * n_clusters + cmc_cluster_idx
        mc_sizes_flat = np.bincount(flat_mc, weights=w_cmc, minlength=n_models * n_clusters)
        mc_sizes = mc_sizes_flat.reshape((n_models, n_clusters))
        mc_means = self._compute_weighted_means_2d(
            row_idx=cmc_model_idx,
            col_idx=cmc_cluster_idx,
            n_rows=n_models,
            n_cols=n_clusters,
            weights=w_cmc,
            values=prep["cmc_scores"],
        )

        # ---- Proportions ----
        with np.errstate(divide="ignore", invalid="ignore"):
            proportions = np.divide(
                mc_sizes,
                model_sizes.reshape((n_models, 1)),
                out=np.zeros_like(mc_sizes, dtype=float),
                where=model_sizes.reshape((n_models, 1)) > 0,
            )
        prop_delta = self._bootstrap_salience_from_proportions(proportions, n_models=n_models)

        arrays["mc_prop"][i] = proportions
        arrays["mc_prop_delta"][i] = prop_delta
        arrays["c_prop"][i] = (cluster_sizes / global_size) if global_size > 0 else np.zeros(n_clusters, dtype=float)
        arrays["m_prop"][i] = np.where(model_sizes > 0, 1.0, 0.0)

        # ---- Quality + deltas ----
        arrays["mc_quality"][i] = mc_means
        arrays["c_quality"][i] = cluster_means
        arrays["m_quality"][i] = model_means

        # Compute deltas: cluster/model - baseline
        # If baseline is NaN (missing data), delta is NaN
        # If baseline is 0.0 (real mean), delta is computed normally
        baseline_model = model_means[:, None, :]  # (n_models, 1, n_metrics)
        arrays["mc_quality_delta"][i] = mc_means - baseline_model
        arrays["c_quality_delta"][i] = cluster_means - global_means[None, :]

        # Model delta: compare each model to cross-model average (global_means)
        arrays["m_quality_delta"][i] = model_means - global_means[None, :]

    @staticmethod
    def _bootstrap_attach_results(
        *,
        prep: Dict[str, Any],
        arrays: Dict[str, Any],
        model_cluster_scores: Dict[str, Dict[str, Dict[str, Any]]],
        cluster_scores: Dict[str, Dict[str, Any]],
        model_scores: Dict[str, Dict[str, Any]],
    ) -> None:
        """Attach CI dicts + significance flags and replace point-estimates with bootstrap means."""
        metric_keys = prep["metric_keys"]
        model_names = prep["model_names"]
        cluster_names = prep["cluster_names"]

        def _ci_dict(arr: "Any") -> Dict[str, float]:
            return {
                "lower": float(np.percentile(arr, 2.5)),
                "upper": float(np.percentile(arr, 97.5)),
                "mean": float(np.mean(arr)),
            }

        def _is_significant(lower, upper, contains=0):
            """Check if CI excludes the reference value."""
            return not (lower <= contains <= upper)

        # Model-cluster
        for mi, model in enumerate(model_names):
            for ci, cluster in enumerate(cluster_names):
                mc = model_cluster_scores[model][cluster]

                ci_prop = _ci_dict(arrays["mc_prop"][:, mi, ci])
                mc["proportion_ci"] = ci_prop
                mc["proportion"] = ci_prop["mean"]

                ci_pd = _ci_dict(arrays["mc_prop_delta"][:, mi, ci])
                mc["proportion_delta_ci"] = ci_pd
                mc["proportion_delta"] = ci_pd["mean"]
                mc["proportion_delta_significant"] = _is_significant(ci_pd["lower"], ci_pd["upper"], 0)

                mc_q_ci: Dict[str, Dict[str, float]] = {}
                mc_qd_ci: Dict[str, Dict[str, float]] = {}
                mc_qd_sig: Dict[str, bool] = {}
                for mj, metric in enumerate(metric_keys):
                    ci_q = _ci_dict(arrays["mc_quality"][:, mi, ci, mj])
                    mc_q_ci[metric] = ci_q
                    mc["quality"][metric] = ci_q["mean"]

                    ci_qd = _ci_dict(arrays["mc_quality_delta"][:, mi, ci, mj])
                    mc_qd_ci[metric] = ci_qd
                    mc["quality_delta"][metric] = ci_qd["mean"]
                    mc_qd_sig[metric] = _is_significant(ci_qd["lower"], ci_qd["upper"], 0)

                if mc_q_ci:
                    mc["quality_ci"] = mc_q_ci
                if mc_qd_ci:
                    mc["quality_delta_ci"] = mc_qd_ci
                mc["quality_delta_significant"] = mc_qd_sig

        # Cluster scores
        for ci, cluster in enumerate(cluster_names):
            cs = cluster_scores[cluster]

            ci_prop = _ci_dict(arrays["c_prop"][:, ci])
            cs["proportion_ci"] = ci_prop
            cs["proportion"] = ci_prop["mean"]

            c_q_ci: Dict[str, Dict[str, float]] = {}
            c_qd_ci: Dict[str, Dict[str, float]] = {}
            c_qd_sig: Dict[str, bool] = {}
            for mj, metric in enumerate(metric_keys):
                ci_q = _ci_dict(arrays["c_quality"][:, ci, mj])
                c_q_ci[metric] = ci_q
                cs["quality"][metric] = ci_q["mean"]

                ci_qd = _ci_dict(arrays["c_quality_delta"][:, ci, mj])
                c_qd_ci[metric] = ci_qd
                cs["quality_delta"][metric] = ci_qd["mean"]
                c_qd_sig[metric] = _is_significant(ci_qd["lower"], ci_qd["upper"], 0)

            if c_q_ci:
                cs["quality_ci"] = c_q_ci
            if c_qd_ci:
                cs["quality_delta_ci"] = c_qd_ci
            cs["quality_delta_significant"] = c_qd_sig

        # Model scores
        for mi, model in enumerate(model_names):
            ms = model_scores[model]

            ci_prop = _ci_dict(arrays["m_prop"][:, mi])
            ms["proportion_ci"] = ci_prop
            ms["proportion"] = ci_prop["mean"]

            m_q_ci: Dict[str, Dict[str, float]] = {}
            m_qd_ci: Dict[str, Dict[str, float]] = {}
            m_qd_sig: Dict[str, bool] = {}
            for mj, metric in enumerate(metric_keys):
                ci_q = _ci_dict(arrays["m_quality"][:, mi, mj])
                m_q_ci[metric] = ci_q
                ms["quality"][metric] = ci_q["mean"]

                ci_qd = _ci_dict(arrays["m_quality_delta"][:, mi, mj])
                m_qd_ci[metric] = ci_qd
                ms["quality_delta"][metric] = ci_qd["mean"]
                m_qd_sig[metric] = _is_significant(ci_qd["lower"], ci_qd["upper"], 0)

            if m_q_ci:
                ms["quality_ci"] = m_q_ci
            if m_qd_ci:
                ms["quality_delta_ci"] = m_qd_ci
            ms["quality_delta_significant"] = m_qd_sig

    def add_bootstrap_analysis(
        self,
        df: pd.DataFrame,
        model_cluster_scores: Dict[str, Dict[str, Dict[str, Any]]],
        cluster_scores: Dict[str, Dict[str, Any]],
        model_scores: Dict[str, Dict[str, Any]],
        *,
        cluster_names: List[str],
        model_names: List[str],
        metric_keys: List[str],
        progress_callback=None,
        log_fn=None,
    ) -> Tuple[Dict[str, Dict[str, Dict[str, Any]]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """Add bootstrap confidence intervals and significance testing.

        This implementation is a **true with-replacement bootstrap** over `conversation_id`s.
        To make it fast, it uses per-conversation **weights** (draw counts) instead of
        materializing a duplicated DataFrame for each replicate.

        Args:
            df: Long dataframe with columns: conversation_id, model, cluster, scores
            model_cluster_scores: Nested dict of model -> cluster -> metrics
            cluster_scores: Dict of cluster -> metrics
            model_scores: Dict of model -> metrics
            cluster_names: List of cluster names
            model_names: List of model names
            metric_keys: List of metric names
            progress_callback: Optional callback for progress updates
            log_fn: Optional logging function

        Returns:
            Tuple of (model_cluster_scores, cluster_scores, model_scores) with CIs added

        Behavior:
            - Always uses exactly `samples` replicates (no skipping).
            - Empty subsets (e.g., a model gets 0 draws) yield empty metrics (zeros).
            - Point estimates are set to bootstrap means.
        """
        if log_fn:
            log_fn(f"Computing bootstrap confidence intervals with {self.samples} samples...")

        # Setup deterministic RNG (optional)
        rng = np.random.default_rng(self.seed)

        prep = self._bootstrap_prepare(
            df=df,
            cluster_names=cluster_names,
            model_names=model_names,
            metric_keys=metric_keys
        )
        if prep["n_conv"] == 0:
            return model_cluster_scores, cluster_scores, model_scores

        S = int(self.samples)
        arrays = self._bootstrap_allocate_arrays(
            S=S, n_models=prep["n_models"], n_clusters=prep["n_clusters"], n_metrics=prep["n_metrics"]
        )

        # Bootstrap sampling distribution over conversation ids (uniform)
        p = np.full(prep["n_conv"], 1.0 / float(prep["n_conv"]), dtype=float)

        for i in range(S):
            if log_fn and i % 20 == 0:
                log_fn(f"Bootstrap progress: {i}/{S} ({i/S*100:.1f}%)")
            if progress_callback and i % 5 == 0:
                try:
                    progress_callback(i / S)
                except Exception:
                    pass

            # True with-replacement bootstrap counts for each conversation_id
            conv_weights = rng.multinomial(prep["n_conv"], p).astype(float)
            self._bootstrap_compute_one_replicate(prep=prep, arrays=arrays, i=i, conv_weights=conv_weights)

        self._bootstrap_attach_results(
            prep=prep,
            arrays=arrays,
            model_cluster_scores=model_cluster_scores,
            cluster_scores=cluster_scores,
            model_scores=model_scores,
        )

        if log_fn:
            log_fn(f"✅ Bootstrap analysis completed with {S} samples")
        return model_cluster_scores, cluster_scores, model_scores
