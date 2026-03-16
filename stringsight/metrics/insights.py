"""
LLM-ready metrics insight construction and summarization.

This module converts flattened `model_cluster_scores` rows into a compact,
ranked payload focused on potentially interesting behaviors, then generates a
sectioned natural-language overview.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

from stringsight.core.llm_utils import single_completion
from stringsight.prompts.insights import (
    METRICS_INSIGHTS_SYSTEM_PROMPT,
    build_metrics_insights_user_prompt,
)


@dataclass(frozen=True)
class MetricsInsightConfig:
    """Configuration for selecting and narrating metrics insights.

    Attributes:
        top_k_per_section: Maximum candidate clusters passed per section.
        min_cluster_proportion: Minimum overall cluster proportion to consider.
        min_abs_quality_delta: Minimum absolute quality delta for impact sections.
        min_model_freq_gap: Minimum model frequency gap (max - min proportion).
        llm_model: LLM model used for final sectioned narration.
    """

    top_k_per_section: int = 10
    min_cluster_proportion: float = 0.01
    min_abs_quality_delta: float = 0.03
    min_model_freq_gap: float = 0.05
    llm_model: str = "gpt-4.1-mini"


def _to_float(value: Any) -> float | None:
    """Return float(value) when value is numeric, otherwise None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _normalize_group(group_value: Any) -> str:
    """Normalize group labels to frontend/backend canonical categories."""
    raw: str = str(group_value).strip().lower()
    if raw == "negative (critical)" or raw == "negative critical":
        return "negative_critical"
    if (
        raw == "negative (non-critical)"
        or raw == "negative non-critical"
        or raw == "negative (non critical)"
    ):
        return "negative_non_critical"
    if raw == "positive":
        return "positive"
    if raw == "style":
        return "style"
    return raw


def _is_outlier_cluster(cluster_label: str) -> bool:
    """Return True when cluster label appears to be an outlier bucket."""
    lowered: str = cluster_label.lower()
    return "outlier" in lowered


def _extract_quality_delta_metrics(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    """Extract metric names from flattened `quality_delta_<metric>` columns."""
    metric_names: set[str] = set()
    for row in rows:
        for key in row.keys():
            if not key.startswith("quality_delta_"):
                continue
            if key.endswith("_significant") or "_ci_" in key:
                continue
            metric_names.add(key.replace("quality_delta_", "", 1))
    return sorted(metric_names)


def _choose_primary_quality_metric(metric_names: Sequence[str]) -> str | None:
    """Pick a primary quality metric used for failure ranking."""
    if "win_rate" in metric_names:
        return "win_rate"
    if len(metric_names) == 0:
        return None
    return metric_names[0]


def _build_cluster_scores_lookup(
    cluster_scores_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Mapping[str, Any]]:
    """Build cluster->row lookup from flattened `cluster_scores` rows."""
    lookup: Dict[str, Mapping[str, Any]] = {}
    for row in cluster_scores_rows:
        if "cluster" not in row:
            continue
        cluster_name: str = str(row["cluster"])
        lookup[cluster_name] = row
    return lookup


def _mean(values: Sequence[float]) -> float:
    """Compute arithmetic mean of non-empty numeric sequence."""
    return float(sum(values) / len(values))


def _extract_behavior_descriptions_from_examples(
    examples: Any,
    *,
    max_items: int = 3,
) -> List[str]:
    """Extract short behavior descriptions from flattened `examples`.

    Expected `examples` format from metrics rows:
        List[Tuple[conversation_id, conversation_metadata, property_metadata]]
    where `property_metadata` may include `property_description`.

    Args:
        examples: Any value from metrics row `examples` key.
        max_items: Maximum number of unique snippets to return.

    Returns:
        List[str] of unique property descriptions (possibly empty).
    """
    if not isinstance(examples, list):
        return []

    snippets: List[str] = []
    seen: set[str] = set()
    for item in examples:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        property_metadata: Any = item[2]
        if not isinstance(property_metadata, Mapping):
            continue
        if "property_description" not in property_metadata:
            continue
        desc_raw: Any = property_metadata["property_description"]
        desc: str = str(desc_raw).strip()
        if len(desc) == 0 or desc in seen:
            continue
        seen.add(desc)
        snippets.append(desc)
        if len(snippets) >= max_items:
            break
    return snippets


def _extract_behavior_descriptions_from_cluster_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_items: int = 5,
) -> List[str]:
    """Extract unique behavior descriptions across all rows for one cluster.

    Args:
        rows: List of per-model rows for the same cluster. Each row may contain
            an `examples` key with
            List[Tuple[conversation_id, conversation_metadata, property_metadata]].
            `property_metadata` may include `property_description` (str).
        max_items: Maximum number of unique snippets to return.

    Returns:
        List[str] of unique behavior descriptions ordered by first appearance.
    """
    snippets: List[str] = []
    seen: set[str] = set()
    for row in rows:
        examples: Any = row["examples"] if "examples" in row else None
        row_snippets: List[str] = _extract_behavior_descriptions_from_examples(
            examples,
            max_items=max_items,
        )
        for snippet in row_snippets:
            if snippet in seen:
                continue
            seen.add(snippet)
            snippets.append(snippet)
            if len(snippets) >= max_items:
                return snippets
    return snippets


def _sanitize_overview_candidate(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a candidate dict safe for persisted overview output.

    Expected input:
        Candidate rows built from feature selection, which may include internal
        grouping keys (e.g., `group`, `behavior_type`) used for ranking logic.

    Output:
        Dict[str, Any] without internal grouping labels so the overview focuses
        on cluster-level evidence and metrics only.
    """
    sanitized: Dict[str, Any] = dict(candidate)
    if "group" in sanitized:
        del sanitized["group"]
    if "behavior_type" in sanitized:
        del sanitized["behavior_type"]
    return sanitized


def _build_cluster_feature_rows(
    model_cluster_scores_rows: Sequence[Mapping[str, Any]],
    cluster_scores_rows: Sequence[Mapping[str, Any]],
    quality_metric_names: Sequence[str],
) -> List[Dict[str, Any]]:
    """Build per-cluster feature rows used for candidate selection.

    Expected input row format:
        model_cluster_scores_rows: each row has keys:
            - model (str), cluster (str), proportion (float), proportion_delta (float)
            - metadata (dict, optional; includes group)
            - quality_delta_<metric> (float, optional)
            - quality_delta_<metric>_significant (bool, optional)

    Returns:
        List[dict] where each dict contains:
            - cluster, group
            - cluster_proportion_overall, cluster_size_estimate
            - model_proportions: dict[str, float]
            - model_freq_deltas: dict[str, float]
            - max_model_gap
            - max_gap_model, min_gap_model
            - max_gap_model_proportion, min_gap_model_proportion
            - behavior_description_examples: list[str]
            - quality_delta_by_metric: dict[str, float]
            - quality_significance_by_metric: dict[str, bool]
            - failure_impact_score, quality_impact_score, frequency_gap_score
    """
    by_cluster: Dict[str, List[Mapping[str, Any]]] = {}
    for row in model_cluster_scores_rows:
        if "cluster" not in row:
            continue
        cluster_name: str = str(row["cluster"])
        if cluster_name not in by_cluster:
            by_cluster[cluster_name] = []
        by_cluster[cluster_name].append(row)

    cluster_lookup: Dict[str, Mapping[str, Any]] = _build_cluster_scores_lookup(cluster_scores_rows)
    primary_metric: str | None = _choose_primary_quality_metric(quality_metric_names)

    features: List[Dict[str, Any]] = []
    for cluster_name, rows in by_cluster.items():
        if _is_outlier_cluster(cluster_name):
            continue

        group_value: str = ""
        first_metadata: Any = rows[0]["metadata"] if "metadata" in rows[0] else None
        if isinstance(first_metadata, Mapping) and "group" in first_metadata:
            group_value = _normalize_group(first_metadata["group"])

        model_proportions: Dict[str, float] = {}
        model_freq_deltas: Dict[str, float] = {}
        for row in rows:
            if "model" not in row:
                continue
            model_name: str = str(row["model"])
            proportion_value: float | None = _to_float(row["proportion"] if "proportion" in row else None)
            if proportion_value is not None:
                model_proportions[model_name] = proportion_value
            delta_value: float | None = _to_float(
                row["proportion_delta"] if "proportion_delta" in row else None
            )
            if delta_value is not None:
                model_freq_deltas[model_name] = delta_value

        cluster_score_row: Mapping[str, Any] | None = None
        if cluster_name in cluster_lookup:
            cluster_score_row = cluster_lookup[cluster_name]

        cluster_proportion_overall: float | None = None
        cluster_size_estimate: float | None = None
        if cluster_score_row is not None:
            cluster_proportion_overall = _to_float(
                cluster_score_row["proportion"] if "proportion" in cluster_score_row else None
            )
            cluster_size_estimate = _to_float(
                cluster_score_row["size"] if "size" in cluster_score_row else None
            )

        if cluster_proportion_overall is None and len(model_proportions) > 0:
            cluster_proportion_overall = _mean(list(model_proportions.values()))
        if cluster_size_estimate is None:
            size_values: List[float] = []
            for row in rows:
                size_value: float | None = _to_float(row["size"] if "size" in row else None)
                if size_value is not None:
                    size_values.append(size_value)
            if len(size_values) > 0:
                cluster_size_estimate = float(sum(size_values))

        quality_delta_by_metric: Dict[str, float] = {}
        quality_significance_by_metric: Dict[str, bool] = {}
        for metric_name in quality_metric_names:
            value_key: str = f"quality_delta_{metric_name}"
            sig_key: str = f"quality_delta_{metric_name}_significant"

            metric_values: List[float] = []
            metric_significance_flags: List[bool] = []
            for row in rows:
                metric_value: float | None = _to_float(row[value_key] if value_key in row else None)
                if metric_value is not None:
                    metric_values.append(metric_value)
                if sig_key in row and isinstance(row[sig_key], bool):
                    metric_significance_flags.append(row[sig_key])

            if len(metric_values) > 0:
                quality_delta_by_metric[metric_name] = _mean(metric_values)
            if len(metric_significance_flags) > 0:
                quality_significance_by_metric[metric_name] = any(metric_significance_flags)

        max_model_gap: float = 0.0
        max_gap_model: str | None = None
        min_gap_model: str | None = None
        max_gap_model_proportion: float | None = None
        min_gap_model_proportion: float | None = None
        if len(model_proportions) > 1:
            sorted_by_proportion: List[tuple[str, float]] = sorted(
                model_proportions.items(),
                key=lambda kv: kv[1],
            )
            min_gap_model, min_gap_model_proportion = sorted_by_proportion[0]
            max_gap_model, max_gap_model_proportion = sorted_by_proportion[-1]
            max_model_gap = max_gap_model_proportion - min_gap_model_proportion

        primary_quality_delta: float = 0.0
        if primary_metric is not None and primary_metric in quality_delta_by_metric:
            primary_quality_delta = quality_delta_by_metric[primary_metric]

        negative_quality_harm: float = 0.0
        if primary_quality_delta < 0:
            negative_quality_harm = -primary_quality_delta

        proportion_for_scoring: float = cluster_proportion_overall if cluster_proportion_overall is not None else 0.0
        failure_impact_score: float = proportion_for_scoring * negative_quality_harm
        quality_impact_score: float = 0.0
        if len(quality_delta_by_metric) > 0:
            quality_impact_score = max(abs(v) for v in quality_delta_by_metric.values())
        frequency_gap_score: float = max_model_gap

        features.append(
            {
                "cluster": cluster_name,
                "group": group_value,
                "cluster_proportion_overall": cluster_proportion_overall,
                "cluster_size_estimate": cluster_size_estimate,
                "model_proportions": model_proportions,
                "model_freq_deltas": model_freq_deltas,
                "max_model_gap": max_model_gap,
                "max_gap_model": max_gap_model,
                "min_gap_model": min_gap_model,
                "max_gap_model_proportion": max_gap_model_proportion,
                "min_gap_model_proportion": min_gap_model_proportion,
                "behavior_description_examples": _extract_behavior_descriptions_from_cluster_rows(
                    rows
                ),
                "quality_delta_by_metric": quality_delta_by_metric,
                "quality_significance_by_metric": quality_significance_by_metric,
                "failure_impact_score": failure_impact_score,
                "quality_impact_score": quality_impact_score,
                "frequency_gap_score": frequency_gap_score,
                "primary_quality_metric": primary_metric,
            }
        )
    return features


def _filter_potentially_interesting_clusters(
    cluster_feature_rows: Sequence[Mapping[str, Any]],
    config: MetricsInsightConfig,
) -> List[Dict[str, Any]]:
    """Filter out clusters that do not meet interestingness gates."""
    kept: List[Dict[str, Any]] = []
    for row in cluster_feature_rows:
        proportion_value: float = 0.0
        if "cluster_proportion_overall" in row:
            parsed = _to_float(row["cluster_proportion_overall"])
            if parsed is not None:
                proportion_value = parsed

        quality_impact_score: float = 0.0
        if "quality_impact_score" in row:
            parsed = _to_float(row["quality_impact_score"])
            if parsed is not None:
                quality_impact_score = parsed

        frequency_gap_score: float = 0.0
        if "frequency_gap_score" in row:
            parsed = _to_float(row["frequency_gap_score"])
            if parsed is not None:
                frequency_gap_score = parsed

        failure_impact_score: float = 0.0
        if "failure_impact_score" in row:
            parsed = _to_float(row["failure_impact_score"])
            if parsed is not None:
                failure_impact_score = parsed

        meets_failure_signal: bool = failure_impact_score > 0.0
        meets_quality_signal: bool = quality_impact_score >= config.min_abs_quality_delta
        meets_frequency_signal: bool = frequency_gap_score >= config.min_model_freq_gap
        meets_prevalence_signal: bool = proportion_value >= config.min_cluster_proportion

        if (meets_failure_signal or meets_quality_signal or meets_frequency_signal) and meets_prevalence_signal:
            kept.append(dict(row))
    return kept


def _take_top_by_score(
    rows: Sequence[Mapping[str, Any]],
    score_key: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    """Sort rows descending by score key and keep top_k."""
    sorted_rows: List[Mapping[str, Any]] = sorted(
        rows,
        key=lambda r: _to_float(r[score_key] if score_key in r else None) or 0.0,
        reverse=True,
    )
    return [dict(row) for row in sorted_rows[:top_k]]


def _build_llm_payload(
    *,
    method: str,
    model_names: Sequence[str],
    total_unique_conversations: int | None,
    quality_metric_names: Sequence[str],
    failure_candidates: Sequence[Mapping[str, Any]],
    quality_impact_candidates: Sequence[Mapping[str, Any]],
    frequency_gap_candidates: Sequence[Mapping[str, Any]],
    config: MetricsInsightConfig,
) -> Dict[str, Any]:
    """Construct a compact payload for final sectioned narration."""
    sanitized_failure_candidates: List[Dict[str, Any]] = [
        _sanitize_overview_candidate(candidate) for candidate in failure_candidates
    ]
    sanitized_quality_candidates: List[Dict[str, Any]] = [
        _sanitize_overview_candidate(candidate) for candidate in quality_impact_candidates
    ]
    sanitized_frequency_candidates: List[Dict[str, Any]] = [
        _sanitize_overview_candidate(candidate) for candidate in frequency_gap_candidates
    ]

    return {
        "context": {
            "method": method,
            "models": list(model_names),
            "total_unique_conversations": total_unique_conversations,
            "quality_metrics": list(quality_metric_names),
            "selection_rules": {
                "top_k_per_section": config.top_k_per_section,
                "min_cluster_proportion": config.min_cluster_proportion,
                "min_abs_quality_delta": config.min_abs_quality_delta,
                "min_model_freq_gap": config.min_model_freq_gap,
                "interesting_only": True,
            },
        },
        "failures": sanitized_failure_candidates,
        "quality_impact": sanitized_quality_candidates,
        "frequency_discrepancies": sanitized_frequency_candidates,
    }


def generate_metrics_insights(
    *,
    model_cluster_scores_rows: Sequence[Mapping[str, Any]],
    cluster_scores_rows: Sequence[Mapping[str, Any]],
    model_names: Sequence[str],
    method: str,
    total_unique_conversations: int | None,
    config: MetricsInsightConfig | None = None,
) -> Dict[str, Any]:
    """Generate sectioned natural-language insights from model-cluster metrics.

    Args:
        model_cluster_scores_rows: Flattened rows with keys:
            - model, cluster, proportion, proportion_delta, metadata (optional),
            - quality_delta_<metric> (optional), quality_delta_<metric>_significant (optional).
        cluster_scores_rows: Flattened cluster-level rows with keys:
            - cluster, proportion, size (optional), quality/quality_delta columns (optional).
        model_names: Ordered model names shown in outputs.
        method: Analysis method name (`single_model` or `side_by_side`).
        total_unique_conversations: Total unique conversations for context in narration.
        config: Optional selector/narration configuration.

    Returns:
        Dict with keys:
            - narrative_markdown: str
            - candidates: Dict[str, List[Dict[str, Any]]]
            - selection_metadata: Dict[str, Any]
    """
    effective_config: MetricsInsightConfig = config or MetricsInsightConfig()

    quality_metric_names: List[str] = _extract_quality_delta_metrics(model_cluster_scores_rows)
    feature_rows: List[Dict[str, Any]] = _build_cluster_feature_rows(
        model_cluster_scores_rows=model_cluster_scores_rows,
        cluster_scores_rows=cluster_scores_rows,
        quality_metric_names=quality_metric_names,
    )
    interesting_rows: List[Dict[str, Any]] = _filter_potentially_interesting_clusters(
        cluster_feature_rows=feature_rows,
        config=effective_config,
    )

    failure_rows: List[Dict[str, Any]] = []
    for row in interesting_rows:
        group_name: str = str(row["group"]) if "group" in row else ""
        if group_name in {"negative_critical", "negative_non_critical"}:
            failure_rows.append(row)

    quality_impact_rows: List[Dict[str, Any]] = []
    for row in interesting_rows:
        group_name: str = str(row["group"]) if "group" in row else ""
        if group_name in {"negative_critical", "negative_non_critical", "style"}:
            quality_impact_rows.append(row)

    frequency_rows: List[Dict[str, Any]] = []
    for row in interesting_rows:
        freq_gap_value: float = _to_float(row["frequency_gap_score"] if "frequency_gap_score" in row else None) or 0.0
        if freq_gap_value >= effective_config.min_model_freq_gap:
            frequency_rows.append(row)

    failure_candidates: List[Dict[str, Any]] = _take_top_by_score(
        rows=failure_rows,
        score_key="failure_impact_score",
        top_k=effective_config.top_k_per_section,
    )
    quality_candidates: List[Dict[str, Any]] = _take_top_by_score(
        rows=quality_impact_rows,
        score_key="quality_impact_score",
        top_k=effective_config.top_k_per_section,
    )
    frequency_candidates: List[Dict[str, Any]] = _take_top_by_score(
        rows=frequency_rows,
        score_key="frequency_gap_score",
        top_k=effective_config.top_k_per_section,
    )

    failure_candidates_output: List[Dict[str, Any]] = [
        _sanitize_overview_candidate(candidate) for candidate in failure_candidates
    ]
    quality_candidates_output: List[Dict[str, Any]] = [
        _sanitize_overview_candidate(candidate) for candidate in quality_candidates
    ]
    frequency_candidates_output: List[Dict[str, Any]] = [
        _sanitize_overview_candidate(candidate) for candidate in frequency_candidates
    ]

    llm_payload: Dict[str, Any] = _build_llm_payload(
        method=method,
        model_names=model_names,
        total_unique_conversations=total_unique_conversations,
        quality_metric_names=quality_metric_names,
        failure_candidates=failure_candidates,
        quality_impact_candidates=quality_candidates,
        frequency_gap_candidates=frequency_candidates,
        config=effective_config,
    )
    if len(failure_candidates) == 0 and len(quality_candidates) == 0 and len(frequency_candidates) == 0:
        return {
            "narrative_markdown": "",
            "candidates": {
                "failures": failure_candidates_output,
                "quality_impact": quality_candidates_output,
                "frequency_discrepancies": frequency_candidates_output,
            },
            "selection_metadata": {
                "top_k_per_section": effective_config.top_k_per_section,
                "min_cluster_proportion": effective_config.min_cluster_proportion,
                "min_abs_quality_delta": effective_config.min_abs_quality_delta,
                "min_model_freq_gap": effective_config.min_model_freq_gap,
                "quality_metrics": quality_metric_names,
                "input_cluster_count": len(feature_rows),
                "interesting_cluster_count": len(interesting_rows),
            },
        }

    user_prompt: str = build_metrics_insights_user_prompt(llm_payload)
    narrative_markdown: str | None = single_completion(
        message=user_prompt,
        model=effective_config.llm_model,
        system_prompt=METRICS_INSIGHTS_SYSTEM_PROMPT,
        temperature=0.0,
    )

    return {
        "narrative_markdown": narrative_markdown or "",
        "candidates": {
            "failures": failure_candidates_output,
            "quality_impact": quality_candidates_output,
            "frequency_discrepancies": frequency_candidates_output,
        },
        "selection_metadata": {
            "top_k_per_section": effective_config.top_k_per_section,
            "min_cluster_proportion": effective_config.min_cluster_proportion,
            "min_abs_quality_delta": effective_config.min_abs_quality_delta,
            "min_model_freq_gap": effective_config.min_model_freq_gap,
            "quality_metrics": quality_metric_names,
            "input_cluster_count": len(feature_rows),
            "interesting_cluster_count": len(interesting_rows),
        },
    }

