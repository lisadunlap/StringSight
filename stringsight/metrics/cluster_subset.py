from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def _build_property_maps(properties: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], Dict[Tuple[str, str], List[str]]]:
    """Create helper maps from properties list.

    Returns:
        - prop_by_id: property_id -> minimal record with question_id, model, property_description, category, behavior_type
        - prop_ids_by_q_model: (question_id, model) -> list[property_id]
    """
    prop_by_id: Dict[str, Dict[str, Any]] = {}
    prop_ids_by_q_model: Dict[Tuple[str, str], List[str]] = {}
    for p in properties:
        pid = str(p.get("id"))
        raw_qid = str(p.get("question_id"))
        # Strip property index suffix to get base conversation ID
        # Handle cases where properties might have compound IDs like "48-0"
        qid = raw_qid.split('-')[0] if '-' in raw_qid else raw_qid
        model = str(p.get("model"))
        prop_by_id[pid] = {
            "property_id": pid,
            "question_id": qid,
            "model": model,
            "property_description": p.get("property_description"),
            "category": p.get("category"),
            "behavior_type": p.get("behavior_type"),
            "property_meta": p.get("meta", {}),
        }
        key = (qid, model)
        prop_ids_by_q_model.setdefault(key, []).append(pid)
    return prop_by_id, prop_ids_by_q_model


def _build_score_map(operational_rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, float]]:
    """Create a map from (question_id, model) to consolidated score dict."""
    score_map: Dict[Tuple[str, str], Dict[str, float]] = {}
    for r in operational_rows:
        raw_qid = str(r.get("question_id"))
        # Strip property index suffix to get base conversation ID
        # Operational rows might have compound IDs like "48-0" from frontend
        qid = raw_qid.split('-')[0] if '-' in raw_qid else raw_qid
        # operationalRows are standardized to single model in this UI; if side-by-side appears later, extend mapping
        model = str(r.get("model"))
        score = r.get("score") or {}
        if isinstance(score, dict):
            score_map[(qid, model)] = score
    return score_map


def prepare_long_frame(
    *,
    clusters: List[Dict[str, Any]],
    properties: List[Dict[str, Any]],
    operational_rows: List[Dict[str, Any]],
    included_property_ids: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Build a long dataframe for metrics with rows per (conversation_id, model, cluster, property_id).

    Columns: [conversation_id, model, cluster, property_id, property_description, scores, cluster_metadata]
    """
    prop_by_id, _ = _build_property_maps(properties)
    score_map = _build_score_map(operational_rows)

    include_set = set(str(pid) for pid in (included_property_ids or [])) if included_property_ids else None

    rows: List[Dict[str, Any]] = []
    for c in clusters:
        cluster_label = c.get("label")
        cluster_meta = c.get("meta", {})
        member_prop_ids: List[str] = [str(x) for x in (c.get("property_ids") or [])]
        for pid in member_prop_ids:
            if include_set is not None and pid not in include_set:
                continue
            prop = prop_by_id.get(pid)
            if not prop:
                continue
            qid = prop["question_id"]
            model = prop["model"]
            scores = score_map.get((qid, model), {})
            rows.append({
                "conversation_id": qid,
                "model": model,
                "cluster": cluster_label,
                "property_id": pid,
                "property_description": prop.get("property_description"),
                "scores": scores,
                "cluster_metadata": cluster_meta,
            })

    if not rows:
        return pd.DataFrame(columns=[
            "conversation_id", "model", "cluster", "property_id", "property_description", "scores", "cluster_metadata"
        ])

    df = pd.DataFrame(rows)
    return df


def _avg_scores(df: pd.DataFrame) -> Dict[str, float]:
    """Compute per-metric mean from a column of score dicts."""
    if df.empty or "scores" not in df.columns:
        logger.debug("_avg_scores: DataFrame empty or no scores column")
        return {}
    valid = df[df["scores"].apply(lambda x: isinstance(x, dict) and len(x) > 0)]
    if valid.empty:
        logger.debug(f"_avg_scores: No valid scores found. Total rows: {len(df)}, Sample scores: {df['scores'].head().tolist()}")
        return {}
    score_df = pd.DataFrame(valid["scores"].tolist())
    result = {col: float(score_df[col].mean()) for col in score_df.columns}
    logger.debug(f"_avg_scores: Computed quality metrics: {result}")
    return result


def compute_total_conversations_by_model(properties: List[Dict[str, Any]]) -> Dict[str, int]:
    """Compute total unique conversation counts per model from the full properties dataset.
    
    Args:
        properties: List of property dictionaries containing question_id and model fields
        
    Returns:
        Dict mapping model name to count of unique conversation_ids (question_ids) for that model
    """
    if not properties:
        return {}
    
    # Convert to DataFrame for cleaner operations
    props_df = pd.DataFrame(properties)
    
    # Ensure required columns exist
    if 'question_id' not in props_df.columns or 'model' not in props_df.columns:
        return {}
    
    # Convert to strings and filter out empty values
    props_df['question_id'] = props_df['question_id'].astype(str)
    props_df['model'] = props_df['model'].astype(str)
    props_df = props_df[(props_df['question_id'] != '') & (props_df['model'] != '')]
    
    # Count unique conversations (question_ids) per model
    conversation_counts = props_df.drop_duplicates(subset=['question_id', 'model']).groupby('model')['question_id'].nunique()
    
    return conversation_counts.to_dict()


def compute_subset_metrics(long_df: pd.DataFrame, total_conversations_by_model: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    """Compute cluster-level metrics and per-model proportions from a long frame.

    Args:
        long_df: DataFrame with conversation_id, model, cluster columns
        total_conversations_by_model: Dict mapping model name to total unique conversation count in full dataset.
                                    If None, falls back to counting only conversations present in clusters.

    Returns dict with keys:
      - cluster_scores: { cluster: { size, proportion, quality, quality_delta } }
      - model_cluster_scores: { model: { cluster: { proportion } } }
      - total_conversations_by_model: { model: total_conversation_count }
    """
    out: Dict[str, Any] = {"cluster_scores": {}, "model_cluster_scores": {}}
    if long_df.empty:
        return out

    # De-duplicate at (conversation_id, model, cluster) level for size and proportions
    base = long_df.drop_duplicates(subset=["conversation_id", "model", "cluster"]).copy()

    # Global denominators
    global_total = len(base)
    global_quality = _avg_scores(base)

    # Cluster-level aggregates across all models
    for cluster_name, sub in base.groupby("cluster", dropna=False):
        size = len(sub)
        proportion = float(size) / float(global_total) if global_total > 0 else 0.0
        quality = _avg_scores(sub)
        quality_raw_delta = {k: quality.get(k, 0.0) - global_quality.get(k, 0.0) for k in quality.keys()}
        # Quality delta is just the raw difference (no proportion weighting)
        quality_delta = quality_raw_delta
        
        out["cluster_scores"][cluster_name] = {
            "size": int(size),
            "proportion": proportion,
            "quality": quality,
            "quality_delta": quality_delta,
        }

    # Per-model proportions within clusters
    # Use total conversations from full dataset if provided, otherwise fall back to subset
    if total_conversations_by_model is not None:
        model_denoms = total_conversations_by_model
    else:
        # Fallback: count conversations only in clusters (original behavior)
        model_denoms = base.drop_duplicates(subset=["conversation_id", "model"]).groupby("model").size().to_dict()
    
    # Get all unique models and clusters to ensure complete matrix
    all_models = set(str(m) for m in model_denoms.keys())
    all_clusters = set(str(c) for c in base["cluster"].unique())
    
    # Initialize model_cluster_scores with all model-cluster combinations
    for model_name in all_models:
        out["model_cluster_scores"][model_name] = {}
        for cluster_name in all_clusters:
            out["model_cluster_scores"][model_name][cluster_name] = {"proportion": 0.0}
    
    # Fill in actual proportions and quality metrics for models that appear in clusters
    for model_name, model_sub in base.groupby("model", dropna=False):
        model_name_str = str(model_name)
        denom = int(model_denoms.get(model_name_str, 0))
        
        # Compute model's overall quality scores for delta calculation
        model_quality = _avg_scores(model_sub)
        
        for cluster_name, sub in model_sub.groupby("cluster", dropna=False):
            cluster_name_str = str(cluster_name)
            numer = len(sub)
            prop = float(numer) / float(denom) if denom > 0 else 0.0
            
            # Compute quality scores for this model-cluster combination
            cluster_quality = _avg_scores(sub)
            
            # Compute quality deltas (how this cluster differs from model's overall average)
            quality_delta = {k: cluster_quality.get(k, 0.0) - model_quality.get(k, 0.0) for k in cluster_quality.keys()}
            
            # Compute proportion delta (how this model over/under-represents in this cluster vs average)
            # Average proportion across all models for this cluster
            cluster_avg_prop = float(len(base[base["cluster"] == cluster_name])) / float(global_total) if global_total > 0 else 0.0
            proportion_delta = prop - cluster_avg_prop
            
            out["model_cluster_scores"][model_name_str][cluster_name_str] = {
                "size": int(numer),
                "proportion": prop,
                "proportion_delta": proportion_delta,
                "quality": cluster_quality,
                "quality_delta": quality_delta,
            }
            
            # Debug log first model-cluster combo
            if logger.isEnabledFor(logging.DEBUG):
                if model_name_str == list(all_models)[0] and cluster_name_str == list(all_clusters)[0]:
                    logger.debug(f"Sample model_cluster_scores for {model_name_str} / {cluster_name_str}:")
                    logger.debug(f"  - size: {numer}")
                    logger.debug(f"  - quality: {cluster_quality}")
                    logger.debug(f"  - quality_delta: {quality_delta}")

    # Include total conversation counts in output for frontend display
    out["total_conversations_by_model"] = dict(model_denoms)
    
    # Log summary of what was computed
    if out["model_cluster_scores"]:
        sample_model = list(out["model_cluster_scores"].keys())[0]
        sample_cluster = list(out["model_cluster_scores"][sample_model].keys())[0]
        sample_metrics = out["model_cluster_scores"][sample_model][sample_cluster]
        logger.info(f"✅ compute_subset_metrics completed:")
        logger.info(f"  - Models: {len(out['model_cluster_scores'])}")
        logger.info(f"  - Clusters: {len(out['cluster_scores'])}")
        logger.info(f"  - Sample metrics keys: {list(sample_metrics.keys())}")
        logger.info(f"  - Sample quality keys: {list(sample_metrics.get('quality', {}).keys())}")
    
    return out


def enrich_clusters_with_metrics(
    clusters: List[Dict[str, Any]],
    scores: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Attach computed metrics back onto cluster dicts and update size.

    - meta.quality
    - meta.quality_delta
    - meta.proportion_by_model
    - size (from subset)
    """
    cluster_scores: Dict[str, Any] = scores.get("cluster_scores", {})
    model_cluster_scores: Dict[str, Any] = scores.get("model_cluster_scores", {})

    # Precompute per-cluster per-model proportions
    proportions_by_cluster: Dict[str, Dict[str, float]] = {}
    for model_name, per_cluster in model_cluster_scores.items():
        for cluster_name, vals in per_cluster.items():
            proportions_by_cluster.setdefault(cluster_name, {})[model_name] = float(vals.get("proportion", 0.0))

    enriched: List[Dict[str, Any]] = []
    for c in clusters:
        label = c.get("label")
        cs = cluster_scores.get(label, {})
        # update size if available
        if isinstance(cs.get("size"), int):
            c["size"] = int(cs["size"])
        meta = dict(c.get("meta", {}))
        if "quality" in cs:
            meta["quality"] = cs["quality"]
        if "quality_delta" in cs:
            meta["quality_delta"] = cs["quality_delta"]
        if label in proportions_by_cluster:
            meta["proportion_by_model"] = proportions_by_cluster[label]
        c["meta"] = meta
        enriched.append(c)
    return enriched


