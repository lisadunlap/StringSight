from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd


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
        qid = str(p.get("question_id"))
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
        qid = str(r.get("question_id"))
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
        return {}
    valid = df[df["scores"].apply(lambda x: isinstance(x, dict) and len(x) > 0)]
    if valid.empty:
        return {}
    score_df = pd.DataFrame(valid["scores"].tolist())
    return {col: float(score_df[col].mean()) for col in score_df.columns}


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
        quality_delta = {k: quality.get(k, 0.0) - global_quality.get(k, 0.0) for k in quality.keys()}
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
    
    # Fill in actual proportions for models that appear in clusters
    for model_name, model_sub in base.groupby("model", dropna=False):
        model_name_str = str(model_name)
        denom = int(model_denoms.get(model_name_str, 0))
        for cluster_name, sub in model_sub.groupby("cluster", dropna=False):
            cluster_name_str = str(cluster_name)
            numer = len(sub)
            prop = float(numer) / float(denom) if denom > 0 else 0.0
            out["model_cluster_scores"][model_name_str][cluster_name_str] = {"proportion": prop}

    # Include total conversation counts in output for frontend display
    out["total_conversations_by_model"] = dict(model_denoms)
    
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


