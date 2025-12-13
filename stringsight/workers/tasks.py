import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import pandas as pd
import uuid

from stringsight.celery_app import celery_app
from stringsight.database import SessionLocal, init_db
from stringsight.models.job import Job
from stringsight.utils.paths import _get_results_dir
from stringsight.storage.adapter import get_storage_adapter
from stringsight.schemas import ExtractJobStartRequest, PipelineJobRequest
from stringsight.formatters import detect_method

# Import core logic
from stringsight.core.data_objects import PropertyDataset
from stringsight.extractors import get_extractor
from stringsight.postprocess import LLMJsonParser, PropertyValidator
from stringsight.prompts import get_system_prompt
from stringsight import explain

logger = logging.getLogger(__name__)

def _coerce_job_id(job_id: str | uuid.UUID) -> uuid.UUID:
    """Coerce a job identifier to a UUID.

    Args:
        job_id: Job identifier as a UUID or a UUID string.

    Returns:
        A `uuid.UUID` instance.
    """
    if isinstance(job_id, uuid.UUID):
        return job_id
    return uuid.UUID(job_id)

async def _run_extract_job_async(job_id: str, req_data: Dict[str, Any]):
    """Async implementation of the extraction logic."""
    init_db()
    job_uuid = _coerce_job_id(job_id)
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_uuid).first()
        if not job:
            logger.error(f"Job {job_id} not found in database")
            return

        job.status = "running"
        db.commit()

        # Reconstruct request object
        req = ExtractJobStartRequest(**req_data)
        
        df = pd.DataFrame(req.rows)

        # Apply sample_size if specified
        if req.sample_size and req.sample_size < len(df):
            df = df.sample(n=req.sample_size, random_state=42)
            logger.info(f"Sampled {req.sample_size} rows from {len(req.rows)} total rows")

        method = req.method or detect_method(list(df.columns))
        if method is None:
            raise RuntimeError("Unable to detect dataset method from columns.")

        # Ensure model column exists for single_model
        if method == "single_model" and "model" not in df.columns:
            model_name = req.model_name or "gpt-4.1"
            logger.info(f"Adding 'model' column with value '{model_name}'")
            df["model"] = model_name

        total = len(df)
        
        # Define progress callback to update job status in real-time
        # We need to be careful not to overload the DB with updates
        last_update = datetime.now()
        
        def update_progress(completed: int, total_count: int):
            nonlocal last_update
            now = datetime.now()
            # Update at most every 1 second
            if (now - last_update).total_seconds() > 1.0 or completed == total_count:
                try:
                    # Create new session for update to avoid transaction issues
                    with SessionLocal() as session:
                        current_job = session.query(Job).filter(Job.id == job_uuid).first()
                        if current_job:
                            current_job.progress = completed / total_count if total_count > 0 else 0.0
                            session.commit()
                    last_update = now
                except Exception as e:
                    logger.error(f"Failed to update progress: {e}")

        system_prompt = get_system_prompt(method, req.system_prompt, req.task_description)
        dataset = PropertyDataset.from_dataframe(df, method=method)

        extractor = get_extractor(
            model_name=req.model_name or "gpt-4.1",
            system_prompt=system_prompt,
            temperature=req.temperature or 0.7,
            top_p=req.top_p or 0.95,
            max_tokens=req.max_tokens or 16000,
            max_workers=req.max_workers or 64,
            include_scores_in_prompt=False if req.include_scores_in_prompt is None else req.include_scores_in_prompt,
            verbose=False,
            use_wandb=False,
        )

        # Run extraction with progress callback
        extracted_dataset = await extractor.run(dataset, progress_callback=update_progress)

        # Determine output directory
        base_results_dir = _get_results_dir()
        if req.output_dir:
            output_dir = str(base_results_dir / req.output_dir)
        else:
            # Create a directory for this extract job
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Use job_id in path
            output_dir = str(base_results_dir / f"extract_{job_id}_{timestamp}")
            
        storage = get_storage_adapter()
        storage.ensure_directory(output_dir)
        logger.info(f"Results will be saved to: {output_dir}")

        # Run parsing and validation
        parser = LLMJsonParser(fail_fast=False, verbose=False, use_wandb=False, output_dir=output_dir)
        parsed_dataset = parser.run(extracted_dataset)

        validator = PropertyValidator(verbose=False, use_wandb=False, output_dir=output_dir)
        result = validator.run(parsed_dataset)
        
        # Update job with success
        # Refresh job object
        job = db.query(Job).filter(Job.id == job_uuid).first()
        job.status = "completed"
        job.progress = 1.0
        job.result_path = req.output_dir if req.output_dir else f"extract_{job_id}_{timestamp}"
        db.commit()

    except Exception as e:
        logger.error(f"Error in extract job {job_id}: {e}", exc_info=True)
        try:
            job = db.query(Job).filter(Job.id == job_uuid).first()
            if job:
                job.status = "failed"
                job.error_message = str(e)
                db.commit()
        except Exception as db_e:
            logger.error(f"Failed to update job error state: {db_e}")
    finally:
        db.close()

@celery_app.task(bind=True, name="stringsight.workers.tasks.run_extract_job")
def run_extract_job(self, job_id: str, req_data: Dict[str, Any]):
    """Celery task wrapper for async extraction."""
    asyncio.run(_run_extract_job_async(job_id, req_data))


def run_extract_job_inprocess(job_id: str, req_data: Dict[str, Any]) -> None:
    """In-process runner for extraction jobs (no Celery/Redis required).

    Args:
        job_id: Job id as a UUID string.
        req_data: Serialized ExtractJobStartRequest.
    """
    asyncio.run(_run_extract_job_async(job_id, req_data))


def _run_pipeline_job(job_id: str, req_data: Dict[str, Any]) -> None:
    """Synchronous implementation of the full pipeline (extraction + clustering).

    Args:
        job_id: Job id as a UUID string.
        req_data: Serialized PipelineJobRequest.
    """
    init_db()
    job_uuid = _coerce_job_id(job_id)
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_uuid).first()
        if not job:
            logger.error(f"Job {job_id} not found")
            return
        
        job.status = "running"
        db.commit()
        
        req = PipelineJobRequest(**req_data)
        df = pd.DataFrame(req.rows)
        
        # Determine output directory
        base_results_dir = _get_results_dir()
        if req.output_dir:
            output_dir = str(base_results_dir / req.output_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = str(base_results_dir / f"pipeline_{job_id}_{timestamp}")
            
        storage = get_storage_adapter()
        storage.ensure_directory(output_dir)
        
        # Prepare arguments for explain()
        explain_kwargs = {
            "method": req.method,
            "system_prompt": req.system_prompt,
            "task_description": req.task_description,
            "prompt_expansion": req.prompt_expansion,
            "expansion_num_traces": req.expansion_num_traces,
            "expansion_model": req.expansion_model,
            "clusterer": req.clusterer,
            "min_cluster_size": req.min_cluster_size,
            "embedding_model": req.embedding_model,
            "max_workers": req.max_workers,
            "use_wandb": req.use_wandb,
            "verbose": False,
            "output_dir": output_dir,
            "groupby_column": req.groupby_column,
            "assign_outliers": req.assign_outliers,
            "score_columns": req.score_columns,
            "track_costs": True,
        }
        
        if req.extraction_model:
            explain_kwargs["model_name"] = req.extraction_model
        if req.summary_model:
            explain_kwargs["summary_model"] = req.summary_model
        if req.cluster_assignment_model:
            explain_kwargs["cluster_assignment_model"] = req.cluster_assignment_model
            
        if req.sample_size and req.sample_size < len(df):
             df = df.sample(n=req.sample_size, random_state=42)
        
        def update_progress(progress: float) -> None:
            try:
                with SessionLocal() as session:
                    current_job = session.query(Job).filter(Job.id == job_uuid).first()
                    if current_job:
                        current_job.progress = progress
                        session.commit()
            except Exception as e:
                logger.error(f"Failed to update progress: {e}")

        explain(df, **explain_kwargs, progress_callback=update_progress)

        job = db.query(Job).filter(Job.id == job_uuid).first()
        job.status = "completed"
        job.progress = 1.0
        job.result_path = req.output_dir if req.output_dir else f"pipeline_{job_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        db.commit()

    except Exception as e:
        logger.error(f"Error in pipeline job {job_id}: {e}", exc_info=True)
        try:
            job = db.query(Job).filter(Job.id == job_uuid).first()
            if job:
                job.status = "failed"
                job.error_message = str(e)
                db.commit()
        except Exception as db_e:
            logger.error(f"Failed to update job error state: {db_e}")
    finally:
        db.close()


@celery_app.task(bind=True, name="stringsight.workers.tasks.run_pipeline_job")
def run_pipeline_job(self, job_id: str, req_data: Dict[str, Any]):
    """Celery task for full pipeline (extraction + clustering)."""
    _run_pipeline_job(job_id, req_data)


def run_pipeline_job_inprocess(job_id: str, req_data: Dict[str, Any]) -> None:
    """In-process runner for pipeline jobs (no Celery/Redis required)."""
    _run_pipeline_job(job_id, req_data)

@celery_app.task(bind=True, name="stringsight.workers.tasks.run_cluster_job")
def run_cluster_job(self, job_id: str, req_data: Dict[str, Any]):
    """Celery task for clustering existing properties (no extraction)."""
    asyncio.run(_run_cluster_job_async(job_id, req_data))


def run_cluster_job_inprocess(job_id: str, req_data: Dict[str, Any]) -> None:
    """In-process runner for clustering jobs (no Celery/Redis required)."""
    asyncio.run(_run_cluster_job_async(job_id, req_data))

async def _run_cluster_job_async(job_id: str, req_data: Dict[str, Any]):
    """Async implementation of clustering logic."""
    from stringsight.schemas import ClusterJobRequest
    from stringsight.core.data_objects import PropertyDataset, Property, ConversationRecord
    from stringsight.clusterers import get_clusterer
    from stringsight.metrics.functional_metrics import FunctionalMetrics
    from stringsight.metrics.side_by_side import SideBySideMetrics
    from stringsight.api import format_conversations
    import json
    import time
    
    init_db()
    job_uuid = _coerce_job_id(job_id)
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_uuid).first()
        if not job:
            logger.error(f"Job {job_id} not found")
            return
        
        job.status = "running"
        job.progress = 0.0
        db.commit()
        
        # Reconstruct request
        req = ClusterJobRequest(**req_data)
        
        if not req.properties:
            raise ValueError("No properties provided for clustering")
        
        # Helper to update progress
        def update_progress(progress: float):
            try:
                with SessionLocal() as session:
                    current_job = session.query(Job).filter(Job.id == job_uuid).first()
                    if current_job:
                        current_job.progress = progress
                        session.commit()
            except Exception as e:
                logger.error(f"Failed to update progress: {e}")
        
        # Phase 1: Convert properties (5%)
        update_progress(0.05)
        properties: List[Property] = []
        for p in req.properties:
            try:
                raw_question_id = str(p.get("question_id", ""))
                base_question_id = raw_question_id.split('-')[0] if '-' in raw_question_id else raw_question_id
                
                prop = Property(
                    id=str(p.get("id", "")),
                    question_id=base_question_id,
                    model=str(p.get("model", "")),
                    property_description=p.get("property_description"),
                    category=p.get("category"),
                    reason=p.get("reason"),
                    evidence=p.get("evidence"),
                    behavior_type=p.get("behavior_type"),
                    raw_response=p.get("raw_response"),
                    contains_errors=p.get("contains_errors"),
                    unexpected_behavior=p.get("unexpected_behavior"),
                    meta=p.get("meta", {})
                )
                properties.append(prop)
            except Exception as e:
                logger.warning(f"Skipping invalid property: {e}")
                continue
        
        if not properties:
            raise ValueError("No valid properties after conversion")
        
        # Phase 2: Create conversations (10%)
        update_progress(0.10)
        conversations: List[ConversationRecord] = []
        all_models = set()
        property_keys = {(prop.question_id, prop.model) for prop in properties}
        
        for question_id, model in property_keys:
            all_models.add(model)
            matching_row = None
            for row in req.operationalRows:
                row_qid = str(row.get("question_id", ""))
                row_model = str(row.get("model", ""))
                
                if row_qid == question_id and row_model == model:
                    matching_row = row
                    break
                
                row_qid_base = row_qid.split('-')[0] if '-' in row_qid else row_qid
                question_id_base = question_id.split('-')[0] if '-' in question_id else question_id
                
                if (row_qid_base == question_id or row_qid == question_id_base) and row_model == model:
                    matching_row = row
                    break
            
            if matching_row:
                scores = matching_row.get("score") or matching_row.get("scores") or {}
            else:
                scores = {}
            
            response_value = ""
            if matching_row:
                response_value = matching_row.get("responses") or matching_row.get("model_response") or ""
            
            base_question_id = question_id.split('-')[0] if '-' in question_id else question_id
            
            conv = ConversationRecord(
                question_id=base_question_id,
                model=model,
                prompt=matching_row.get("prompt", "") if matching_row else "",
                responses=response_value,
                scores=scores,
                meta={}
            )
            conversations.append(conv)
        
        # Handle side-by-side if needed
        if req.method == "side_by_side":
            properties_by_qid = {}
            for p in properties:
                if p.question_id not in properties_by_qid:
                    properties_by_qid[p.question_id] = []
                properties_by_qid[p.question_id].append(p)
            
            operational_rows_map = {}
            for row in req.operationalRows:
                row_qid = str(row.get("question_id", ""))
                operational_rows_map[row_qid] = row
                if '-' in row_qid:
                    base_id = row_qid.split('-')[0]
                    if base_id not in operational_rows_map:
                        operational_rows_map[base_id] = row
            
            sxs_conversations = []
            for qid, props in properties_by_qid.items():
                matching_row = operational_rows_map.get(qid)
                if not matching_row and '-' in qid:
                    matching_row = operational_rows_map.get(qid.split('-')[0])
                
                if matching_row:
                    model_a = matching_row.get("model_a")
                    model_b = matching_row.get("model_b")
                    
                    if not model_a or not model_b:
                        unique_models = list(set(p.model for p in props))
                        if len(unique_models) >= 2:
                            model_a = unique_models[0]
                            model_b = unique_models[1]
                        else:
                            model_a = "model_a"
                            model_b = "model_b"
                    
                    score_a = matching_row.get("score_a", {})
                    score_b = matching_row.get("score_b", {})
                    
                    if not score_a and not score_b:
                        combined_score = matching_row.get("score") or matching_row.get("scores") or {}
                        if combined_score:
                            score_a = combined_score
                            score_b = combined_score
                    
                    meta = {}
                    if "winner" in matching_row:
                        meta["winner"] = matching_row["winner"]
                    elif "score" in matching_row and isinstance(matching_row["score"], dict) and "winner" in matching_row["score"]:
                        meta["winner"] = matching_row["score"]["winner"]
                    
                    conv = ConversationRecord(
                        question_id=qid,
                        model=[model_a, model_b],
                        prompt=matching_row.get("prompt", ""),
                        responses=[matching_row.get("model_a_response", ""), matching_row.get("model_b_response", "")],
                        scores=[score_a, score_b],
                        meta=meta
                    )
                    sxs_conversations.append(conv)
            
            if sxs_conversations:
                conversations = sxs_conversations
        
        # Create dataset
        dataset = PropertyDataset(
            conversations=conversations,
            all_models=list(all_models),
            properties=properties,
            clusters=[],
            model_stats={}
        )
        
        # Phase 3: Run clustering (60%)
        update_progress(0.15)
        groupby_column = None if req.params.groupBy == "none" else req.params.groupBy
        
        clusterer = get_clusterer(
            method="hdbscan",
            min_cluster_size=req.params.minClusterSize,
            embedding_model=req.params.embeddingModel,
            assign_outliers=False,
            include_embeddings=False,
            cache_embeddings=True,
            groupby_column=groupby_column,
        )
        
        clustered_dataset = await clusterer.run(dataset, column_name="property_description")
        update_progress(0.75)
        
        # Phase 4: Compute metrics (20%)
        if req.method == "side_by_side":
            metrics_computer = SideBySideMetrics(
                output_dir=None,
                compute_bootstrap=False,
                log_to_wandb=False,
                generate_plots=False
            )
        else:
            metrics_computer = FunctionalMetrics(
                output_dir=None,
                compute_bootstrap=False,
                log_to_wandb=False,
                generate_plots=False
            )
        
        clustered_dataset = metrics_computer.run(clustered_dataset)
        update_progress(0.95)
        
        # Phase 5: Save results (5%)
        base_results_dir = _get_results_dir()
        if req.output_dir:
            results_dir = base_results_dir / req.output_dir
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = "clustering"
            if req.operationalRows and len(req.operationalRows) > 0:
                first_row = req.operationalRows[0]
                if "__source_filename" in first_row:
                    base_filename = Path(str(first_row["__source_filename"])).stem
            results_dir = base_results_dir / f"{base_filename}_{timestamp}"
        
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save full dataset
        full_dataset_path = results_dir / "full_dataset.json"
        clustered_dataset.save(str(full_dataset_path))
        
        # Save conversations
        try:
            method = "side_by_side" if any(isinstance(conv.model, list) for conv in clustered_dataset.conversations) else "single_model"
            conv_df = clustered_dataset.to_dataframe(type="base", method=method)
            formatted_conversations = format_conversations(conv_df, method)
            conversation_path = results_dir / "conversation.jsonl"
            with open(conversation_path, 'w') as f:
                for conv in formatted_conversations:
                    f.write(json.dumps(conv, default=str) + '\n')
        except Exception as e:
            logger.warning(f"Failed to save conversation.jsonl: {e}")
        
        # Save clusters
        clusters_data = []
        for cluster in clustered_dataset.clusters:
            clusters_data.append({
                "id": cluster.id,
                "label": cluster.label,
                "size": cluster.size,
                "property_descriptions": cluster.property_descriptions,
                "property_ids": cluster.property_ids,
                "question_ids": cluster.question_ids,
                "meta": cluster.meta,
            })
        
        clusters_path = results_dir / "clusters.json"
        with open(clusters_path, 'w') as f:
            json.dump(clusters_data, f, indent=2, default=str)
        
        # Save properties
        properties_path = results_dir / "parsed_properties.jsonl"
        with open(properties_path, 'w') as f:
            for p in req.properties:
                f.write(json.dumps(p, default=str) + '\n')
        
        # Save summary
        summary_path = results_dir / "summary.txt"
        with open(summary_path, 'w') as f:
            f.write("StringSight Clustering Results Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Job ID: {job_id}\n")
            f.write(f"Total properties: {len(req.properties)}\n")
            f.write(f"Total clusters: {len(clustered_dataset.clusters)}\n")
            f.write(f"Models: {', '.join(clustered_dataset.all_models)}\n\n")
            f.write(f"Clustering parameters:\n")
            f.write(f"  - Min cluster size: {req.params.minClusterSize}\n")
            f.write(f"  - Embedding model: {req.params.embeddingModel}\n")
            f.write(f"  - Group by: {req.params.groupBy}\n")
        
        # Update job completion
        update_progress(1.0)
        job = db.query(Job).filter(Job.id == job_uuid).first()
        job.status = "completed"
        job.progress = 1.0
        # Store relative path from results directory for API compatibility
        relative_path = req.output_dir if req.output_dir else f"{base_filename}_{timestamp}"
        job.result_path = relative_path
        db.commit()

        logger.info(f"Cluster job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Error in cluster job {job_id}: {e}", exc_info=True)
        try:
            job = db.query(Job).filter(Job.id == job_uuid).first()
            if job:
                job.status = "failed"
                job.error_message = str(e)
                db.commit()
        except Exception as db_e:
            logger.error(f"Failed to update job error state: {db_e}")
    finally:
        db.close()

