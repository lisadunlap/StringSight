import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import pandas as pd
import uuid

from stringsight.celery_app import celery_app
from stringsight.database import SessionLocal
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

async def _run_extract_job_async(job_id: str, req_data: Dict[str, Any]):
    """Async implementation of the extraction logic."""
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
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
                        current_job = session.query(Job).filter(Job.id == job_id).first()
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
        job = db.query(Job).filter(Job.id == job_id).first()
        job.status = "completed"
        job.progress = 1.0
        job.result_path = output_dir
        db.commit()
        
    except Exception as e:
        logger.error(f"Error in extract job {job_id}: {e}", exc_info=True)
        try:
            job = db.query(Job).filter(Job.id == job_id).first()
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

@celery_app.task(bind=True, name="stringsight.workers.tasks.run_pipeline_job")
def run_pipeline_job(self, job_id: str, req_data: Dict[str, Any]):
    """Celery task for full pipeline (extraction + clustering)."""
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
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
            
        # Run the pipeline (sync)
        # Note: explain() handles its own sampling if sample_size is passed,
        # but here we might want to sample beforehand or pass it.
        # explain() signature: explain(df, ...)
        
        if req.sample_size and req.sample_size < len(df):
             # Simple random sampling if not using the complex sampling logic in run_full_pipeline
             # For full parity, we should use sample_prompts_evenly if available, but simple sample is ok for now
             df = df.sample(n=req.sample_size, random_state=42)
        
        clustered_df, model_stats = explain(df, **explain_kwargs)
        
        job = db.query(Job).filter(Job.id == job_id).first()
        job.status = "completed"
        job.progress = 1.0
        job.result_path = output_dir
        db.commit()
        
    except Exception as e:
        logger.error(f"Error in pipeline job {job_id}: {e}", exc_info=True)
        try:
            job = db.query(Job).filter(Job.id == job_id).first()
            if job:
                job.status = "failed"
                job.error_message = str(e)
                db.commit()
        except Exception as db_e:
            logger.error(f"Failed to update job error state: {db_e}")
    finally:
        db.close()
