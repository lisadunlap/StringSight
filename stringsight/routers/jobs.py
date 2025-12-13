import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request, Response, BackgroundTasks
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional
from uuid import UUID
import uuid

from stringsight.database import get_db
from stringsight.models.job import Job
from stringsight.schemas import ExtractJobStartRequest, PipelineJobRequest, ClusterJobRequest
from stringsight.workers.tasks import run_extract_job_inprocess, run_pipeline_job_inprocess, _run_cluster_job_async
from stringsight.storage.adapter import get_storage_adapter
from stringsight.utils.paths import _get_results_dir
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])

@router.post("/pipeline")
async def start_pipeline_job(
    req: PipelineJobRequest,
    response: Response,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    # Create job record
    job_id = uuid.uuid4()
    job = Job(
        id=job_id,
        user_id=None,
        status="queued",
        progress=0.0,
        job_type="pipeline"
    )
    db.add(job)
    db.commit()
    
    # Inject email from current_user if not provided
    # if not req.email and current_user and current_user.email:
    #     req.email = current_user.email
    #     logger.info(f"Injecting email {req.email} for job {job_id}")
    
    # Convert request to dict for serialization
    req_data = req.dict()
    
    # Enqueue task (in-process for easy installs; avoids requiring Celery/Redis)
    background_tasks.add_task(run_pipeline_job_inprocess, str(job_id), req_data)
    
    return {"job_id": str(job_id), "status": "queued", "job_type": "pipeline"}

@router.post("/extract")
async def start_extract_job(
    req: ExtractJobStartRequest,
    response: Response,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    # Create job record
    job_id = uuid.uuid4()
    job = Job(
        id=job_id,
        user_id=None,
        status="queued",
        progress=0.0
    )
    db.add(job)
    db.commit()
    
    # Inject email from current_user if not provided
    # if not req.email and current_user and current_user.email:
    #     req.email = current_user.email
    #     logger.info(f"Injecting email {req.email} for job {job_id}")
    
    # Convert request to dict for serialization
    req_data = req.dict()
    
    # Enqueue task (in-process for easy installs; avoids requiring Celery/Redis)
    background_tasks.add_task(run_extract_job_inprocess, str(job_id), req_data)
    
    return {"job_id": str(job_id), "status": "queued"}

@router.post("/cluster")
def start_cluster_job(
    req: ClusterJobRequest,
    response: Response,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    # Create job record
    job_id = uuid.uuid4()
    job = Job(
        id=job_id,
        user_id=None,
        status="queued",
        progress=0.0,
        job_type="cluster"
    )
    db.add(job)
    db.commit()
    
    # Inject email from current_user if not provided
    # if not req.email and current_user and current_user.email:
    #     req.email = current_user.email
    #     logger.info(f"Injecting email {req.email} for job {job_id}")
    
    # Convert request to dict for serialization
    req_data = req.dict()
    
    # Schedule the coroutine directly; FastAPI/Starlette will await it as a background task.
    background_tasks.add_task(_run_cluster_job_async, str(job_id), req_data)
    
    return {"job_id": str(job_id), "status": "queued", "job_type": "cluster"}

@router.get("/{job_id}")
def get_job_status(
    job_id: UUID,
    db: Session = Depends(get_db)
):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Check ownership only if user is logged in and job has a user
    # if current_user and job.user_id and job.user_id != current_user.id:
    #     raise HTTPException(status_code=403, detail="Not authorized to access this job")
        
    return {
        "id": str(job.id),
        "status": job.status,
        "progress": job.progress,
        "result_path": job.result_path,
        "error_message": job.error_message,
        "created_at": job.created_at
    }

@router.get("/{job_id}/results")
def get_job_results(
    job_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Fetch the actual results content from the job's result_path.
    Returns the properties extracted by the job.
    """
    import json
    from pathlib import Path
    
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Check ownership only if user is logged in and job has a user
    # if current_user and job.user_id and job.user_id != current_user.id:
    #     raise HTTPException(status_code=403, detail="Not authorized to access this job")
    
    if job.status != "completed":
        raise HTTPException(status_code=400, detail=f"Job not completed yet. Status: {job.status}")
    
    if not job.result_path:
        raise HTTPException(status_code=404, detail="No results available for this job")
 
    # Read the results from storage (works with both filesystem and S3)
    from stringsight.utils.paths import _get_results_dir
    from pathlib import Path

    storage = get_storage_adapter()

    # Resolve result_path relative to results directory if it's not absolute
    result_path_obj = Path(job.result_path)
    if not result_path_obj.is_absolute():
        results_base = _get_results_dir()
        full_result_path = results_base / job.result_path
    else:
        full_result_path = result_path_obj

    result_file_path = str(full_result_path / "validated_properties.jsonl")

    if not storage.exists(result_file_path):
        raise HTTPException(status_code=404, detail=f"Results file not found: {result_file_path}")

    try:
        # Read JSONL file using storage adapter
        properties = storage.read_jsonl(result_file_path)

        return {
            "properties": properties,
            "result_path": job.result_path,
            "count": len(properties)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read results: {str(e)}")
