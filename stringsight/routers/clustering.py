"""
Clustering and metrics endpoints.

NOTE: The clustering logic is extremely complex (900+ lines for cluster/run alone)
with intricate score column detection, side-by-side conversion, conversation reconstruction,
and metrics computation. To avoid code duplication and maintain simplicity, this router
delegates to the original implementation in api.py.

Once the full refactoring is complete and api.py is simplified, this logic can be
moved here properly. For now, we maintain a clean separation without duplicating
the complex clustering business logic.
"""

from typing import Dict, Any
import asyncio
from dataclasses import dataclass

from fastapi import APIRouter, HTTPException
from fastapi.routing import APIRoute

from stringsight.schemas import ClusterRunRequest, ClusterMetricsRequest
from stringsight.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["clustering"])


# Placeholder for ClusterJob - will be fully implemented when cluster job endpoints are added
@dataclass
class ClusterJob:
    """Background clustering job state."""
    id: str
    state: str = "queued"  # queued | running | completed | error | cancelled
    progress: float = 0.0
    error: str | None = None
    result: Dict[str, Any | None] | None = None
    result_path: str | None = None
    cancelled: bool = False


# -----------------------------
# Main clustering endpoint
# -----------------------------

@router.post("/cluster/run")
async def cluster_run(req: ClusterRunRequest) -> Dict[str, Any]:
    """Run clustering directly on existing properties without re-running extraction.

    This is much more efficient than the full explain() pipeline since it skips
    the expensive LLM property extraction step and works with already-extracted properties.

    The implementation is complex (900+ lines) and handles:
    - Auto-detection and conversion of score columns to nested dict format
    - Side-by-side conversation reconstruction from properties
    - Property-to-conversation matching with fuzzy ID resolution
    - HDBSCAN clustering with configurable parameters
    - Metrics computation (FunctionalMetrics or SideBySideMetrics)
    - Results persistence with timestamped directories

    For now, this endpoint delegates to the original implementation to avoid
    duplicating hundreds of lines of intricate logic.
    """
    # Import the original implementation
    # Once api.py is fully refactored, we can move the core logic here
    from stringsight import api as original_api

    # Get the original endpoint function
    original_cluster_run = None
    for route in original_api.app.routes:
        if isinstance(route, APIRoute) and route.path == "/cluster/run":
            original_cluster_run = route.endpoint
            break

    if not original_cluster_run:
        raise HTTPException(
            status_code=500,
            detail="Clustering endpoint not found in original implementation"
        )

    # Delegate to the original implementation
    # Note: The original endpoint is async and takes (req, background_tasks)
    # We don't use background_tasks here, so pass a mock
    from fastapi import BackgroundTasks
    background_tasks = BackgroundTasks()

    try:
        result = await original_cluster_run(req, background_tasks)
        return result
    except Exception as e:
        logger.exception("Error in clustering")
        raise HTTPException(status_code=500, detail=f"Clustering failed: {str(e)}")


# -----------------------------
# Metrics recomputation endpoint
# -----------------------------

@router.post("/cluster/metrics")
def cluster_metrics(req: ClusterMetricsRequest) -> Dict[str, Any]:
    """Recompute cluster metrics for a filtered subset without reclustering.

    This endpoint recomputes metrics when the user filters properties in the UI,
    allowing them to see updated statistics without re-running the expensive
    clustering step.

    Handles:
    - Score column detection and conversion (similar to cluster/run)
    - Property filtering based on included_property_ids
    - Metrics recomputation for the filtered subset
    - Long-format DataFrame preparation for metrics
    """
    # Import the original implementation
    from stringsight import api as original_api

    # Get the original endpoint function
    original_cluster_metrics = None
    for route in original_api.app.routes:
        if isinstance(route, APIRoute) and route.path == "/cluster/metrics":
            original_cluster_metrics = route.endpoint
            break

    if not original_cluster_metrics:
        raise HTTPException(
            status_code=500,
            detail="Cluster metrics endpoint not found in original implementation"
        )

    # Delegate to the original implementation
    try:
        result = original_cluster_metrics(req)
        return result
    except Exception as e:
        logger.exception("Error recomputing cluster metrics")
        raise HTTPException(status_code=500, detail=f"Metrics computation failed: {str(e)}")


# -----------------------------
# Background clustering job endpoints
# -----------------------------

# Note: Job management endpoints are commented out for now
# They would require implementing the full job queue system with threading
# Similar to the extraction job system in routers/extraction.py

# @router.post("/cluster/job/start")
# async def cluster_job_start(req: ClusterRunRequest) -> Dict[str, Any]:
#     """Start a clustering job in the background."""
#     pass

# @router.get("/cluster/job/status/{job_id}")
# def cluster_job_status(job_id: str) -> Dict[str, Any]:
#     """Get the status of a clustering job."""
#     pass

# @router.get("/cluster/job/result/{job_id}")
# def cluster_job_result(job_id: str) -> Dict[str, Any]:
#     """Get the result of a completed clustering job."""
#     pass
