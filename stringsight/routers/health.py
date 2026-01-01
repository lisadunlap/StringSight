"""
Health check and debug endpoints.

Simple endpoints for monitoring server status and debugging.
"""

from typing import Dict, Any
import os

from fastapi import APIRouter

from stringsight.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> Dict[str, bool]:
    """Basic health check endpoint."""
    logger.debug("BACKEND: Health check called")
    return {"ok": True}


@router.get("/api/health")
def api_health() -> Dict[str, bool]:
    """Health check alias at /api/health to match frontend expectations."""
    logger.debug("BACKEND: API Health check called")
    return {"ok": True}


@router.get("/embedding-models")
def get_embedding_models() -> Dict[str, Any]:
    """Return a curated list of embedding model identifiers.

    Later we can make this dynamic via config/env. Keep it simple for now.
    """
    models = [
        "openai/text-embedding-3-large",
        "openai/text-embedding-3-large",
        "bge-m3",
        "sentence-transformers/all-MiniLM-L6-v2",
    ]
    return {"models": models}


@router.get("/debug")
def debug() -> Dict[str, Any]:
    """Debug endpoint to verify server is running."""
    if os.environ.get("STRINGSIGHT_DEBUG") in ("1", "true", "True"):
        logger.debug("BACKEND: Debug endpoint called")
    return {"status": "server_running", "message": "Backend is alive!"}


@router.post("/debug/post")
def debug_post(body: Dict[str, Any]) -> Dict[str, Any]:
    """Debug POST endpoint to test request/response cycle."""
    if os.environ.get("STRINGSIGHT_DEBUG") in ("1", "true", "True"):
        logger.debug(f"BACKEND: Debug POST called with keys: {list(body.keys())}")
    return {"status": "post_working", "received_keys": list(body.keys())}
