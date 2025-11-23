import os
from pathlib import Path
from stringsight.logging_config import get_logger

logger = get_logger(__name__)

def _get_persistent_data_dir() -> Path:
    """Get the base directory for persistent data (results, cache) on Render.
    
    If RENDER_DISK_PATH is set, use that as the base for all persistent data.
    Otherwise, default to the current working directory (local development).
    """
    render_disk = os.environ.get("RENDER_DISK_PATH")
    if render_disk:
        base = Path(render_disk).resolve()
        logger.info(f"Using Render persistent disk: {base}")
        return base
    return Path.cwd()

def _get_results_dir() -> Path:
    """Get the results directory, potentially on persistent disk."""
    base = _get_persistent_data_dir()
    return base / "results"

def _get_cache_dir() -> Path:
    """Get the cache directory, potentially on persistent disk."""
    # Check if RENDER_DISK_PATH is set and STRINGSIGHT_CACHE_DIR is not explicitly set
    # If so, automatically configure cache to use the persistent disk
    if os.environ.get("RENDER_DISK_PATH") and not os.environ.get("STRINGSIGHT_CACHE_DIR"):
        base = _get_persistent_data_dir()
        cache_dir = base / ".cache" / "stringsight"
        # Set the environment variable so the Cache class picks it up
        os.environ["STRINGSIGHT_CACHE_DIR"] = str(cache_dir)
        logger.info(f"Auto-configured cache directory to use persistent disk: {cache_dir}")
        return cache_dir
    # Otherwise, let Cache class handle it using STRINGSIGHT_CACHE_DIR env var or default
    return Path.cwd() / ".cache" / "stringsight"
