"""
Simple in-memory cache for parsed JSONL data with TTL.

This module provides caching functionality for API endpoints to avoid
repeatedly parsing large JSONL files.
"""

from __future__ import annotations

from typing import Dict, List, Any
from datetime import datetime, timedelta
from pathlib import Path
import threading
import hashlib

from stringsight.logging_config import get_logger

logger = get_logger(__name__)

_JSONL_CACHE: Dict[str, tuple[List[Dict[str, Any]], datetime]] = {}
_CACHE_TTL = timedelta(minutes=15)
_CACHE_LOCK = threading.Lock()


def _get_file_hash(path: Path) -> str:
    """Get a hash of file path and modification time for cache key."""
    stat = path.stat()
    key_str = f"{path}:{stat.st_mtime}:{stat.st_size}"
    return hashlib.md5(key_str.encode()).hexdigest()


def _read_jsonl_as_list(path: Path, nrows: int | None = None) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dicts. Optional row cap."""
    import json
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if nrows is not None and (i + 1) >= nrows:
                break
    return rows


def get_cached_jsonl(path: Path, nrows: int | None = None) -> List[Dict[str, Any]]:
    """Read JSONL file with caching. Cache key includes file mtime to auto-invalidate on changes.

    Only caches full file reads (nrows=None) to avoid cache bloat. For partial reads,
    reads directly from disk.

    Args:
        path: Path to the JSONL file
        nrows: Optional limit on number of rows to read. If provided, skips caching.

    Returns:
        List of dicts representing each line in the JSONL file
    """
    # Only cache full file reads to avoid memory bloat
    if nrows is not None:
        logger.debug(f"Partial read requested for {path.name} (nrows={nrows}), skipping cache")
        return _read_jsonl_as_list(path, nrows)

    cache_key = _get_file_hash(path)

    with _CACHE_LOCK:
        if cache_key in _JSONL_CACHE:
            cached_data, cached_time = _JSONL_CACHE[cache_key]
            # Check if cache is still valid
            if datetime.now() - cached_time < _CACHE_TTL:
                logger.debug(f"Cache hit for {path.name}")
                return cached_data
            else:
                # Remove expired entry
                del _JSONL_CACHE[cache_key]
                logger.debug(f"Cache expired for {path.name}")

    # Cache miss - read from disk
    logger.debug(f"Cache miss for {path.name}, reading from disk")
    data = _read_jsonl_as_list(path, nrows)

    # Store in cache (only if full file read)
    if nrows is None:
        with _CACHE_LOCK:
            _JSONL_CACHE[cache_key] = (data, datetime.now())

    return data
