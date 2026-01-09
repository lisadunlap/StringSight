"""Project-wide constants.

This module centralizes defaults that are shared across the backend and scripts
to avoid drift (e.g., mismatched defaults between API schemas, routers, and CLIs).
"""

from __future__ import annotations

# Default parallelism for LLM/embedding calls across the project.
# This value is used when a caller does not explicitly pass `max_workers`.
DEFAULT_MAX_WORKERS: int = 16






