"""
StringSight API Routers.

This package contains modular FastAPI routers for different endpoint categories.
Each router handles a specific domain (health, validation, clustering, etc.).
"""

from stringsight.routers import (
    health,
    validation,
    clustering,
    extraction,
    dataframe,
    prompts,
    explain,
)

__all__ = [
    "health",
    "validation",
    "clustering",
    "extraction",
    "dataframe",
    "prompts",
    "explain",
]
