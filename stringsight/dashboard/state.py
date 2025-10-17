"""
Shared application state for the StringSight Gradio viewer.

This module centralises mutable globals so they can be imported from any other
sub-module without circular-import problems.
"""
from typing import Any, Dict, Optional
import os
from pathlib import Path

# Global runtime state – mutable and shared across all tabs
app_state: Dict[str, Any] = {
    "clustered_df": None,
    # NEW canonical key for the FunctionalMetrics dict
    "metrics": None,
    # DEPRECATED alias kept temporarily so that untouched modules continue to work
    "model_stats": None,
    "results_path": None,
    "available_models": [],
    "current_results_dir": None,
    # Loading state flag to prevent redundant overview updates during data loading
    # (Currently disabled for debugging multiple loads issue)
    "is_loading_data": False,
}

# Base directory that contains experiment result folders. Can be changed at
# runtime via launch_app(results_dir=…).  A value of None means "not set".
# Defaults to the StringSight repository root directory.
def _get_default_base() -> str:
    """Get the default base directory (StringSight repository root)."""
    current_file = Path(__file__).resolve()
    # Go up from stringsight/dashboard/state.py to repo root
    repo_root = current_file.parent.parent.parent
    return str(repo_root)

_default_base = _get_default_base()
BASE_RESULTS_DIR: Optional[str] = os.getenv("BASE_RESULTS_DIR", _default_base) 