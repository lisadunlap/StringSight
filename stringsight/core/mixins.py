"""
Mixins for common pipeline stage functionality.

These mixins provide reusable functionality that can be composed into pipeline stages.
"""

import time
import logging
from typing import Any, Dict, Optional
from functools import wraps
from tqdm import tqdm


class LoggingMixin:
    """Mixin for consistent logging across pipeline stages."""
    
    def __init__(self, *, verbose: bool = True, **kwargs):
        self.verbose = verbose
        self.logger = logging.getLogger(self.__class__.__name__)
        # Forward all remaining kwargs so downstream mixins (e.g., WandbMixin)
        # receive configuration such as `use_wandb`, `wandb_project`, etc.
        super().__init__(**kwargs)
        
    def log(self, message: str, level: str = "info") -> None:
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            getattr(self.logger, level.lower())(f"[{self.name}] {message}")
    
    def log_progress(self, iterable, desc: str = "Processing"):
        """Create a progress bar for an iterable."""
        if self.verbose:
            return tqdm(iterable, desc=f"[{self.name}] {desc}")
        return iterable


class CacheMixin:
    """Mixin for caching expensive operations."""
    
    def __init__(self, *args, use_cache: bool = True, **kwargs):
        # Remove our kwargs before passing to super()
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['use_cache']}
        super().__init__(*args, **filtered_kwargs)
        self.use_cache = use_cache
        self._cache = {}
    
    def cache_key(self, *args, **kwargs) -> str:
        """Generate a cache key from arguments."""
        return f"{args}_{kwargs}"
    
    def get_cached(self, key: str) -> Optional[Any]:
        """Get a cached value."""
        if not self.use_cache:
            return None
        return self._cache.get(key)
    
    def set_cached(self, key: str, value: Any) -> None:
        """Set a cached value."""
        if self.use_cache:
            self._cache[key] = value
    
    def clear_cache(self) -> None:
        """Clear the cache."""
        self._cache.clear()


class ErrorHandlingMixin:
    """Mixin for consistent error handling across pipeline stages."""
    
    def __init__(self, *, fail_fast: bool = False, **kwargs):
        self.fail_fast = fail_fast
        self.errors    = []
        # Forward remaining kwargs (e.g., wandb flags) to next mixin
        super().__init__(**kwargs)
    
    def handle_error(self, error: Exception, context: str = "") -> None:
        """Handle an error based on the fail_fast setting."""
        error_info = {
            'error': error,
            'context': context,
            'stage': self.name,
            'timestamp': time.time()
        }
        self.errors.append(error_info)
        
        if hasattr(self, 'log'):
            self.log(f"Error in {context}: {error}", level="error")
        
        if self.fail_fast:
            raise error
    
    def get_errors(self) -> list:
        """Get all errors encountered during processing."""
        return self.errors
    
    def clear_errors(self) -> None:
        """Clear the error list."""
        self.errors.clear()


class WandbMixin:
    """Mixin for Weights & Biases logging."""
    
    def __init__(self, *, use_wandb: bool = True, wandb_project: str = None, **_):
        self.use_wandb     = use_wandb
        self.wandb_project = wandb_project
        self._wandb_ok     = False
        self._summary_metrics = {}  # Accumulate summary metrics
        super().__init__()
    
    def init_wandb(self, project: str = None, run_name: str = None, **kwargs) -> None:
        """Initialize wandb if enabled."""
        if not self.use_wandb:
            return
            
        try:
            import wandb
            # import weave
            # Check if wandb is already initialized globally
            if wandb.run is not None:
                # Mark that wandb is available for this stage
                self._wandb_ok = True
                if hasattr(self, 'log'):
                    self.log(f"Using existing wandb run: {wandb.run.name}", level="debug")
                return
                
            # Check if this stage already marked wandb as available
            if self._wandb_ok:
                return
                
            # Only initialize if no existing run
            if run_name is None:
                run_name = f"{self.name}_{int(time.time())}"
                
            wandb.init(
                project=project or self.wandb_project or "StringSight",
                name=run_name,
                reinit=False,  # Don't reinitialize if already exists
                **kwargs
            )
            self._wandb_ok = True
            if hasattr(self, 'log'):
                self.log(f"Initialized wandb run: {run_name}", level="debug")
        except Exception as e:
            if hasattr(self, 'log'):
                self.log(f"Failed to initialize wandb: {e}", level="warning")
            # If wandb initialization fails, continue without it
            self.use_wandb = False
    
    def log_wandb(self, data: Dict[str, Any], step: int = None, is_summary: bool = False) -> None:
        """
        Log data to wandb.
        
        Args:
            data: Dictionary of data to log
            step: Optional step number for time series data
            is_summary: If True, accumulate as summary statistics instead of logging immediately
        """
        if not self.use_wandb:
            return
            
        try:
            import wandb
            # import weave
            # Check if wandb is available globally
            if wandb.run is not None:
                try:
                    if is_summary:
                        # Accumulate summary metrics for later logging
                        self._summary_metrics.update(data)
                    else:
                        # Log immediately for non-summary data (tables, artifacts, etc.)
                        wandb.log(data, step=step)
                    # Mark that wandb is working for this stage
                    self._wandb_ok = True
                except Exception as e:
                    if hasattr(self, 'log'):
                        self.log(f"Failed to log to wandb: {e}", level="warning")
            else:
                # If no global wandb run, log warning
                if hasattr(self, 'log'):
                    self.log("wandb not initialized, skipping logging", level="warning")
        except ImportError:
            # wandb not installed or not available
            self.use_wandb = False
    
    def log_summary_metrics(self) -> None:
        """Log accumulated summary metrics to wandb run summary."""
        if not self.use_wandb or not self._summary_metrics:
            return
            
        try:
            import wandb
            # import weave
            # Check if wandb is available globally
            if wandb.run is not None:
                try:
                    # Log summary metrics to run summary (not as regular metrics)
                    for key, value in self._summary_metrics.items():
                        wandb.run.summary[key] = value
                    
                    if hasattr(self, 'log'):
                        self.log(f"Logged {len(self._summary_metrics)} summary metrics to wandb", level="debug")
                    
                    # Clear accumulated metrics after logging
                    self._summary_metrics.clear()
                    
                except Exception as e:
                    if hasattr(self, 'log'):
                        self.log(f"Failed to log summary metrics to wandb: {e}", level="warning")
            else:
                if hasattr(self, 'log'):
                    self.log("wandb not initialized, skipping summary logging", level="warning")
        except ImportError:
            # wandb not installed or not available
            self.use_wandb = False
    
    def get_summary_metrics(self) -> Dict[str, Any]:
        """Get accumulated summary metrics without logging them."""
        return self._summary_metrics.copy()
    
    def clear_summary_metrics(self) -> None:
        """Clear accumulated summary metrics."""
        self._summary_metrics.clear()
    
    def log_artifact(self, artifact_name: str, artifact_type: str, file_path: str) -> None:
        """Log an artifact to wandb."""
        if not self.use_wandb:
            return
            
        try:
            import wandb
            # import weave
            # Check if wandb is available globally
            if wandb.run is not None:
                try:
                    artifact = wandb.Artifact(artifact_name, type=artifact_type)
                    artifact.add_file(file_path)
                    wandb.log_artifact(artifact)
                    # Mark that wandb is working for this stage
                    self._wandb_ok = True
                except Exception as e:
                    if hasattr(self, 'log'):
                        self.log(f"Failed to log artifact to wandb: {e}", level="warning")
            else:
                # If no global wandb run, log warning
                if hasattr(self, 'log'):
                    self.log("wandb not initialized, skipping artifact logging", level="warning")
        except ImportError:
            # wandb not installed or not available
            self.use_wandb = False


class TimingMixin:
    """Mixin for timing stage execution."""
    
    def __init__(self, **kwargs):
        self._start = None
        # Forward remaining kwargs to next mixin
        super().__init__(**kwargs)
    
    def start_timer(self) -> None:
        """Start timing the execution."""
        self._start = time.time()
    
    def end_timer(self) -> float:
        """End timing and return the execution time."""
        if self._start is None:
            return 0.0
        execution_time = time.time() - self._start
        return execution_time
    
    def get_execution_time(self) -> float:
        """Get the last execution time."""
        return self.end_timer() or 0.0


def timed_stage(func):
    """Decorator to automatically time stage execution."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if hasattr(self, 'start_timer'):
            self.start_timer()
        result = func(self, *args, **kwargs)
        if hasattr(self, 'end_timer'):
            execution_time = self.end_timer()
            if hasattr(self, 'log'):
                self.log(f"Execution time: {execution_time:.2f}s")
        return result
    return wrapper 