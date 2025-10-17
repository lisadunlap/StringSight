"""
Pipeline orchestration for StringSight.

The Pipeline class manages the execution of multiple pipeline stages in sequence.
"""

from typing import List, Dict, Any, Optional
import time
from .core.stage import PipelineStage
from .core.data_objects import PropertyDataset
from .core.mixins import LoggingMixin, TimingMixin, ErrorHandlingMixin, WandbMixin


class Pipeline(LoggingMixin, TimingMixin, ErrorHandlingMixin, WandbMixin):
    """
    A pipeline for processing data through multiple stages.
    
    The Pipeline class coordinates the execution of multiple pipeline stages,
    handles error recovery, and provides logging and timing information.
    """
    
    def __init__(self, name: str, stages: List[PipelineStage] = None, **kwargs):
        """
        Initialize a new Pipeline.
        
        Args:
            name: Name of the pipeline
            stages: List of pipeline stages to execute
            **kwargs: Additional configuration options
        """
        # Set name first, before calling parent __init__ methods that might use it
        self.name = name
        self.stages = stages or []
        self.stage_times = {}
        self.stage_errors = {}
        # Store output directory (if any) so that we can automatically persist
        # intermediate pipeline results after each stage.  This enables tooling
        # such as compute_metrics_only() to pick up from any point in the
        # pipeline without the caller having to remember to save explicitly.
        self.output_dir = kwargs.get('output_dir')
        
        # Now call parent __init__ methods safely
        super().__init__(**kwargs)
        
        # Initialize wandb if enabled (after all parent inits are done)
        if hasattr(self, 'use_wandb') and self.use_wandb:
            self.init_wandb()
        
        # Mark all stages as using the same wandb run
        for stage in self.stages:
            if hasattr(stage, 'use_wandb') and stage.use_wandb:
                stage._wandb_ok = True  # Mark that wandb is available
        
    def add_stage(self, stage: PipelineStage) -> None:
        """Add a stage to the end of the pipeline."""
        self.stages.append(stage)
        
        # Mark the new stage as using the same wandb run if wandb is enabled
        if hasattr(self, 'use_wandb') and self.use_wandb and hasattr(stage, 'use_wandb') and stage.use_wandb:
            stage._wandb_ok = True  # Mark that wandb is available
        
    def insert_stage(self, index: int, stage: PipelineStage) -> None:
        """Insert a stage at a specific position in the pipeline."""
        self.stages.insert(index, stage)
        
        # Mark the inserted stage as using the same wandb run if wandb is enabled
        if hasattr(self, 'use_wandb') and self.use_wandb and hasattr(stage, 'use_wandb') and stage.use_wandb:
            stage._wandb_ok = True  # Mark that wandb is available
        
    def remove_stage(self, index: int) -> PipelineStage:
        """Remove and return a stage at a specific position."""
        return self.stages.pop(index)
        
    def run(self, data: PropertyDataset) -> PropertyDataset:
        """
        Execute all stages in the pipeline.
        
        Args:
            data: Input PropertyDataset
            
        Returns:
            PropertyDataset after processing through all stages
        """
        self.log(f"Starting pipeline '{self.name}' with {len(self.stages)} stages")
        self.start_timer()
        
        # Count initial models
        initial_models = set()
        for conv in data.conversations:
            if isinstance(conv.model, list):
                initial_models.update(conv.model)
            else:
                initial_models.add(conv.model)
        
        print(f"\n🚀 Starting pipeline '{self.name}'")
        print(f"   • Input conversations: {len(data.conversations)}")
        print(f"   • Input models: {len(initial_models)}")
        if len(initial_models) <= 20:
            model_list = sorted(list(initial_models))
            print(f"   • Model names: {', '.join(model_list)}")
        print()
        
        current_data = data
        
        for i, stage in enumerate(self.stages):
            stage_start_time = time.time()
            
            # try:
            self.log(f"Running stage {i+1}/{len(self.stages)}: {stage.name}")
            
            # Execute the stage
            current_data = stage(current_data)
            
            # Track timing
            stage_execution_time = time.time() - stage_start_time
            self.stage_times[stage.name] = stage_execution_time
            
            self.log(f"Stage {stage.name} completed in {stage_execution_time:.2f}s")
            
            # Log stage-specific metrics
            self._log_stage_metrics(stage, current_data)

            # --------------------------------------------------------------
            # 📝  Auto-save full dataset snapshot after each stage
            # --------------------------------------------------------------
            if getattr(self, "output_dir", None):
                from pathlib import Path
                import os

                # Ensure the directory exists (mkdir ‑p semantics)
                Path(self.output_dir).mkdir(parents=True, exist_ok=True)

                # File name pattern: full_dataset_after_<idx>_<stage>.json
                # snapshot_name = (
                #     f"full_dataset_after_{i+1}_{stage.name.replace(' ', '_').lower()}.json"
                # )
                snapshot_name = f"full_dataset.json"
                snapshot_path = os.path.join(self.output_dir, snapshot_name)

                # Persist using the JSON format for maximum portability
                current_data.save(snapshot_path)

                if getattr(self, "verbose", False):
                    print(f"   • Saved dataset snapshot: {snapshot_path}")
                
            # except Exception as e:
            #     self.stage_errors[stage.name] = str(e)
            #     self.handle_error(e, f"stage {i+1} ({stage.name})")
                
        total_time = self.end_timer()
        self.log(f"Pipeline '{self.name}' completed in {total_time:.2f}s")
        
        # Print final summary
        final_models = set()
        for conv in current_data.conversations:
            if isinstance(conv.model, list):
                final_models.update(conv.model)
            else:
                final_models.add(conv.model)
        
        print(f"\n🎉 Pipeline '{self.name}' completed!")
        print(f"   • Total execution time: {total_time:.2f}s")
        print(f"   • Final conversations: {len(current_data.conversations)}")
        print(f"   • Final properties: {len(current_data.properties)}")
        print(f"   • Final models: {len(final_models)}")
        if current_data.clusters:
            print(f"   • Final clusters: {len(current_data.clusters)}")
        if current_data.model_stats:
            print(f"   • Models with final stats: {len(current_data.model_stats)}")
        print()
        
        return current_data
    
    def _log_stage_metrics(self, stage: PipelineStage, data: PropertyDataset) -> None:
        """Log metrics for a completed stage."""
        metrics = {
            'conversations': len(data.conversations),
            'properties': len(data.properties),
            'clusters': len(data.clusters),
            'models_in_stats': len(data.model_stats)
        }
        
        # Count unique models from conversations
        unique_models = set()
        for conv in data.conversations:
            if isinstance(conv.model, list):
                unique_models.update(conv.model)
            else:
                unique_models.add(conv.model)
        
        total_models = len(unique_models)
        
        # Add model count to metrics
        metrics['total_models'] = total_models
        
        self.log(f"Stage {stage.name} metrics: {metrics}")
        
        # Print specific model count information
        print(f"\n📊 Stage '{stage.name}' completed:")
        print(f"   • Total conversations: {len(data.conversations)}")
        print(f"   • Total properties: {len(data.properties)}")
        print(f"   • Total models: {total_models}")
        if data.clusters:
            print(f"   • Total clusters: {len(data.clusters)}")
        if data.model_stats:
            print(f"   • Models with stats: {len(data.model_stats)}")
        
        # Show model names if verbose
        if hasattr(self, 'verbose') and self.verbose and total_models <= 20:
            model_list = sorted(list(unique_models))
            print(f"   • Models: {', '.join(model_list)}")
        
        print()  # Add spacing
        
        # Log to wandb as summary metrics (not regular metrics)
        if hasattr(self, 'log_wandb'):
            wandb_data = {f"{stage.name}_{k}": v for k, v in metrics.items()}
            wandb_data[f"{stage.name}_execution_time"] = self.stage_times.get(stage.name, 0)
            self.log_wandb(wandb_data, is_summary=True)
    
    def log_final_summary(self) -> None:
        """Log all accumulated summary metrics to wandb."""
        if hasattr(self, 'log_summary_metrics'):
            # Add pipeline-level summary metrics
            pipeline_summary = {
                'pipeline_total_stages': len(self.stages),
                'pipeline_total_time': self.get_execution_time(),
                'pipeline_success': len(self.stage_errors) == 0,
                'pipeline_error_count': len(self.stage_errors)
            }
            self.log_wandb(pipeline_summary, is_summary=True)
            
            # Log all accumulated summary metrics
            self.log_summary_metrics()
            
            if hasattr(self, 'log'):
                self.log("Logged final summary metrics to wandb", level="debug")
    
    def get_stage_summary(self) -> Dict[str, Any]:
        """Get a summary of pipeline execution."""
        return {
            'total_stages': len(self.stages),
            'total_time': self.get_execution_time(),
            'stage_times': self.stage_times,
            'stage_errors': self.stage_errors,
            'success': len(self.stage_errors) == 0
        }
    
    def validate_pipeline(self) -> List[str]:
        """
        Validate that the pipeline is correctly configured.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        if not self.stages:
            errors.append("Pipeline has no stages")
            
        for i, stage in enumerate(self.stages):
            if not isinstance(stage, PipelineStage):
                errors.append(f"Stage {i} is not a PipelineStage instance")
                
        return errors
    
    def __repr__(self) -> str:
        stage_names = [stage.name for stage in self.stages]
        return f"Pipeline({self.name}, stages={stage_names})"
    
    def __len__(self) -> int:
        return len(self.stages)
    
    def __getitem__(self, index: int) -> PipelineStage:
        return self.stages[index]
    
    def __iter__(self):
        return iter(self.stages)


class PipelineBuilder:
    """
    Builder pattern for constructing pipelines.
    
    Makes it easy to construct pipelines with method chaining.
    """
    
    def __init__(self, name: str = "Pipeline"):
        self.name = name
        self.stages = []
        self.config = {}
        
    def add_stage(self, stage: PipelineStage) -> "PipelineBuilder":
        """Add a stage to the pipeline."""
        self.stages.append(stage)
        return self
        
    def extract_properties(self, extractor: PipelineStage) -> "PipelineBuilder":
        """Add a property extraction stage."""
        return self.add_stage(extractor)
        
    def parse_properties(self, parser: PipelineStage) -> "PipelineBuilder":
        """Add a property parsing stage."""
        return self.add_stage(parser)
        
    def cluster_properties(self, clusterer: PipelineStage) -> "PipelineBuilder":
        """Add a clustering stage."""
        return self.add_stage(clusterer)
        
    def compute_metrics(self, metrics: PipelineStage) -> "PipelineBuilder":
        """Add a metrics computation stage."""
        return self.add_stage(metrics)
        
    def configure(self, **kwargs) -> "PipelineBuilder":
        """Set configuration options for the pipeline."""
        self.config.update(kwargs)
        return self
        
    def build(self) -> Pipeline:
        """Build the pipeline."""
        return Pipeline(self.name, self.stages, **self.config) 