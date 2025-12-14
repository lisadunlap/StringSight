from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Literal

class ExtractBatchRequest(BaseModel):
    rows: List[Dict[str, Any]]
    method: Optional[Literal["single_model", "side_by_side"]] = None
    system_prompt: Optional[str] = None
    task_description: Optional[str] = None
    model_name: Optional[str] = "gpt-4.1"
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    max_tokens: Optional[int] = 16000
    max_workers: Optional[int] = 128
    include_scores_in_prompt: Optional[bool] = False
    use_wandb: Optional[bool] = False
    output_dir: Optional[str] = None
    return_debug: Optional[bool] = False
    sample_size: Optional[int] = None

class ExtractJobStartRequest(ExtractBatchRequest):
    pass

class PipelineJobRequest(BaseModel):
    # Data input (can be rows or a path if we supported it, but for API usually rows)
    rows: List[Dict[str, Any]]
    
    # Pipeline config
    method: Optional[Literal["single_model", "side_by_side"]] = "single_model"
    system_prompt: Optional[str] = None
    task_description: Optional[str] = None
    
    # Prompt expansion config
    prompt_expansion: Optional[bool] = False
    expansion_num_traces: Optional[int] = 5
    expansion_model: Optional[str] = "gpt-4.1"
    
    # Clustering config
    clusterer: Optional[str] = "hdbscan"
    min_cluster_size: Optional[int] = 15
    embedding_model: Optional[str] = "text-embedding-3-large"
    
    # Models
    extraction_model: Optional[str] = "gpt-4.1"
    summary_model: Optional[str] = None
    cluster_assignment_model: Optional[str] = None
    
    # Execution
    max_workers: Optional[int] = 64
    use_wandb: Optional[bool] = False
    sample_size: Optional[int] = None
    
    # Columns
    groupby_column: Optional[str] = "behavior_type"
    assign_outliers: Optional[bool] = False
    score_columns: Optional[List[str]] = None

    # Output
    output_dir: Optional[str] = None

class ClusterParams(BaseModel):
    minClusterSize: Optional[int] = 5
    embeddingModel: str = "openai/text-embedding-3-large"
    groupBy: Optional[str] = "none"  # none | category | behavior_type

class ClusterJobRequest(BaseModel):
    # Data
    properties: List[Dict[str, Any]]
    operationalRows: List[Dict[str, Any]]
    
    # Clustering params
    params: ClusterParams
    method: Optional[Literal["single_model", "side_by_side"]] = "single_model"
    score_columns: Optional[List[str]] = None

    # Output
    output_dir: Optional[str] = None

