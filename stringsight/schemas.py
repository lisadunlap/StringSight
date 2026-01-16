from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Literal
from stringsight.constants import DEFAULT_MAX_WORKERS


class PromptsMetadata(BaseModel):
    """Metadata about prompts used during extraction."""
    discovery_prompt: str
    clustering_prompt: Optional[str] = None
    dedup_prompt: Optional[str] = None
    outlier_prompt: Optional[str] = None
    expanded_task_description: Optional[str] = None
    task_description_original: Optional[str] = None
    dynamic_prompts_used: bool
    verification_passed: Optional[bool] = None
    reflection_attempts: Optional[int] = None

class ExtractBatchRequest(BaseModel):
    rows: List[Dict[str, Any]]
    method: Optional[Literal["single_model", "side_by_side"]] = None
    system_prompt: Optional[str] = None
    task_description: Optional[str] = None
    model_name: Optional[str] = "gpt-4.1"
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    max_tokens: Optional[int] = 16000
    max_workers: Optional[int] = DEFAULT_MAX_WORKERS
    include_scores_in_prompt: Optional[bool] = False
    use_wandb: Optional[bool] = False
    output_dir: Optional[str] = None
    return_debug: Optional[bool] = False
    sample_size: Optional[int] = None
    use_dynamic_prompts: Optional[bool] = True
    dynamic_prompt_samples: Optional[int] = 10
    custom_clustering_prompts: Optional[Dict[str, str]] = None  # {clustering, deduplication, outlier}

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
    max_workers: Optional[int] = DEFAULT_MAX_WORKERS
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

class LabelRequest(BaseModel):
    """Request for fixed-taxonomy labeling pipeline."""
    rows: List[Dict[str, Any]]
    taxonomy: Dict[str, str]  # Label name -> description

    # Column mapping
    prompt_column: Optional[str] = "prompt"
    model_column: Optional[str] = "model"
    model_response_column: Optional[str] = "model_response"
    question_id_column: Optional[str] = None

    # LLM config (defaults optimized for labeling)
    model_name: Optional[str] = "gpt-4.1"
    temperature: Optional[float] = 0.0
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = 2048
    max_workers: Optional[int] = DEFAULT_MAX_WORKERS

    # Data preparation
    sample_size: Optional[int] = None
    score_columns: Optional[List[str]] = None

    # Metrics config
    metrics_kwargs: Optional[Dict[str, Any]] = None

    # Logging & output
    use_wandb: Optional[bool] = False
    wandb_project: Optional[str] = None
    verbose: Optional[bool] = False
    output_dir: Optional[str] = None
    extraction_cache_dir: Optional[str] = None
    metrics_cache_dir: Optional[str] = None


class LabelPromptRequest(BaseModel):
    """Request to get the system prompt for labeling with a given taxonomy."""
    taxonomy: Dict[str, str]  # Label name -> description


class RowsPayload(BaseModel):
    rows: List[Dict[str, Any]]
    method: Optional[Literal["single_model", "side_by_side"]] = None


class ReadRequest(BaseModel):
    """Request body for reading a dataset from the server filesystem."""
    path: str
    method: Optional[Literal["single_model", "side_by_side"]] = None
    limit: Optional[int] = None


class ListRequest(BaseModel):
    path: str
    exts: Optional[List[str]] = None


class ResultsLoadRequest(BaseModel):
    """Request to load a results directory from the server filesystem."""
    path: str
    max_conversations: Optional[int] = None
    max_properties: Optional[int] = None
    conversations_page: int = 1
    conversations_per_page: int = 100
    properties_page: int = 1
    properties_per_page: int = 100


class ExtractSingleRequest(BaseModel):
    row: Dict[str, Any]
    sample_rows: Optional[List[Dict[str, Any]]] = None  # Optional k sample rows for dynamic prompt generation
    method: Optional[Literal["single_model", "side_by_side"]] = None
    system_prompt: Optional[str] = None
    task_description: Optional[str] = None
    model_name: Optional[str] = "gpt-4.1"
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    max_tokens: Optional[int] = 16000
    max_workers: Optional[int] = DEFAULT_MAX_WORKERS
    include_scores_in_prompt: Optional[bool] = False
    use_wandb: Optional[bool] = False
    output_dir: Optional[str] = None
    return_debug: Optional[bool] = False
    use_dynamic_prompts: Optional[bool] = True
    dynamic_prompt_samples: Optional[int] = 5


class GeneratePromptsRequest(BaseModel):
    """Request for generating dynamic prompts without extraction."""
    rows: List[Dict[str, Any]]
    method: Optional[Literal["single_model", "side_by_side"]] = None
    task_description: Optional[str] = None
    num_samples: Optional[int] = 5
    model: Optional[str] = "gpt-4.1"
    output_dir: Optional[str] = None
    seed: Optional[int] = 42


class DFRows(BaseModel):
    rows: List[Dict[str, Any]]


class DFSelectRequest(DFRows):
    include: Dict[str, List[Any]] = {}
    exclude: Dict[str, List[Any]] = {}


class DFGroupPreviewRequest(DFRows):
    by: str
    numeric_cols: Optional[List[str]] = None


class DFCustomRequest(DFRows):
    code: str


class ClusterRunParams(BaseModel):
    minClusterSize: Optional[int] = None
    embeddingModel: str = "openai/text-embedding-3-large"
    groupBy: Optional[str] = "none"


class ClusterRunRequest(BaseModel):
    operationalRows: List[Dict[str, Any]]
    properties: List[Dict[str, Any]]
    params: ClusterRunParams
    output_dir: Optional[str] = None
    score_columns: Optional[List[str]] = None
    method: Optional[str] = "single_model"


class ClusterMetricsRequest(BaseModel):
    clusters: List[Dict[str, Any]]
    properties: List[Dict[str, Any]]
    operationalRows: List[Dict[str, Any]]
    included_property_ids: Optional[List[str]] = None
    score_columns: Optional[List[str]] = None
    method: Optional[str] = "single_model"


class TidyRow(BaseModel):
    """A single tidy row for single-model data."""
    question_id: Optional[str] = None
    prompt: str
    model: str
    model_response: Any
    score: Optional[Dict[str, Optional[float]]] = None

    class Config:
        extra = "allow"


class ExplainSideBySideTidyRequest(BaseModel):
    """Request payload to run side-by-side analysis from tidy rows."""
    method: Literal["side_by_side"]
    model_a: str
    model_b: str
    data: List[TidyRow]
    score_columns: Optional[List[str]] = None
    sample_size: Optional[int] = None
    output_dir: Optional[str] = None
