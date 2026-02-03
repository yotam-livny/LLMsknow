"""
Pydantic schemas for API request/response validation.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# Dataset schemas
class DatasetInfo(BaseModel):
    id: str
    name: str
    filename: str
    output_id: str
    question_col: str
    answer_col: str
    category: str
    exists: bool
    total_samples: int
    columns: List[str]


class DatasetSample(BaseModel):
    index: int
    question: str
    answer: Optional[str] = None


class DatasetSampleFull(DatasetSample):
    raw_data: Dict[str, Any]


class DatasetSamplesResponse(BaseModel):
    dataset_id: str
    samples: List[DatasetSample]
    page: int
    page_size: int
    total_samples: int
    total_pages: int
    has_next: bool
    has_prev: bool


# Model/Combination schemas
class ProbeConfig(BaseModel):
    layer: int
    token: str


class CombinationInfo(BaseModel):
    model_id: str
    model_name: str
    model_friendly: str
    dataset_id: str
    dataset_name: str
    output_id: str
    has_answers: bool
    has_input_output_ids: bool
    has_probe: bool
    probe_config: Optional[ProbeConfig] = None
    samples_processed: int
    samples_total: int
    samples_coverage: float
    accuracy: Optional[float] = None
    status: str  # "READY" | "PARTIAL" | "NOT_PROCESSED"
    ready_for_visualization: bool
    ready_for_probe_predictions: bool


class ModelSummary(BaseModel):
    model_id: str
    display_name: str
    friendly_name: str
    total_datasets: int
    ready_datasets: int
    partial_datasets: int
    has_any_data: bool


# Inference request/response schemas
class InferenceRequest(BaseModel):
    """Request for model inference."""
    model_id: str = Field(..., description="HuggingFace model ID")
    question: str = Field(..., description="Input question/prompt")
    expected_answer: Optional[str] = Field(None, description="Expected answer for correctness check")
    dataset_id: Optional[str] = Field(None, description="Source dataset (for metadata)")
    sample_idx: Optional[int] = Field(None, description="Sample index in dataset")
    max_new_tokens: int = Field(100, description="Maximum tokens to generate")
    extract_layers: bool = Field(True, description="Whether to extract layer representations")
    extract_attention: bool = Field(True, description="Whether to extract attention patterns")


class TokenInfo(BaseModel):
    """Information about a single token."""
    id: int
    text: str
    position: int
    is_input: bool = True


class LayerStats(BaseModel):
    """Statistics for a single layer."""
    mean: float
    std: float
    min: float
    max: float
    norm: float


class AttentionHeadStats(BaseModel):
    """Statistics for an attention head."""
    entropy: float
    sparsity: float
    max_attention: float
    mean_self_attention: float


class ProbePrediction(BaseModel):
    """Prediction from a trained probe."""
    layer: int
    token: str
    prediction: int  # 0 = correct, 1 = incorrect
    confidence: float
    probabilities: List[float]


class TokenAlternative(BaseModel):
    """A single alternative token option."""
    token_id: int
    token_text: str
    probability: float


class InferenceResponse(BaseModel):
    """Full inference result with optional extractions."""
    # Basic info
    model_id: str
    question: str
    generated_answer: str
    expected_answer: Optional[str] = None
    
    # Token information
    tokens: List[TokenInfo]
    input_token_count: int
    output_token_count: int
    total_token_count: int
    
    # Token alternatives (top-k for each generated token)
    token_alternatives: Optional[List[List[TokenAlternative]]] = None
    
    # Correctness info
    actual_correct: Optional[bool] = None  # Objective correctness if expected_answer provided
    
    # Probe predictions (if available)
    probe_predictions: Optional[Dict[int, ProbePrediction]] = None  # At last INPUT token
    probe_predictions_output: Optional[Dict[int, ProbePrediction]] = None  # At last OUTPUT token
    probe_available: bool = False
    
    # Extraction flags
    has_layer_data: bool = False
    has_attention_data: bool = False


class LayerDataResponse(BaseModel):
    """Layer representations for visualization."""
    model_id: str
    layers: Dict[int, List[List[float]]]  # layer_idx -> (seq_len, hidden_size)
    layer_stats: Dict[int, LayerStats]
    seq_len: int
    num_layers: int
    hidden_size: int


class AttentionDataResponse(BaseModel):
    """Attention patterns for visualization."""
    model_id: str
    patterns: Dict[int, Dict[int, List[List[float]]]]  # layer -> head -> (seq_len, seq_len)
    statistics: Dict[int, Dict[int, AttentionHeadStats]]
    seq_len: int
    num_layers: int
    num_heads: int


class ModelLoadRequest(BaseModel):
    """Request to load a model."""
    model_id: str
    use_quantization: bool = False


class ModelStatusResponse(BaseModel):
    """Current model status."""
    loaded: bool
    model_id: Optional[str] = None
    device: str
    
    
# Health check
class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"


# Correctness Evolution schemas
class ExactAnswerInfo(BaseModel):
    """Information about extracted exact answer tokens."""
    exact_answer: Optional[str] = None
    start_char: int = -1
    end_char: int = -1
    extraction_method: str  # "direct" | "llm"
    valid: bool = False
    token_positions: List[int] = []  # Positions in the output sequence


class LayerProbeResult(BaseModel):
    """Probe prediction at a specific layer."""
    layer: int
    prediction: int  # 0 = correct, 1 = incorrect
    confidence: float  # Confidence in the prediction
    prob_correct: float  # P(correct)
    prob_incorrect: float  # P(incorrect)


class CorrectnessEvolutionResponse(BaseModel):
    """
    Response containing correctness evolution across layers.
    Shows how the model's internal "belief" about correctness evolves.
    """
    # Basic info
    question: str
    generated_answer: str
    expected_answer: Optional[str] = None
    actual_correct: Optional[bool] = None
    
    # Exact answer extraction
    exact_answer: ExactAnswerInfo
    
    # Token position used for measurement
    measured_token_position: int = -1
    measured_token_text: str = ""
    
    # Layer-by-layer probe predictions at the measured token
    layer_predictions: List[LayerProbeResult]
    
    # Summary statistics
    first_confident_layer: Optional[int] = None  # First layer with >70% confidence
    peak_confidence_layer: int = -1
    peak_confidence: float = 0.0
    
    # Interpretation
    interpretation: str = ""


class LogitLensToken(BaseModel):
    """A token prediction from logit lens."""
    token_id: int
    token_text: str
    probability: float


class LogitLensLayerResult(BaseModel):
    """Logit lens result for a single layer."""
    layer: int
    top_tokens: List[LogitLensToken]
    target_token_rank: Optional[int] = None  # Rank of the actual generated token
    target_token_prob: Optional[float] = None  # Probability of actual token


class LogitLensResponse(BaseModel):
    """Response for logit lens analysis."""
    token_position: int  # Position of the token being predicted
    token_text: str  # Text of the token being predicted
    token_id: int  # ID of the token being predicted
    prediction_position: int  # Position of hidden state used for prediction (token_position - 1)
    prediction_token_text: str  # Token at prediction position
    layers: List[LogitLensLayerResult]


class LogitLensRequest(BaseModel):
    """Request for logit lens analysis."""
    token_position: int
    top_k: int = 5


class CorrectnessEvolutionRequest(BaseModel):
    """Request for correctness evolution analysis."""
    # Can use existing inference result or provide new question
    use_current_session: bool = True
    question: Optional[str] = None
    expected_answer: Optional[str] = None
    # Optional: specify token position to measure at (default: last exact answer token)
    token_position: Optional[int] = None


# Error response
class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    code: Optional[str] = None
