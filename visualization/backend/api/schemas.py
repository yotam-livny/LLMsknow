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


# Health check
class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
