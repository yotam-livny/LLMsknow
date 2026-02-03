"""
FastAPI application for the LLMsKnow visualization tool.
"""
import sys
from pathlib import Path

# Add backend directory to path for imports
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional

from config import API_HOST, API_PORT, CORS_ORIGINS, SUPPORTED_MODELS
from utils.logging import get_logger
from api.schemas import (
    DatasetInfo, DatasetSample, DatasetSampleFull,
    DatasetSamplesResponse, CombinationInfo, ModelSummary, HealthResponse
)
from core.dataset_manager import (
    list_datasets, get_dataset_info, get_dataset_samples, get_sample_by_index
)
from core.availability_scanner import (
    get_available_combinations, get_ready_combinations,
    get_combinations_for_model, get_models_summary
)

logger = get_logger("api")

# Create FastAPI app
app = FastAPI(
    title="LLMsKnow Visualization API",
    description="API for exploring LLM layer representations and probe predictions",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse()


# Models endpoints
@app.get("/api/models", response_model=List[ModelSummary], tags=["Models"])
async def list_models():
    """List all supported models with their data availability summary."""
    logger.info("Listing models")
    return get_models_summary()


@app.get("/api/models/{model_id}/combinations", response_model=List[CombinationInfo], tags=["Models"])
async def get_model_combinations(model_id: str):
    """Get available dataset combinations for a specific model."""
    if model_id not in SUPPORTED_MODELS:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    
    logger.info(f"Getting combinations for model: {model_id}")
    return get_combinations_for_model(model_id)


# Combinations endpoints
@app.get("/api/combinations", response_model=List[CombinationInfo], tags=["Combinations"])
async def list_combinations(
    model_id: Optional[str] = Query(None, description="Filter by model ID"),
    ready_only: bool = Query(False, description="Only show READY combinations"),
):
    """List all available model/dataset combinations."""
    logger.info(f"Listing combinations (model={model_id}, ready_only={ready_only})")
    
    if ready_only:
        combinations = get_ready_combinations()
    else:
        combinations = get_available_combinations()
    
    if model_id:
        combinations = [c for c in combinations if c["model_id"] == model_id]
    
    return combinations


# Datasets endpoints
@app.get("/api/datasets", response_model=List[DatasetInfo], tags=["Datasets"])
async def get_datasets():
    """List all available datasets."""
    logger.info("Listing datasets")
    return list_datasets()


@app.get("/api/datasets/{dataset_id}", response_model=DatasetInfo, tags=["Datasets"])
async def get_dataset(dataset_id: str):
    """Get information about a specific dataset."""
    try:
        logger.info(f"Getting dataset info: {dataset_id}")
        return get_dataset_info(dataset_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/datasets/{dataset_id}/samples", response_model=DatasetSamplesResponse, tags=["Datasets"])
async def get_samples(
    dataset_id: str,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Samples per page"),
    search: Optional[str] = Query(None, description="Search query"),
):
    """Get paginated samples from a dataset."""
    try:
        logger.info(f"Getting samples from {dataset_id} (page={page}, search={search})")
        return get_dataset_samples(dataset_id, page, page_size, search)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/datasets/{dataset_id}/samples/{index}", response_model=DatasetSampleFull, tags=["Datasets"])
async def get_sample(dataset_id: str, index: int):
    """Get a specific sample by index."""
    try:
        logger.info(f"Getting sample {index} from {dataset_id}")
        return get_sample_by_index(dataset_id, index)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# Main entry point
if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting API server on {API_HOST}:{API_PORT}")
    uvicorn.run(app, host=API_HOST, port=API_PORT)
