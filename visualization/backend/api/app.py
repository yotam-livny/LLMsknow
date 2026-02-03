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
    DatasetSamplesResponse, CombinationInfo, ModelSummary, HealthResponse,
    InferenceRequest, InferenceResponse, TokenInfo, LayerStats,
    LayerDataResponse, AttentionDataResponse, AttentionHeadStats,
    ModelLoadRequest, ModelStatusResponse, ProbePrediction, ErrorResponse
)
from core.dataset_manager import (
    list_datasets, get_dataset_info, get_dataset_samples, get_sample_by_index
)
from core.availability_scanner import (
    get_available_combinations, get_ready_combinations,
    get_combinations_for_model, get_models_summary
)
from core.model_manager import ModelManager
from core.layer_extractor import LayerExtractor
from core.attention_extractor import AttentionExtractor
from core.probe_runner import ProbeRunner, get_probe_info

# Global storage for current inference session
_current_session = {
    "inference_result": None,
    "layer_data": None,
    "attention_data": None,
}

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


@app.get("/api/models/{model_id:path}/combinations", response_model=List[CombinationInfo], tags=["Models"])
async def get_model_combinations(model_id: str):
    """Get available dataset combinations for a specific model."""
    # URL decode the model_id (handles slashes like mistralai/Mistral-7B-Instruct-v0.2)
    from urllib.parse import unquote
    model_id = unquote(model_id)
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


# Model management endpoints
@app.get("/api/model/status", response_model=ModelStatusResponse, tags=["Model"])
async def get_model_status():
    """Get current model loading status."""
    mm = ModelManager()
    return ModelStatusResponse(
        loaded=mm._current_model_id is not None,
        model_id=mm._current_model_id,
        device=mm.get_device()
    )


@app.post("/api/model/load", response_model=ModelStatusResponse, tags=["Model"])
async def load_model(request: ModelLoadRequest):
    """Load a model onto the GPU/device."""
    try:
        logger.info(f"Loading model: {request.model_id}")
        mm = ModelManager()
        mm.load_model(request.model_id, request.use_quantization)
        return ModelStatusResponse(
            loaded=True,
            model_id=request.model_id,
            device=mm.get_device()
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@app.post("/api/model/unload", response_model=ModelStatusResponse, tags=["Model"])
async def unload_model():
    """Unload the current model to free memory."""
    mm = ModelManager()
    mm.unload_model()
    return ModelStatusResponse(
        loaded=False,
        model_id=None,
        device=mm.get_device()
    )


# Inference endpoints
@app.post("/api/inference", response_model=InferenceResponse, tags=["Inference"])
async def run_inference(request: InferenceRequest):
    """
    Run model inference and optionally extract layer representations and attention.
    """
    global _current_session
    
    logger.info(f"Starting inference with model={request.model_id}")
    
    mm = ModelManager()
    
    # Load model if not already loaded
    if not mm.is_model_loaded(request.model_id):
        logger.info(f"Model {request.model_id} not loaded, loading now...")
        try:
            mm.load_model(request.model_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
    
    try:
        model, tokenizer = mm.get_model_and_tokenizer()
        
        # Tokenize input
        input_ids = mm.tokenize(request.question)
        input_length = input_ids.shape[1]
        
        # Generate response with scores for alternatives
        output = mm.generate(
            input_ids,
            max_new_tokens=request.max_new_tokens,
            output_attentions=request.extract_attention,
            output_hidden_states=request.extract_layers,
            output_scores=True  # Capture logits for alternative tokens
        )
        
        # Get generated sequence
        generated_ids = output.sequences[0]
        generated_answer = mm.decode(generated_ids[input_length:])
        
        # Get top-k alternatives for each generated token
        token_alternatives = None
        if hasattr(output, 'scores') and output.scores:
            try:
                token_alternatives = mm.get_top_k_alternatives(output.scores, k=5)
            except Exception as e:
                logger.warning(f"Failed to get token alternatives: {e}")
        
        # Build token info
        tokens = []
        for i, token_id in enumerate(generated_ids.tolist()):
            token_text = tokenizer.decode([token_id])
            tokens.append(TokenInfo(
                id=token_id,
                text=token_text,
                position=i,
                is_input=(i < input_length)
            ))
        
        total_length = len(generated_ids)
        output_length = total_length - input_length
        
        # Extract layer representations if requested
        layer_data = None
        layer_stats = {}
        if request.extract_layers:
            logger.debug("Extracting layer representations...")
            extractor = LayerExtractor(model, request.model_id)
            layer_result = extractor.extract_all_layers(generated_ids.unsqueeze(0))
            layer_stats = extractor.compute_layer_statistics(
                {k: layer_result["layers"][k] for k in layer_result["layers"]}
            )
            layer_data = {
                "model_id": request.model_id,
                "layers": layer_result["layers"],
                "layer_stats": {k: LayerStats(**v) for k, v in layer_stats.items()},
                "seq_len": layer_result["seq_len"],
                "num_layers": SUPPORTED_MODELS[request.model_id]["num_layers"],
                "hidden_size": SUPPORTED_MODELS[request.model_id]["hidden_size"]
            }
            _current_session["layer_data"] = layer_data
        
        # Extract attention if requested
        attention_data = None
        if request.extract_attention:
            logger.debug("Extracting attention patterns...")
            try:
                attn_extractor = AttentionExtractor(model, request.model_id)
                attn_result = attn_extractor.extract_attention_patterns(generated_ids.unsqueeze(0))
                
                # Only set attention_data if we got actual patterns
                if attn_result["patterns"]:
                    attention_data = {
                        "model_id": request.model_id,
                        "patterns": attn_result["patterns"],
                        "statistics": {
                            layer_idx: {
                                head_idx: AttentionHeadStats(**head_stats)
                                for head_idx, head_stats in heads.items()
                            }
                            for layer_idx, heads in attn_result["statistics"].items()
                        },
                        "seq_len": attn_result["seq_len"],
                        "num_layers": SUPPORTED_MODELS[request.model_id]["num_layers"],
                        "num_heads": SUPPORTED_MODELS[request.model_id]["num_heads"]
                    }
                    _current_session["attention_data"] = attention_data
                else:
                    logger.warning("Attention extraction returned empty patterns")
            except Exception as e:
                logger.warning(f"Attention extraction failed: {e}")
        
        # Check actual correctness if expected answer provided
        actual_correct = None
        if request.expected_answer:
            from core.correctness import compute_correctness
            actual_correct = compute_correctness(
                dataset_id=request.dataset_id or "default",
                model_answer=generated_answer,
                expected_answer=request.expected_answer,
                wrong_answer=getattr(request, 'wrong_answer', None)
            )
        
        # Try to run probe predictions if available
        probe_predictions = None
        probe_available = False
        if request.dataset_id and layer_data:
            try:
                probe_info = get_probe_info(request.model_id, request.dataset_id)
                if probe_info["available"]:
                    probe_available = True
                    runner = ProbeRunner(request.model_id, request.dataset_id)
                    probes = runner.list_available_probes()
                    
                    # Use the last input token for predictions (common choice)
                    token_idx = input_length - 1
                    
                    # Get predictions from all available probes
                    import numpy as np
                    probe_predictions = {}
                    for probe in probes:
                        try:
                            layer_idx = probe["layer"]
                            layer_rep = np.array(layer_data["layers"][layer_idx])
                            token_rep = layer_rep[token_idx]
                            pred = runner.predict(token_rep, layer_idx, probe["token"])
                            probe_predictions[layer_idx] = ProbePrediction(**pred)
                        except Exception as e:
                            logger.warning(f"Failed probe at layer {probe['layer']}: {e}")
            except Exception as e:
                logger.warning(f"Failed to run probes: {e}")
        
        # Build response
        # Convert alternatives to schema format
        formatted_alternatives = None
        if token_alternatives:
            from api.schemas import TokenAlternative
            formatted_alternatives = [
                [TokenAlternative(**alt) for alt in step_alts]
                for step_alts in token_alternatives
            ]
        
        response = InferenceResponse(
            model_id=request.model_id,
            question=request.question,
            generated_answer=generated_answer,
            expected_answer=request.expected_answer,
            tokens=tokens,
            input_token_count=input_length,
            output_token_count=output_length,
            total_token_count=total_length,
            token_alternatives=formatted_alternatives,
            actual_correct=actual_correct,
            probe_predictions=probe_predictions,
            probe_available=probe_available,
            has_layer_data=(layer_data is not None),
            has_attention_data=(attention_data is not None)
        )
        
        _current_session["inference_result"] = response
        
        logger.info(f"Inference complete: {output_length} tokens generated")
        return response
        
    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.get("/api/inference/layers", response_model=LayerDataResponse, tags=["Inference"])
async def get_layer_data():
    """Get layer representations from the last inference."""
    if _current_session["layer_data"] is None:
        raise HTTPException(
            status_code=404, 
            detail="No layer data available. Run inference with extract_layers=true first."
        )
    return LayerDataResponse(**_current_session["layer_data"])


@app.get("/api/inference/attention", response_model=AttentionDataResponse, tags=["Inference"])
async def get_attention_data():
    """Get attention patterns from the last inference."""
    if _current_session["attention_data"] is None:
        raise HTTPException(
            status_code=404,
            detail="No attention data available. Run inference with extract_attention=true first."
        )
    return AttentionDataResponse(**_current_session["attention_data"])


# Probe endpoints
@app.get("/api/probes/{model_id}/{dataset_id}", tags=["Probes"])
async def get_probes(model_id: str, dataset_id: str):
    """Get information about available probes for a model/dataset combination."""
    if model_id not in SUPPORTED_MODELS:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    
    return get_probe_info(model_id, dataset_id)


# Main entry point
if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting API server on {API_HOST}:{API_PORT}")
    uvicorn.run(app, host=API_HOST, port=API_PORT)
