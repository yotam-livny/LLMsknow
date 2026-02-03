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
    ModelLoadRequest, ModelStatusResponse, ProbePrediction, ErrorResponse,
    CorrectnessEvolutionRequest, CorrectnessEvolutionResponse,
    ExactAnswerInfo, LayerProbeResult
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
from core.exact_answer_extractor import ExactAnswerExtractor

# Global storage for current inference session
_current_session = {
    "inference_result": None,
    "layer_data": None,
    "attention_data": None,
    "dataset_id": None,
    "correctness_evolution": None,
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
        probe_predictions = None  # At last input token
        probe_predictions_output = None  # At last output token
        probe_available = False
        if request.dataset_id and layer_data:
            try:
                probe_info = get_probe_info(request.model_id, request.dataset_id)
                if probe_info["available"]:
                    probe_available = True
                    runner = ProbeRunner(request.model_id, request.dataset_id)
                    probes = runner.list_available_probes()
                    
                    import numpy as np
                    
                    # Position 1: Last input token (before generation)
                    input_token_idx = input_length - 1
                    probe_predictions = {}
                    for probe in probes:
                        try:
                            layer_idx = probe["layer"]
                            layer_rep = np.array(layer_data["layers"][layer_idx])
                            token_rep = layer_rep[input_token_idx]
                            pred = runner.predict(token_rep, layer_idx, probe["token"])
                            probe_predictions[layer_idx] = ProbePrediction(**pred)
                        except Exception as e:
                            logger.warning(f"Failed probe at layer {probe['layer']} (input): {e}")
                    
                    # Position 2: Last output token (after full generation)
                    output_token_idx = total_length - 1  # Last generated token
                    probe_predictions_output = {}
                    for probe in probes:
                        try:
                            layer_idx = probe["layer"]
                            layer_rep = np.array(layer_data["layers"][layer_idx])
                            if output_token_idx < len(layer_rep):
                                token_rep = layer_rep[output_token_idx]
                                pred = runner.predict(token_rep, layer_idx, probe["token"])
                                probe_predictions_output[layer_idx] = ProbePrediction(**pred)
                        except Exception as e:
                            logger.warning(f"Failed probe at layer {probe['layer']} (output): {e}")
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
            probe_predictions_output=probe_predictions_output,
            probe_available=probe_available,
            has_layer_data=(layer_data is not None),
            has_attention_data=(attention_data is not None)
        )
        
        _current_session["inference_result"] = response
        _current_session["dataset_id"] = request.dataset_id
        
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


# Correctness Evolution endpoint
@app.post("/api/inference/correctness-evolution", response_model=CorrectnessEvolutionResponse, tags=["Inference"])
async def get_correctness_evolution(request: CorrectnessEvolutionRequest):
    """
    Analyze how the model's correctness "belief" evolves across layers.
    
    This endpoint:
    1. Extracts "exact answer" tokens from the generated response
    2. Runs probes at all available layers for those token positions
    3. Returns the evolution of correctness confidence
    
    Based on: "LLMs Know More Than They Show" - truthfulness information
    is concentrated in exact answer tokens.
    """
    global _current_session
    
    logger.info("Computing correctness evolution")
    
    # Get inference result from session or require new inference
    inference_result = _current_session.get("inference_result")
    layer_data = _current_session.get("layer_data")
    
    if not inference_result or not layer_data:
        raise HTTPException(
            status_code=400,
            detail="No inference data available. Run /api/inference first with extract_layers=true."
        )
    
    mm = ModelManager()
    if not mm._current_model_id:
        raise HTTPException(status_code=400, detail="No model loaded")
    
    try:
        # Extract exact answer tokens
        extractor = ExactAnswerExtractor(mm)
        exact_result = extractor.extract_exact_answer(
            question=inference_result.question,
            model_answer=inference_result.generated_answer,
            expected_answer=inference_result.expected_answer,
            is_correct=inference_result.actual_correct
        )
        
        logger.info(f"Extracted exact answer: {exact_result}")
        
        # Find token positions in the output
        token_positions = []
        if exact_result["valid"] and exact_result["exact_answer"] != "NO ANSWER":
            token_positions = extractor.get_exact_answer_token_indices_in_output(
                tokens=[t.model_dump() for t in inference_result.tokens],
                exact_answer=exact_result["exact_answer"],
                generated_answer=inference_result.generated_answer
            )
        
        exact_answer_info = ExactAnswerInfo(
            exact_answer=exact_result.get("exact_answer"),
            start_char=exact_result.get("start_char", -1),
            end_char=exact_result.get("end_char", -1),
            extraction_method=exact_result.get("extraction_method", "unknown"),
            valid=exact_result.get("valid", False),
            token_positions=token_positions
        )
        
        # Run probes at all available layers
        layer_predictions = []
        
        # Determine which token position to measure at
        # Priority: 1) User-specified, 2) Last exact answer token, 3) Last input token
        if request.token_position is not None:
            target_token_idx = request.token_position
        elif token_positions:
            target_token_idx = token_positions[-1]  # Last exact answer token
        else:
            target_token_idx = inference_result.input_token_count - 1  # Last input token
        
        # Get the token text at this position
        measured_token_text = ""
        if target_token_idx < len(inference_result.tokens):
            measured_token_text = inference_result.tokens[target_token_idx].text
        
        logger.info(f"Measuring correctness at token position {target_token_idx}: '{measured_token_text}'")
        
        if inference_result.probe_available:
            dataset_id = getattr(inference_result, 'dataset_id', None)
            if not dataset_id:
                # Try to infer from session
                dataset_id = _current_session.get("dataset_id", "movie_qa_train")
            
            try:
                runner = ProbeRunner(mm._current_model_id, dataset_id)
                available_probes = runner.list_available_probes()
                
                # Collect available layers
                import numpy as np
                layers_data = layer_data.get("layers", {})
                
                for probe in available_probes:
                    layer_idx = probe["layer"]
                    token_type = probe["token"]
                    
                    if layer_idx not in layers_data:
                        continue
                    
                    layer_rep = np.array(layers_data[layer_idx])
                    if target_token_idx < len(layer_rep):
                        token_rep = layer_rep[target_token_idx]
                        
                        try:
                            pred = runner.predict(token_rep, layer_idx, token_type)
                            layer_predictions.append(LayerProbeResult(
                                layer=layer_idx,
                                prediction=pred["prediction"],
                                confidence=pred["confidence"],
                                prob_correct=pred["probabilities"][0],
                                prob_incorrect=pred["probabilities"][1]
                            ))
                        except Exception as e:
                            logger.warning(f"Probe prediction failed at layer {layer_idx}: {e}")
            except Exception as e:
                logger.warning(f"Failed to run probes: {e}")
        
        # Sort by layer
        layer_predictions.sort(key=lambda x: x.layer)
        
        # Compute summary statistics
        first_confident_layer = None
        peak_confidence_layer = -1
        peak_confidence = 0.0
        
        for pred in layer_predictions:
            confidence = pred.prob_correct if pred.prediction == 0 else pred.prob_incorrect
            if first_confident_layer is None and confidence > 0.7:
                first_confident_layer = pred.layer
            if confidence > peak_confidence:
                peak_confidence = confidence
                peak_confidence_layer = pred.layer
        
        # Generate interpretation
        interpretation = _generate_interpretation(
            exact_result, layer_predictions, 
            inference_result.actual_correct,
            first_confident_layer
        )
        
        return CorrectnessEvolutionResponse(
            question=inference_result.question,
            generated_answer=inference_result.generated_answer,
            expected_answer=inference_result.expected_answer,
            actual_correct=inference_result.actual_correct,
            exact_answer=exact_answer_info,
            measured_token_position=target_token_idx,
            measured_token_text=measured_token_text,
            layer_predictions=layer_predictions,
            first_confident_layer=first_confident_layer,
            peak_confidence_layer=peak_confidence_layer,
            peak_confidence=peak_confidence,
            interpretation=interpretation
        )
        
    except Exception as e:
        logger.error(f"Correctness evolution failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Correctness evolution failed: {str(e)}")


def _generate_interpretation(
    exact_result: dict,
    layer_predictions: list,
    actual_correct: bool,
    first_confident_layer: int
) -> str:
    """Generate a human-readable interpretation of the correctness evolution."""
    
    if not exact_result.get("valid"):
        return "Could not extract exact answer tokens from the response."
    
    if exact_result.get("exact_answer") == "NO ANSWER":
        return "Model did not provide a clear answer to the question."
    
    if not layer_predictions:
        return "No probe predictions available. Train probes at multiple layers for detailed analysis."
    
    # Single layer only
    if len(layer_predictions) == 1:
        pred = layer_predictions[0]
        prob_correct = pred.prob_correct * 100
        if pred.prediction == 0:  # Thinks correct
            if actual_correct:
                return f"Model's internal representation at layer {pred.layer} indicates it believes the answer is correct ({prob_correct:.1f}% confidence), which matches reality."
            else:
                return f"⚠️ Model's internal representation at layer {pred.layer} indicates it believes the answer is correct ({prob_correct:.1f}% confidence), but the answer is actually incorrect. This suggests miscalibration."
        else:  # Thinks incorrect
            if actual_correct:
                return f"⚠️ Model's internal representation at layer {pred.layer} indicates uncertainty ({100-prob_correct:.1f}% thinks incorrect), but the answer is actually correct."
            else:
                return f"Model's internal representation at layer {pred.layer} encodes that the answer is likely incorrect ({100-prob_correct:.1f}% confidence), which matches reality."
    
    # Multiple layers - show evolution
    early_layers = [p for p in layer_predictions if p.layer < 16]
    late_layers = [p for p in layer_predictions if p.layer >= 16]
    
    if first_confident_layer is not None:
        return f"Model develops confidence around layer {first_confident_layer}. " \
               f"This suggests the 'correctness signal' emerges in {'early' if first_confident_layer < 16 else 'mid-to-late'} layers."
    
    return "Model shows varying confidence across layers. See the visualization for detailed evolution."


# Main entry point
if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting API server on {API_HOST}:{API_PORT}")
    uvicorn.run(app, host=API_HOST, port=API_PORT)
