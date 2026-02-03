"""
Availability scanner for detecting which model/dataset combinations have pre-computed data.
"""
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional

from config import (
    OUTPUT_DIR, CHECKPOINTS_DIR, DATA_DIR,
    SUPPORTED_MODELS, DATASET_CONFIG
)
from utils.logging import get_logger

logger = get_logger("availability_scanner")


def get_available_combinations() -> List[Dict[str, Any]]:
    """
    Scan output/ and checkpoints/ to find all available model/dataset combinations.
    Returns list of combinations with their availability status.
    
    IMPORTANT: The existing pipeline uses 'output_id' for file naming, not dataset_id.
    For example: movie_qa_train -> "movies" in output filenames.
    """
    combinations = []
    
    for model_id, model_config in SUPPORTED_MODELS.items():
        model_friendly = model_config["friendly_name"]
        
        for dataset_id, dataset_config in DATASET_CONFIG.items():
            output_id = dataset_config["output_id"]
            
            # Check for answers file (uses output_id)
            answers_file = OUTPUT_DIR / f"{model_friendly}-answers-{output_id}.csv"
            has_answers = answers_file.exists()
            
            # Check for input_output_ids file (uses output_id)
            ids_file = OUTPUT_DIR / f"{model_friendly}-input_output_ids-{output_id}.pt"
            has_ids = ids_file.exists()
            
            # Check for probe checkpoint (uses output_id, matching probe.py naming)
            # Pattern: clf_{model_friendly}_{output_id}_layer-{N}_token-{type}.pkl
            probe_files = list(CHECKPOINTS_DIR.glob(
                f"clf_{model_friendly}_{output_id}_layer-*_token-*.pkl"
            ))
            has_probe = len(probe_files) > 0
            
            # Get probe config if exists
            probe_config = None
            if has_probe:
                # Parse first probe filename to get layer/token info
                probe_name = probe_files[0].stem
                parts = probe_name.split("_")
                for i, part in enumerate(parts):
                    if part.startswith("layer-"):
                        layer = int(part.replace("layer-", ""))
                    if part.startswith("token-"):
                        token = "_".join(parts[i:]).replace("token-", "")
                        break
                probe_config = {"layer": layer, "token": token}
            
            # Count processed samples (from output file)
            sample_count = 0
            if has_answers:
                try:
                    df = pd.read_csv(answers_file)
                    sample_count = len(df)
                except Exception as e:
                    logger.warning(f"Error reading {answers_file}: {e}")
            
            # Get total available samples from raw dataset
            total_available = 0
            raw_file = DATA_DIR / dataset_config["filename"]
            if raw_file.exists():
                try:
                    raw_df = pd.read_csv(raw_file)
                    total_available = len(raw_df)
                except Exception as e:
                    logger.warning(f"Error reading {raw_file}: {e}")
            
            # Calculate accuracy if answers file has correctness column
            accuracy = None
            if has_answers and sample_count > 0:
                try:
                    df = pd.read_csv(answers_file)
                    if "automatic_correctness" in df.columns:
                        accuracy = df["automatic_correctness"].mean()
                except Exception as e:
                    logger.warning(f"Error calculating accuracy: {e}")
            
            # Determine status
            if has_answers and has_ids:
                status = "READY"
            elif has_answers or has_ids:
                status = "PARTIAL"
            else:
                status = "NOT_PROCESSED"
            
            combinations.append({
                "model_id": model_id,
                "model_name": model_config["display_name"],
                "model_friendly": model_friendly,
                "dataset_id": dataset_id,
                "dataset_name": dataset_config["display_name"],
                "output_id": output_id,
                "has_answers": has_answers,
                "has_input_output_ids": has_ids,
                "has_probe": has_probe,
                "probe_config": probe_config,
                "samples_processed": sample_count,
                "samples_total": total_available,
                "samples_coverage": round(sample_count / total_available, 2) if total_available > 0 else 0,
                "accuracy": round(accuracy, 3) if accuracy is not None else None,
                "status": status,
                "ready_for_visualization": has_answers and has_ids,
                "ready_for_probe_predictions": has_probe,
            })
    
    return combinations


def get_combinations_for_model(model_id: str) -> List[Dict[str, Any]]:
    """Get available combinations for a specific model."""
    all_combinations = get_available_combinations()
    return [c for c in all_combinations if c["model_id"] == model_id]


def get_combinations_for_dataset(dataset_id: str) -> List[Dict[str, Any]]:
    """Get available combinations for a specific dataset."""
    all_combinations = get_available_combinations()
    return [c for c in all_combinations if c["dataset_id"] == dataset_id]


def get_ready_combinations() -> List[Dict[str, Any]]:
    """Get only combinations that are ready for visualization."""
    all_combinations = get_available_combinations()
    return [c for c in all_combinations if c["status"] == "READY"]


def get_models_summary() -> List[Dict[str, Any]]:
    """Get summary of available models with their dataset availability."""
    combinations = get_available_combinations()
    
    models = {}
    for model_id, model_config in SUPPORTED_MODELS.items():
        model_combos = [c for c in combinations if c["model_id"] == model_id]
        ready_count = len([c for c in model_combos if c["status"] == "READY"])
        partial_count = len([c for c in model_combos if c["status"] == "PARTIAL"])
        
        models[model_id] = {
            "model_id": model_id,
            "display_name": model_config["display_name"],
            "friendly_name": model_config["friendly_name"],
            "total_datasets": len(model_combos),
            "ready_datasets": ready_count,
            "partial_datasets": partial_count,
            "has_any_data": ready_count > 0 or partial_count > 0,
        }
    
    return list(models.values())
