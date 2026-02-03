"""
Dataset manager for loading and browsing datasets.
"""
import pandas as pd
from functools import lru_cache
from typing import Dict, Any, Optional, List

from config import DATA_DIR, DATASET_CONFIG, DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE
from utils.logging import get_logger

logger = get_logger("dataset_manager")


@lru_cache(maxsize=16)
def load_dataset(dataset_id: str) -> pd.DataFrame:
    """
    Load a dataset CSV with LRU caching.
    
    Args:
        dataset_id: Dataset identifier from DATASET_CONFIG
        
    Returns:
        DataFrame with dataset contents
        
    Raises:
        ValueError: If dataset_id is not found in config
        FileNotFoundError: If dataset file doesn't exist
    """
    if dataset_id not in DATASET_CONFIG:
        logger.error(f"Unknown dataset: {dataset_id}")
        raise ValueError(f"Unknown dataset: {dataset_id}")
    
    config = DATASET_CONFIG[dataset_id]
    filepath = DATA_DIR / config["filename"]
    
    if not filepath.exists():
        logger.error(f"Dataset file not found: {filepath}")
        raise FileNotFoundError(f"Dataset file not found: {filepath}")
    
    logger.info(f"Loading dataset: {dataset_id} from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} rows from {dataset_id}")
    
    return df


def get_dataset_info(dataset_id: str) -> Dict[str, Any]:
    """
    Get metadata about a dataset.
    
    Args:
        dataset_id: Dataset identifier
        
    Returns:
        Dictionary with dataset metadata
    """
    if dataset_id not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset: {dataset_id}")
    
    config = DATASET_CONFIG[dataset_id]
    filepath = DATA_DIR / config["filename"]
    
    info = {
        "id": dataset_id,
        "name": config["display_name"],
        "filename": config["filename"],
        "output_id": config["output_id"],
        "question_col": config["question_col"],
        "answer_col": config["answer_col"],
        "category": config["category"],
        "exists": filepath.exists(),
        "total_samples": 0,
        "columns": [],
    }
    
    if filepath.exists():
        df = load_dataset(dataset_id)
        info["total_samples"] = len(df)
        info["columns"] = list(df.columns)
    
    return info


def list_datasets() -> List[Dict[str, Any]]:
    """
    List all available datasets with their metadata.
    
    Returns:
        List of dataset info dictionaries
    """
    datasets = []
    for dataset_id in DATASET_CONFIG:
        try:
            info = get_dataset_info(dataset_id)
            if info["exists"]:
                datasets.append(info)
        except Exception as e:
            logger.warning(f"Error getting info for {dataset_id}: {e}")
    
    return datasets


def get_dataset_samples(
    dataset_id: str,
    page: int = 1,
    page_size: int = DEFAULT_PAGE_SIZE,
    search_query: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get paginated samples from a dataset.
    
    Args:
        dataset_id: Dataset identifier
        page: Page number (1-indexed)
        page_size: Number of samples per page
        search_query: Optional search string to filter samples
        
    Returns:
        Dictionary with samples and pagination info
    """
    if page_size > MAX_PAGE_SIZE:
        page_size = MAX_PAGE_SIZE
    if page < 1:
        page = 1
    
    df = load_dataset(dataset_id)
    config = DATASET_CONFIG[dataset_id]
    question_col = config["question_col"]
    answer_col = config["answer_col"]
    
    # Apply search filter if provided
    if search_query:
        search_query = search_query.lower()
        mask = df[question_col].astype(str).str.lower().str.contains(search_query, na=False)
        if answer_col in df.columns:
            mask |= df[answer_col].astype(str).str.lower().str.contains(search_query, na=False)
        df = df[mask]
    
    total_samples = len(df)
    total_pages = (total_samples + page_size - 1) // page_size
    
    # Get page slice
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    page_df = df.iloc[start_idx:end_idx]
    
    # Convert to list of sample dicts
    samples = []
    for idx, row in page_df.iterrows():
        sample = {
            "index": int(idx),
            "question": str(row.get(question_col, "")),
            "answer": str(row.get(answer_col, "")) if answer_col in row else None,
        }
        samples.append(sample)
    
    return {
        "dataset_id": dataset_id,
        "samples": samples,
        "page": page,
        "page_size": page_size,
        "total_samples": total_samples,
        "total_pages": total_pages,
        "has_next": page < total_pages,
        "has_prev": page > 1,
    }


def get_sample_by_index(dataset_id: str, index: int) -> Dict[str, Any]:
    """
    Get a specific sample by its index.
    
    Args:
        dataset_id: Dataset identifier
        index: Row index in the dataset
        
    Returns:
        Sample dictionary with all columns
    """
    df = load_dataset(dataset_id)
    
    if index < 0 or index >= len(df):
        raise ValueError(f"Index {index} out of range for dataset {dataset_id}")
    
    config = DATASET_CONFIG[dataset_id]
    row = df.iloc[index]
    
    return {
        "index": index,
        "question": str(row.get(config["question_col"], "")),
        "answer": str(row.get(config["answer_col"], "")) if config["answer_col"] in row else None,
        "raw_data": row.to_dict(),
    }
