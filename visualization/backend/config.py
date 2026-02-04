"""
Configuration constants for the visualization backend.
"""
from pathlib import Path

# Base paths (relative to this file's location)
BASE_DIR = Path(__file__).parent.parent.parent  # LLMsKnow root
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"

# Supported models with their configurations
# Note: LLaMA-3-8B has hidden_size=4096, not 8192 (8192 is for larger variants)
SUPPORTED_MODELS = {
    "mistralai/Mistral-7B-Instruct-v0.2": {
        "friendly_name": "mistral-7b-instruct",
        "display_name": "Mistral 7B Instruct v0.2",
        "num_layers": 32,
        "num_heads": 32,
        "hidden_size": 4096,
    },
    "mistralai/Mistral-7B-v0.3": {
        "friendly_name": "mistral-7b",
        "display_name": "Mistral 7B v0.3",
        "num_layers": 32,
        "num_heads": 32,
        "hidden_size": 4096,
    },
    "meta-llama/Meta-Llama-3-8B": {
        "friendly_name": "llama-3-8b",
        "display_name": "LLaMA 3 8B",
        "num_layers": 32,
        "num_heads": 32,
        "hidden_size": 4096,
    },
    "meta-llama/Meta-Llama-3-8B-Instruct": {
        "friendly_name": "llama-3-8b-instruct",
        "display_name": "LLaMA 3 8B Instruct",
        "num_layers": 32,
        "num_heads": 32,
        "hidden_size": 4096,
    },
}

# Dataset configurations
# IMPORTANT: "output_id" is the name used in output filenames (e.g., "movies" -> "mistral-7b-instruct-answers-movies.csv")
# This is different from "filename" which is the raw data CSV name
DATASET_CONFIG = {
    "movie_qa_train": {
        "filename": "movie_qa_train.csv",
        "output_id": "movies",
        "display_name": "Movie QA (Train)",
        "question_col": "Question",
        "answer_col": "Answer",
        "category": "factual",
    },
    "movie_qa_test": {
        "filename": "movie_qa_test.csv",
        "output_id": "movies_test",
        "display_name": "Movie QA (Test)",
        "question_col": "Question",
        "answer_col": "Answer",
        "category": "factual",
    },
    "answerable_math": {
        "filename": "AnswerableMath.csv",
        "output_id": "math",
        "display_name": "Answerable Math",
        "question_col": "question",
        "answer_col": "answer",
        "category": "math",
    },
    "answerable_math_test": {
        "filename": "AnswerableMath_test.csv",
        "output_id": "math_test",
        "display_name": "Answerable Math (Test)",
        "question_col": "question",
        "answer_col": "answer",
        "category": "math",
    },
    "mnli_train": {
        "filename": "mnli_train.csv",
        "output_id": "mnli",
        "display_name": "MNLI (Train)",
        "question_col": "Question",
        "answer_col": "Answer",
        "category": "nli",
    },
    "mnli_validation": {
        "filename": "mnli_validation.csv",
        "output_id": "mnli_validation",
        "display_name": "MNLI (Validation)",
        "question_col": "Question",
        "answer_col": "Answer",
        "category": "nli",
    },
    "winogrande_train": {
        "filename": "winogrande_train.csv",
        "output_id": "winogrande",
        "display_name": "Winogrande (Train)",
        "question_col": "sentence",
        "answer_col": "answer",
        "category": "commonsense",
    },
    "winogrande_test": {
        "filename": "winogrande_test.csv",
        "output_id": "winogrande_test",
        "display_name": "Winogrande (Test)",
        "question_col": "sentence",
        "answer_col": "answer",
        "category": "commonsense",
    },
    "winobias_dev": {
        "filename": "winobias_dev.csv",
        "output_id": "winobias",
        "display_name": "Winobias (Dev)",
        "question_col": "sentence",
        "answer_col": "answer",
        "category": "bias",
    },
    "winobias_test": {
        "filename": "winobias_test.csv",
        "output_id": "winobias_test",
        "display_name": "Winobias (Test)",
        "question_col": "sentence",
        "answer_col": "answer",
        "category": "bias",
    },
    "nq_wc": {
        "filename": "nq_wc_dataset.csv",
        "output_id": "natural_questions_with_context",
        "display_name": "Natural Questions (with context)",
        "question_col": "question",
        "answer_col": "answer",
        "category": "factual",
    },
}

# API settings
API_HOST = "0.0.0.0"
API_PORT = 8000
CORS_ORIGINS = [
    "http://localhost:5173", "http://127.0.0.1:5173",
    "http://localhost:5174", "http://127.0.0.1:5174",
    "http://localhost:5175", "http://127.0.0.1:5175",
    "http://localhost:5176", "http://127.0.0.1:5176",
]

# Pagination defaults
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100
