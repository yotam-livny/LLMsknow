"""
Dataset-specific correctness computation functions.
Adapted from src/compute_correctness.py for the visualization backend.
"""

import unicodedata
from typing import Optional
from utils.logging import get_logger

logger = get_logger(__name__)


def normalize_text(text: str) -> str:
    """Normalize text by removing accents and converting to lowercase."""
    # Normalize unicode to decomposed form (separates base char from accents)
    normalized = unicodedata.normalize('NFD', text)
    # Remove accent marks (combining diacritical marks)
    without_accents = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')
    return without_accents.lower().strip()


def compute_correctness_movies(model_answer: str, expected_answer: str) -> bool:
    """Movies dataset: substring match with accent normalization."""
    return normalize_text(expected_answer) in normalize_text(model_answer)


def compute_correctness_triviaqa(model_answer: str, expected_answer: str) -> bool:
    """TriviaQA: expected_answer can be a list of acceptable answers (as string repr)."""
    try:
        if expected_answer.startswith('['):
            labels = eval(expected_answer)
        else:
            labels = [expected_answer]
        
        normalized_answer = normalize_text(model_answer)
        for ans in labels:
            if normalize_text(ans) in normalized_answer:
                return True
        return False
    except Exception as e:
        logger.warning(f"Error parsing triviaqa answer: {e}")
        return normalize_text(expected_answer) in normalize_text(model_answer)


def compute_correctness_hotpotqa(model_answer: str, expected_answer: str) -> bool:
    """HotpotQA: substring match with normalization."""
    return normalize_text(expected_answer) in normalize_text(model_answer)


def compute_correctness_math(model_answer: str, expected_answer: str) -> bool:
    """Math: check if the numerical answer appears."""
    try:
        label = str(expected_answer).strip()
        return label in model_answer.lower() or str(int(float(label))) in model_answer.lower()
    except ValueError:
        return expected_answer.lower() in model_answer.lower()


def compute_correctness_nli(model_answer: str, expected_answer: str) -> bool:
    """NLI/MNLI: find which label appears first."""
    labels_dict = {
        'neutral': ['neutrality', 'neutral', 'neutality'],
        'entailment': ['entailment', 'entail'],
        'contradiction': ['contradiction', 'contradict']
    }
    
    correct_answer = expected_answer.lower().strip()
    if correct_answer not in labels_dict:
        return False
    
    first_label = None
    min_idx = len(model_answer)
    
    for label_name, label_variants in labels_dict.items():
        for label_str in label_variants:
            idx = model_answer.lower().find(label_str)
            if idx != -1 and idx < min_idx:
                first_label = label_name
                min_idx = idx
    
    return first_label == correct_answer


def compute_correctness_winobias(model_answer: str, expected_answer: str, wrong_answer: Optional[str] = None) -> bool:
    """Winobias: check which answer appears first."""
    if not wrong_answer:
        return expected_answer.lower() in model_answer.lower()
    
    correct_idx = model_answer.lower().find(expected_answer.lower())
    wrong_idx = model_answer.lower().find(wrong_answer.lower())
    
    if correct_idx == -1 and wrong_idx == -1:
        return False
    elif correct_idx != -1 and wrong_idx == -1:
        return True
    elif correct_idx == -1 and wrong_idx != -1:
        return False
    else:
        return correct_idx < wrong_idx


def compute_correctness_winogrande(model_answer: str, expected_answer: str, wrong_answer: Optional[str] = None) -> bool:
    """Winogrande: similar to winobias."""
    return compute_correctness_winobias(model_answer, expected_answer, wrong_answer)


# Dataset name to function mapping
CORRECTNESS_FUNCTIONS = {
    'movies': compute_correctness_movies,
    'movie_qa': compute_correctness_movies,
    'movie_qa_train': compute_correctness_movies,
    'movie_qa_test': compute_correctness_movies,
    'triviaqa': compute_correctness_triviaqa,
    'hotpotqa': compute_correctness_hotpotqa,
    'hotpotqa_with_context': compute_correctness_hotpotqa,
    'math': compute_correctness_math,
    'AnswerableMath': compute_correctness_math,
    'mnli': compute_correctness_nli,
    'mnli_train': compute_correctness_nli,
    'mnli_validation': compute_correctness_nli,
    'winobias': compute_correctness_winobias,
    'winobias_dev': compute_correctness_winobias,
    'winobias_test': compute_correctness_winobias,
    'winogrande': compute_correctness_winogrande,
    'winogrande_train': compute_correctness_winogrande,
    'winogrande_test': compute_correctness_winogrande,
}


def compute_correctness(
    dataset_id: str,
    model_answer: str,
    expected_answer: str,
    wrong_answer: Optional[str] = None
) -> Optional[bool]:
    """
    Compute correctness using dataset-specific logic.
    
    Args:
        dataset_id: Dataset identifier
        model_answer: The model's generated answer
        expected_answer: The correct/expected answer
        wrong_answer: For some datasets (winobias/winogrande), the incorrect answer
        
    Returns:
        True if correct, False if incorrect, None if cannot determine
    """
    if not expected_answer:
        return None
    
    # Find the appropriate function
    func = None
    for key, fn in CORRECTNESS_FUNCTIONS.items():
        if key in dataset_id.lower():
            func = fn
            break
    
    if func is None:
        # Default: simple substring match
        logger.warning(f"No specific correctness function for dataset {dataset_id}, using default")
        return expected_answer.lower().strip() in model_answer.lower()
    
    try:
        # Check if function needs wrong_answer
        if func in [compute_correctness_winobias, compute_correctness_winogrande]:
            return func(model_answer, expected_answer, wrong_answer)
        else:
            return func(model_answer, expected_answer)
    except Exception as e:
        logger.error(f"Error computing correctness: {e}")
        return None
