"""
Load and run trained probe classifiers.

Probes are logistic regression classifiers trained on layer representations
to predict whether the model's answer is correct.
"""
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

from utils.logging import get_logger
from config import CHECKPOINTS_DIR, SUPPORTED_MODELS, DATASET_CONFIG

logger = get_logger(__name__)


class ProbeRunner:
    """
    Load and run trained probe classifiers for correctness prediction.
    """
    
    def __init__(self, model_id: str, dataset_id: str):
        """
        Initialize the probe runner.
        
        Args:
            model_id: HuggingFace model ID
            dataset_id: Dataset identifier (e.g., "movie_qa_train")
        """
        self.model_id = model_id
        self.dataset_id = dataset_id
        self.model_friendly = SUPPORTED_MODELS[model_id]["friendly_name"]
        self.output_id = DATASET_CONFIG[dataset_id]["output_id"]
        
        # Cache for loaded probes
        self._probes: Dict[Tuple[int, str], Any] = {}
        
        logger.info(f"ProbeRunner initialized for {self.model_friendly}/{self.output_id}")
    
    def get_probe_path(self, layer: int, token: str) -> Path:
        """
        Get the path to a probe checkpoint file.
        
        Args:
            layer: Layer index
            token: Token type (e.g., "exact_answer_last_token")
            
        Returns:
            Path to the probe file
        """
        filename = f"clf_{self.model_friendly}_{self.output_id}_layer-{layer}_token-{token}.pkl"
        return CHECKPOINTS_DIR / filename
    
    def probe_exists(self, layer: int, token: str) -> bool:
        """Check if a probe exists for the given layer and token."""
        return self.get_probe_path(layer, token).exists()
    
    def list_available_probes(self) -> List[Dict[str, Any]]:
        """
        List all available probes for this model/dataset combination.
        
        Returns:
            List of dicts with layer, token, and path information
        """
        pattern = f"clf_{self.model_friendly}_{self.output_id}_layer-*_token-*.pkl"
        probes = []
        
        for probe_file in CHECKPOINTS_DIR.glob(pattern):
            # Parse filename: clf_{model}_{dataset}_layer-{N}_token-{type}.pkl
            parts = probe_file.stem.split("_")
            layer_part = [p for p in parts if p.startswith("layer-")]
            token_part = [p for p in parts if p.startswith("token-")]
            
            if layer_part and token_part:
                layer = int(layer_part[0].replace("layer-", ""))
                token = token_part[0].replace("token-", "")
                probes.append({
                    "layer": layer,
                    "token": token,
                    "path": str(probe_file)
                })
        
        logger.debug(f"Found {len(probes)} probes for {self.model_friendly}/{self.output_id}")
        return probes
    
    def load_probe(self, layer: int, token: str) -> Any:
        """
        Load a probe classifier from checkpoint.
        
        Args:
            layer: Layer index
            token: Token type
            
        Returns:
            Loaded scikit-learn classifier
        """
        cache_key = (layer, token)
        
        # Check cache first
        if cache_key in self._probes:
            logger.debug(f"Using cached probe for layer {layer}, token {token}")
            return self._probes[cache_key]
        
        probe_path = self.get_probe_path(layer, token)
        
        if not probe_path.exists():
            raise FileNotFoundError(
                f"Probe not found: {probe_path}. "
                f"Run training first with: python src/probe.py --model {self.model_id} "
                f"--dataset {self.output_id} --layer {layer} --token {token} --save_clf"
            )
        
        logger.info(f"Loading probe from {probe_path}")
        
        with open(probe_path, 'rb') as f:
            clf = pickle.load(f)
        
        self._probes[cache_key] = clf
        return clf
    
    def predict(
        self,
        representation: np.ndarray,
        layer: int,
        token: str
    ) -> Dict[str, Any]:
        """
        Predict correctness using a probe.
        
        Args:
            representation: Hidden state representation (hidden_size,)
            layer: Layer index the representation is from
            token: Token type the representation is from
            
        Returns:
            {
                "prediction": 0 or 1 (0 = correct, 1 = incorrect),
                "confidence": float (probability of predicted class),
                "probabilities": [p_correct, p_incorrect]
            }
        """
        clf = self.load_probe(layer, token)
        
        # Ensure correct shape
        if representation.ndim == 1:
            representation = representation.reshape(1, -1)
        
        prediction = int(clf.predict(representation)[0])
        probabilities = clf.predict_proba(representation)[0].tolist()
        confidence = float(max(probabilities))
        
        logger.debug(f"Probe prediction: {prediction} (confidence: {confidence:.3f})")
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probabilities,
            "layer": layer,
            "token": token
        }
    
    def predict_all_layers(
        self,
        layer_representations: Dict[int, np.ndarray],
        token: str,
        token_idx: int
    ) -> Dict[int, Dict[str, Any]]:
        """
        Run predictions across all layers where probes are available.
        
        Args:
            layer_representations: {layer_idx: (seq_len, hidden_size)}
            token: Token type for probe selection
            token_idx: Index of the token to use for prediction
            
        Returns:
            {layer_idx: prediction_result}
        """
        results = {}
        available_probes = self.list_available_probes()
        probe_layers = {p["layer"] for p in available_probes if p["token"] == token}
        
        for layer_idx, representation in layer_representations.items():
            if layer_idx not in probe_layers:
                continue
            
            # Get representation at specific token position
            if isinstance(representation, list):
                representation = np.array(representation)
            
            token_rep = representation[token_idx]
            
            try:
                results[layer_idx] = self.predict(token_rep, layer_idx, token)
            except Exception as e:
                logger.warning(f"Failed to predict at layer {layer_idx}: {e}")
        
        return results


def get_probe_info(model_id: str, dataset_id: str) -> Dict[str, Any]:
    """
    Get information about available probes for a model/dataset combination.
    
    Returns:
        {
            "available": bool,
            "probes": [{"layer": int, "token": str}, ...],
            "total_count": int
        }
    """
    try:
        runner = ProbeRunner(model_id, dataset_id)
        probes = runner.list_available_probes()
        
        return {
            "available": len(probes) > 0,
            "probes": probes,
            "total_count": len(probes)
        }
    except KeyError:
        return {
            "available": False,
            "probes": [],
            "total_count": 0
        }
