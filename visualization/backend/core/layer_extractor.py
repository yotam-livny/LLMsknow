"""
Extract layer representations (hidden states) from model forward pass.

Uses baukit's TraceDict to capture internal layer activations, matching
the approach used in probing_utils.py for consistency with trained probes.
"""
import sys
import torch
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path

# Add src to path to import probing_utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from utils.logging import get_logger
from config import SUPPORTED_MODELS

logger = get_logger(__name__)


# Layer name patterns for different model architectures
LAYER_PATTERNS = {
    "mistral": {
        "mlp": "model.layers.{layer}.mlp",
        "mlp_down_proj": "model.layers.{layer}.mlp.down_proj",
        "attention": "model.layers.{layer}.self_attn.o_proj",
    },
    "llama": {
        "mlp": "model.layers.{layer}.mlp",
        "mlp_down_proj": "model.layers.{layer}.mlp.down_proj",
        "attention": "model.layers.{layer}.self_attn.o_proj",
    }
}


def get_model_architecture(model_id: str) -> str:
    """Determine model architecture from model ID."""
    model_lower = model_id.lower()
    if "mistral" in model_lower:
        return "mistral"
    elif "llama" in model_lower:
        return "llama"
    else:
        raise ValueError(f"Unknown model architecture for {model_id}")


def get_layer_names(model_id: str, probe_type: str = "mlp") -> List[str]:
    """
    Get the layer names to trace for a given model and probe type.
    
    Args:
        model_id: HuggingFace model ID
        probe_type: Type of probe ("mlp", "mlp_down_proj", "attention")
        
    Returns:
        List of layer module names to trace
    """
    if model_id not in SUPPORTED_MODELS:
        raise ValueError(f"Model {model_id} not supported")
    
    num_layers = SUPPORTED_MODELS[model_id]["num_layers"]
    arch = get_model_architecture(model_id)
    pattern = LAYER_PATTERNS[arch].get(probe_type, LAYER_PATTERNS[arch]["mlp"])
    
    return [pattern.format(layer=i) for i in range(num_layers)]


class LayerExtractor:
    """
    Extract hidden state representations from model layers.
    
    Uses baukit's TraceDict to intercept layer outputs during forward pass,
    matching the extraction method used in probing_utils.py.
    """
    
    def __init__(self, model, model_id: str):
        """
        Initialize the extractor.
        
        Args:
            model: The loaded HuggingFace model
            model_id: Model identifier for configuration lookup
        """
        self.model = model
        self.model_id = model_id
        self.num_layers = SUPPORTED_MODELS[model_id]["num_layers"]
        self.hidden_size = SUPPORTED_MODELS[model_id]["hidden_size"]
        
        logger.info(f"LayerExtractor initialized for {model_id} "
                   f"({self.num_layers} layers, hidden_size={self.hidden_size})")
    
    def extract_all_layers(
        self,
        input_ids: torch.Tensor,
        probe_type: str = "mlp"
    ) -> Dict[str, Any]:
        """
        Extract representations from all layers.
        
        Args:
            input_ids: Input token IDs (1, seq_len)
            probe_type: Type of layer to extract ("mlp", "attention")
            
        Returns:
            {
                "layers": {layer_idx: (seq_len, hidden_size)},
                "tokens": [token_str, ...],
                "input_ids": [id, ...],
                "seq_len": int
            }
        """
        from baukit import TraceDict
        
        logger.debug(f"Extracting {probe_type} representations from all layers")
        
        layer_names = get_layer_names(self.model_id, probe_type)
        
        # Ensure input has batch dimension
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        with torch.no_grad():
            with TraceDict(self.model, layer_names, retain_input=True, clone=True) as ret:
                _ = self.model(input_ids, output_hidden_states=True)
        
        # Extract outputs from each layer
        layer_outputs = {}
        for layer_idx, layer_name in enumerate(layer_names):
            # Output shape: (batch, seq_len, hidden_size)
            output = ret[layer_name].output.squeeze(0).cpu()  # (seq_len, hidden_size)
            layer_outputs[layer_idx] = output.numpy().tolist()
        
        seq_len = input_ids.shape[1]
        
        logger.debug(f"Extracted representations: {len(layer_outputs)} layers, seq_len={seq_len}")
        
        return {
            "layers": layer_outputs,
            "input_ids": input_ids.squeeze(0).cpu().tolist(),
            "seq_len": seq_len
        }
    
    def extract_single_layer(
        self,
        input_ids: torch.Tensor,
        layer_idx: int,
        probe_type: str = "mlp"
    ) -> np.ndarray:
        """
        Extract representations from a single layer.
        
        Args:
            input_ids: Input token IDs
            layer_idx: Layer index to extract
            probe_type: Type of layer to extract
            
        Returns:
            numpy array of shape (seq_len, hidden_size)
        """
        from baukit import TraceDict
        
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError(f"Layer index {layer_idx} out of range [0, {self.num_layers})")
        
        layer_names = get_layer_names(self.model_id, probe_type)
        target_layer = layer_names[layer_idx]
        
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        with torch.no_grad():
            with TraceDict(self.model, [target_layer], retain_input=True, clone=True) as ret:
                _ = self.model(input_ids)
        
        output = ret[target_layer].output.squeeze(0).cpu().numpy()
        return output
    
    def extract_token_at_layer(
        self,
        input_ids: torch.Tensor,
        layer_idx: int,
        token_idx: int,
        probe_type: str = "mlp"
    ) -> np.ndarray:
        """
        Extract representation for a specific token at a specific layer.
        
        Args:
            input_ids: Input token IDs
            layer_idx: Layer index
            token_idx: Token position index
            probe_type: Type of layer to extract
            
        Returns:
            numpy array of shape (hidden_size,)
        """
        layer_output = self.extract_single_layer(input_ids, layer_idx, probe_type)
        
        if token_idx < 0 or token_idx >= layer_output.shape[0]:
            raise ValueError(f"Token index {token_idx} out of range [0, {layer_output.shape[0]})")
        
        return layer_output[token_idx]
    
    def compute_layer_statistics(
        self,
        layer_outputs: Dict[int, Any]
    ) -> Dict[int, Dict[str, float]]:
        """
        Compute statistics for each layer's representations.
        
        Args:
            layer_outputs: Dictionary mapping layer index to representations
            
        Returns:
            Dictionary mapping layer index to statistics
        """
        stats = {}
        for layer_idx, output in layer_outputs.items():
            arr = np.array(output)
            stats[layer_idx] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "norm": float(np.linalg.norm(arr.mean(axis=0)))  # Norm of mean representation
            }
        return stats
