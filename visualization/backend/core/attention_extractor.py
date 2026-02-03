"""
Extract and process attention patterns from model forward pass.

NOTE: This uses model's native `output_attentions=True` parameter rather than
baukit's TraceDict (used in layer_extractor.py for layer representation extraction).
This is intentional: we need raw attention weights (softmax outputs), not the
projected outputs traced for probing.

The two approaches are complementary:
- layer_extractor.py (TraceDict): Extracts layer outputs/hidden states for probing
- AttentionExtractor: Extracts raw attention patterns for visualization
"""
import torch
import numpy as np
from typing import Dict, Any, List, Optional

from utils.logging import get_logger
from config import SUPPORTED_MODELS

logger = get_logger(__name__)


class AttentionExtractor:
    """
    Extract attention patterns from model forward pass.
    Uses output_attentions=True to get raw attention weights.
    """
    
    def __init__(self, model, model_id: str):
        """
        Initialize the attention extractor.
        
        Args:
            model: The loaded HuggingFace model
            model_id: Model identifier for configuration lookup
        """
        self.model = model
        self.model_id = model_id
        self.num_layers = SUPPORTED_MODELS[model_id]["num_layers"]
        self.num_heads = SUPPORTED_MODELS[model_id]["num_heads"]
        
        logger.info(f"AttentionExtractor initialized for {model_id} "
                   f"({self.num_layers} layers, {self.num_heads} heads)")
    
    def extract_attention_patterns(
        self,
        input_ids: torch.Tensor,
        layers: Optional[List[int]] = None,
        heads: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Extract attention patterns from specified layers and heads.
        
        Args:
            input_ids: Input token IDs (1, seq_len)
            layers: Optional list of layer indices to extract (default: all)
            heads: Optional list of head indices to extract (default: all)
            
        Returns:
            {
                "patterns": {layer_idx: {head_idx: [[float]]}},  # seq_len x seq_len
                "statistics": {layer_idx: {head_idx: stats_dict}},
                "seq_len": int
            }
        """
        logger.debug(f"Extracting attention patterns (layers={layers}, heads={heads})")
        
        self.model.eval()
        
        # Ensure batch dimension
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids,
                output_attentions=True,
                return_dict=True
            )
        
        # outputs.attentions is tuple of (batch, num_heads, seq_len, seq_len) per layer
        attention_patterns = {}
        attention_stats = {}
        
        for layer_idx, layer_attention in enumerate(outputs.attentions):
            if layers is not None and layer_idx not in layers:
                continue
            
            attention_patterns[layer_idx] = {}
            attention_stats[layer_idx] = {}
            
            for head_idx in range(layer_attention.shape[1]):
                if heads is not None and head_idx not in heads:
                    continue
                
                # Extract attention matrix (seq_len, seq_len)
                attn = layer_attention[0, head_idx].cpu().numpy()
                
                attention_patterns[layer_idx][head_idx] = attn.tolist()
                attention_stats[layer_idx][head_idx] = self._compute_head_statistics(attn)
        
        seq_len = input_ids.shape[1]
        
        logger.debug(f"Extracted attention: {len(attention_patterns)} layers, seq_len={seq_len}")
        
        return {
            "patterns": attention_patterns,
            "statistics": attention_stats,
            "seq_len": seq_len
        }
    
    def extract_single_head(
        self,
        input_ids: torch.Tensor,
        layer_idx: int,
        head_idx: int
    ) -> np.ndarray:
        """
        Extract attention pattern for a single head.
        
        Args:
            input_ids: Input token IDs
            layer_idx: Layer index
            head_idx: Head index
            
        Returns:
            numpy array of shape (seq_len, seq_len)
        """
        result = self.extract_attention_patterns(
            input_ids,
            layers=[layer_idx],
            heads=[head_idx]
        )
        return np.array(result["patterns"][layer_idx][head_idx])
    
    def get_attention_summary(
        self,
        input_ids: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Get a summary of attention patterns across all layers and heads.
        Useful for overview visualization without full pattern data.
        
        Args:
            input_ids: Input token IDs
            
        Returns:
            {
                "layer_stats": {layer_idx: {"entropy": float, "sparsity": float}},
                "head_importance": {layer_idx: {head_idx: float}},
                "seq_len": int
            }
        """
        result = self.extract_attention_patterns(input_ids)
        
        layer_stats = {}
        head_importance = {}
        
        for layer_idx, layer_patterns in result["patterns"].items():
            layer_entropies = []
            head_importance[layer_idx] = {}
            
            for head_idx, pattern in layer_patterns.items():
                attn = np.array(pattern)
                stats = self._compute_head_statistics(attn)
                layer_entropies.append(stats["entropy"])
                
                # Head importance based on attention concentration
                head_importance[layer_idx][head_idx] = stats["max_attention"]
            
            layer_stats[layer_idx] = {
                "mean_entropy": float(np.mean(layer_entropies)),
                "std_entropy": float(np.std(layer_entropies))
            }
        
        return {
            "layer_stats": layer_stats,
            "head_importance": head_importance,
            "seq_len": result["seq_len"]
        }
    
    @staticmethod
    def _compute_head_statistics(attention_matrix: np.ndarray) -> Dict[str, float]:
        """
        Compute statistics for an attention head.
        
        Args:
            attention_matrix: (seq_len, seq_len) attention weights
            
        Returns:
            Dictionary of statistics
        """
        # Compute entropy (measure of attention distribution)
        # Higher entropy = more distributed attention
        epsilon = 1e-10
        entropy_per_query = -np.sum(
            attention_matrix * np.log(attention_matrix + epsilon),
            axis=-1
        )
        mean_entropy = float(np.mean(entropy_per_query))
        
        # Compute sparsity (fraction of near-zero attention)
        threshold = 0.01
        sparsity = float(np.mean(attention_matrix < threshold))
        
        # Maximum attention weight
        max_attention = float(np.max(attention_matrix))
        
        # Diagonal attention (self-attention strength)
        diagonal = np.diag(attention_matrix)
        mean_self_attention = float(np.mean(diagonal))
        
        return {
            "entropy": mean_entropy,
            "sparsity": sparsity,
            "max_attention": max_attention,
            "mean_self_attention": mean_self_attention
        }


def aggregate_attention_across_heads(
    attention_patterns: Dict[int, Dict[int, List]],
    aggregation: str = "mean"
) -> Dict[int, List]:
    """
    Aggregate attention patterns across heads within each layer.
    
    Args:
        attention_patterns: {layer_idx: {head_idx: pattern}}
        aggregation: "mean" or "max"
        
    Returns:
        {layer_idx: aggregated_pattern}
    """
    aggregated = {}
    
    for layer_idx, heads in attention_patterns.items():
        patterns = [np.array(pattern) for pattern in heads.values()]
        stacked = np.stack(patterns, axis=0)
        
        if aggregation == "mean":
            aggregated[layer_idx] = np.mean(stacked, axis=0).tolist()
        elif aggregation == "max":
            aggregated[layer_idx] = np.max(stacked, axis=0).tolist()
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
    
    return aggregated
