"""
Singleton model manager for loading and managing LLM models.
Ensures only one model is loaded on GPU at a time.
"""
import sys
import torch
from typing import Optional, Tuple, Any
from pathlib import Path

# Add src to path to import probing_utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from utils.logging import get_logger
from config import SUPPORTED_MODELS

logger = get_logger(__name__)


class ModelManager:
    """
    Singleton manager for LLM models.
    Only one model is loaded at a time to manage GPU memory.
    """
    _instance = None
    _model = None
    _tokenizer = None
    _current_model_id: Optional[str] = None
    _device: str = "cpu"
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._detect_device()
        return cls._instance
    
    @classmethod
    def _detect_device(cls):
        """Detect the best available device."""
        if torch.cuda.is_available():
            cls._device = "cuda"
            logger.info("CUDA detected - will use GPU")
        elif torch.backends.mps.is_available():
            cls._device = "mps"
            logger.info("MPS detected - will use Apple Silicon")
        else:
            cls._device = "cpu"
            logger.info("No GPU detected - will use CPU")
    
    @classmethod
    def get_device(cls) -> str:
        """Get the current device."""
        return cls._device
    
    @classmethod
    def get_current_model_id(cls) -> Optional[str]:
        """Get the ID of the currently loaded model."""
        return cls._current_model_id
    
    @classmethod
    def is_model_loaded(cls, model_id: str) -> bool:
        """Check if a specific model is loaded."""
        return cls._current_model_id == model_id and cls._model is not None
    
    @classmethod
    def get_model_and_tokenizer(cls) -> Tuple[Any, Any]:
        """Get the currently loaded model and tokenizer."""
        if cls._model is None:
            raise RuntimeError("No model is currently loaded")
        return cls._model, cls._tokenizer
    
    @classmethod
    def load_model(cls, model_id: str, use_quantization: bool = False) -> Tuple[Any, Any]:
        """
        Load a model, unloading any previously loaded model first.
        
        Args:
            model_id: HuggingFace model ID (e.g., "mistralai/Mistral-7B-Instruct-v0.2")
            use_quantization: Whether to use 4-bit quantization
            
        Returns:
            Tuple of (model, tokenizer)
        """
        if model_id not in SUPPORTED_MODELS:
            raise ValueError(f"Model {model_id} is not supported. Supported: {list(SUPPORTED_MODELS.keys())}")
        
        # If same model is already loaded, return it
        if cls.is_model_loaded(model_id):
            logger.info(f"Model {model_id} is already loaded")
            return cls._model, cls._tokenizer
        
        # Unload existing model first
        if cls._model is not None:
            cls.unload_model()
        
        logger.info(f"Loading model {model_id}...")
        
        try:
            # Import here to avoid circular imports
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            logger.debug(f"Tokenizer loaded for {model_id}")
            
            # Prepare load arguments
            load_kwargs = {
                'torch_dtype': torch.bfloat16,
                'low_cpu_mem_usage': True,
            }
            
            # Handle quantization
            if use_quantization:
                try:
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                    load_kwargs['quantization_config'] = quantization_config
                    logger.info("Using 4-bit quantization")
                except ImportError:
                    logger.warning("bitsandbytes not available, loading without quantization")
            
            # Load model based on device
            if cls._device == "cuda":
                load_kwargs['device_map'] = 'auto'
                model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
                logger.info(f"Model loaded to CUDA with device_map: {model.hf_device_map}")
            else:
                model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
                if cls._device == "mps":
                    model = model.to('mps')
                    logger.info("Model moved to MPS")
                else:
                    logger.info("Model on CPU")
            
            model.eval()  # Set to evaluation mode
            
            cls._model = model
            cls._tokenizer = tokenizer
            cls._current_model_id = model_id
            
            logger.info(f"Model {model_id} loaded successfully")
            return cls._model, cls._tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}", exc_info=True)
            cls._model = None
            cls._tokenizer = None
            cls._current_model_id = None
            raise
    
    @classmethod
    def unload_model(cls):
        """Unload the current model and free memory."""
        if cls._model is not None:
            logger.info(f"Unloading model {cls._current_model_id}...")
            
            # Delete model and tokenizer
            del cls._model
            del cls._tokenizer
            
            cls._model = None
            cls._tokenizer = None
            cls._current_model_id = None
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("CUDA cache cleared")
            
            logger.info("Model unloaded successfully")
    
    @classmethod
    def tokenize(cls, text: str) -> torch.Tensor:
        """
        Tokenize input text using the loaded model's tokenizer.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            Tensor of input IDs
        """
        if cls._tokenizer is None:
            raise RuntimeError("No model/tokenizer is currently loaded")
        
        model_name = cls._current_model_id.lower()
        
        # Use chat template for instruct models
        if 'instruct' in model_name:
            messages = [{"role": "user", "content": text}]
            input_ids = cls._tokenizer.apply_chat_template(
                messages, 
                return_tensors="pt"
            ).to(cls._device)
        else:
            tokenized = cls._tokenizer(text, return_tensors='pt')
            input_ids = tokenized["input_ids"].to(cls._device)
        
        return input_ids
    
    @classmethod
    def generate(
        cls,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        do_sample: bool = False,
        temperature: float = 1.0,
        output_attentions: bool = False,
        output_hidden_states: bool = False
    ) -> dict:
        """
        Generate text from input IDs.
        
        Args:
            input_ids: Input token IDs
            max_new_tokens: Maximum number of new tokens to generate
            do_sample: Whether to use sampling
            temperature: Sampling temperature
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states
            
        Returns:
            Generation output dictionary
        """
        if cls._model is None:
            raise RuntimeError("No model is currently loaded")
        
        with torch.no_grad():
            output = cls._model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict_in_generate=True,
            )
        
        return output
    
    @classmethod
    def decode(cls, token_ids: torch.Tensor) -> str:
        """Decode token IDs to text."""
        if cls._tokenizer is None:
            raise RuntimeError("No tokenizer is currently loaded")
        return cls._tokenizer.decode(token_ids, skip_special_tokens=True)


# Convenience function to get the singleton
def get_model_manager() -> ModelManager:
    """Get the singleton ModelManager instance."""
    return ModelManager()
