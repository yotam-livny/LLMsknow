# LLMsKnow Visualization Tool - Code Design Document

> **Purpose**: Technical blueprint for implementation. See `VISUALIZATION_DESIGN.md` for requirements and UX design.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Backend Design](#2-backend-design)
3. [Frontend Design](#3-frontend-design)
4. [Data Flow](#4-data-flow)
5. [Type Definitions](#5-type-definitions)
6. [API Contracts](#6-api-contracts)
7. [Integration with Existing Codebase](#7-integration-with-existing-codebase)
8. [Error Handling Strategy](#8-error-handling-strategy)
9. [Caching Strategy](#9-caching-strategy)
10. [Implementation Checklist](#10-implementation-checklist)

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                FRONTEND                                      │
│                          (React + TypeScript + D3)                          │
│                                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │    Hooks     │  │  Components  │  │     API      │  │    Utils     │    │
│  │              │  │              │  │    Client    │  │              │    │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────────────┘    │
│         │                 │                 │                               │
│         └─────────────────┴─────────────────┘                               │
│                           │                                                 │
│                           ▼                                                 │
│                    ┌──────────────┐                                         │
│                    │  State Store │  (React Context / Zustand)              │
│                    └──────┬───────┘                                         │
└────────────────────────────┼────────────────────────────────────────────────┘
                             │ HTTP/REST
                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                                BACKEND                                       │
│                          (FastAPI + Python)                                 │
│                                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   app.py     │  │   Managers   │  │  Extractors  │  │   Scanners   │    │
│  │   (routes)   │  │              │  │              │  │              │    │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
│         │                 │                 │                 │             │
│         └─────────────────┴─────────────────┴─────────────────┘             │
│                                    │                                        │
│                                    ▼                                        │
│                    ┌───────────────────────────┐                            │
│                    │  Existing Codebase (src/) │                            │
│                    │  probing_utils.py         │                            │
│                    │  probe.py                 │                            │
│                    │  compute_correctness.py   │                            │
│                    └───────────────────────────┘                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| Frontend | React 18 | UI framework |
| Frontend | TypeScript | Type safety |
| Frontend | D3.js | Data visualizations |
| Frontend | Tailwind CSS | Styling |
| Frontend | Zustand | State management |
| Backend | FastAPI | API framework |
| Backend | Pydantic | Request/response validation |
| Backend | PyTorch | Model inference |
| Backend | Pandas | Dataset handling |

---

## 2. Backend Design

### 2.1 Module Structure

```
visualization/
├── backend/
│   ├── __init__.py
│   ├── app.py                    # FastAPI app, routes, middleware
│   ├── config.py                 # Configuration and constants
│   │
│   ├── managers/
│   │   ├── __init__.py
│   │   ├── model_manager.py      # Model loading, inference
│   │   ├── dataset_manager.py    # Dataset loading, pagination
│   │   └── session_manager.py    # Session state (current inference results)
│   │
│   ├── extractors/
│   │   ├── __init__.py
│   │   ├── layer_extractor.py    # Layer representation extraction
│   │   ├── attention_extractor.py# Attention pattern extraction
│   │   └── dimension_reducer.py  # PCA for dimension flow
│   │
│   ├── scanners/
│   │   ├── __init__.py
│   │   └── availability_scanner.py# Scan for available model/dataset combos
│   │
│   ├── runners/
│   │   ├── __init__.py
│   │   └── probe_runner.py       # Load and run probe classifiers
│   │
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── requests.py           # Pydantic request models
│   │   └── responses.py          # Pydantic response models
│   │
│   └── utils/
│       ├── __init__.py
│       └── correctness.py        # Wrapper for compute_correctness
```

### 2.2 Module Specifications

#### `config.py`
```python
"""Configuration constants and paths."""
from pathlib import Path

# Paths (relative to project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
SRC_DIR = PROJECT_ROOT / "src"

# Model configurations
SUPPORTED_MODELS = {
    "mistralai/Mistral-7B-Instruct-v0.2": {
        "friendly_name": "mistral-7b-instruct",
        "num_layers": 32,
        "num_heads": 32,
        "hidden_size": 4096,
    },
    "mistralai/Mistral-7B-v0.3": {
        "friendly_name": "mistral-7b",
        "num_layers": 32,
        "num_heads": 32,
        "hidden_size": 4096,
    },
    "meta-llama/Meta-Llama-3-8B": {
        "friendly_name": "llama-3-8b",
        "num_layers": 32,
        "num_heads": 32,
        "hidden_size": 8192,  # NOTE: LLaMA-3-8B has 8192 hidden size, not 4096
    },
    "meta-llama/Meta-Llama-3-8B-Instruct": {
        "friendly_name": "llama-3-8b-instruct",
        "num_layers": 32,
        "num_heads": 32,
        "hidden_size": 8192,  # NOTE: LLaMA-3-8B has 8192 hidden size, not 4096
    },
}

# Dataset configurations
# IMPORTANT: "output_id" is the name used in output filenames (e.g., "movies" -> "mistral-7b-instruct-answers-movies.csv")
# This is different from "filename" which is the raw data CSV name
DATASET_CONFIG = {
    "movie_qa_train": {
        "filename": "movie_qa_train.csv",
        "output_id": "movies",  # Used in output file naming by existing pipeline
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
        "question_col": "question",  # NOTE: lowercase in this dataset
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
        "output_id": "mnli_test",
        "display_name": "MNLI (Validation)",
        "question_col": "Question",
        "answer_col": "Answer",
        "category": "nli",
    },
    "winogrande_train": {
        "filename": "winogrande_train.csv",
        "output_id": "winogrande",
        "display_name": "Winogrande (Train)",
        "question_col": "Question",
        "answer_col": "Answer",
        "wrong_answer_col": "Wrong_Answer",  # Additional column for this dataset
        "category": "commonsense",
    },
    "winogrande_test": {
        "filename": "winogrande_test.csv",
        "output_id": "winogrande_test",
        "display_name": "Winogrande (Test)",
        "question_col": "Question",
        "answer_col": "Answer",
        "wrong_answer_col": "Wrong_Answer",
        "category": "commonsense",
    },
    "winobias_dev": {
        "filename": "winobias_dev.csv",
        "output_id": "winobias",
        "display_name": "WinoBias (Dev)",
        "question_col": "sentence",  # NOTE: Different schema
        "answer_col": "answer",
        "category": "bias",
    },
    "winobias_test": {
        "filename": "winobias_test.csv",
        "output_id": "winobias_test",
        "display_name": "WinoBias (Test)",
        "question_col": "sentence",
        "answer_col": "answer",
        "category": "bias",
    },
    "nq_wc_dataset": {
        "filename": "nq_wc_dataset.csv",
        "output_id": "natural_questions_with_context",
        "display_name": "Natural Questions (with Context)",
        "question_col": "Question",
        "answer_col": "Answer",
        "category": "qa",
    },
}

# Server settings
API_HOST = "0.0.0.0"
API_PORT = 8000
CORS_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]
```

#### `managers/model_manager.py`
```python
"""
Model loading and inference management.
Singleton pattern - only one model loaded at a time.
"""
from typing import Optional, Dict, Any, Tuple
import torch

class ModelManager:
    _instance: Optional['ModelManager'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.model = None
        self.tokenizer = None
        self.current_model_id: Optional[str] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def load_model(self, model_id: str) -> Dict[str, Any]:
        """
        Load model into memory. Unloads previous model if different.
        Returns: {"status": "loaded", "model_id": str, "device": str}
        Raises: ModelLoadError on failure
        """
        pass
    
    def is_loaded(self, model_id: str) -> bool:
        """Check if specific model is currently loaded."""
        pass
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        extract_hidden_states: bool = True,
        extract_attentions: bool = True
    ) -> Tuple[str, Optional[Dict], Optional[Dict]]:
        """
        Generate answer and optionally extract internal representations.
        
        Returns:
            - generated_text: str
            - hidden_states: Dict[layer_idx, Tensor] or None
            - attentions: Dict[layer_idx, Tensor] or None
        """
        pass
    
    def unload(self):
        """Free GPU memory by unloading current model."""
        pass
    
    def get_model_config(self) -> Dict[str, Any]:
        """Return current model's configuration (layers, heads, etc.)."""
        pass


# Singleton accessor
def get_model_manager() -> ModelManager:
    return ModelManager()
```

#### `managers/dataset_manager.py`
```python
"""
Dataset loading and pagination.
Uses LRU cache for loaded datasets.
"""
from typing import List, Dict, Any, Optional
from functools import lru_cache
import pandas as pd

@lru_cache(maxsize=8)
def _load_dataset_cached(dataset_id: str) -> pd.DataFrame:
    """Internal cached loader."""
    pass

class DatasetManager:
    
    @staticmethod
    def get_available_datasets() -> List[Dict[str, Any]]:
        """
        List all available datasets with metadata.
        Returns: [{"id", "name", "total_samples", "category", "columns"}, ...]
        """
        pass
    
    @staticmethod
    def get_samples(
        dataset_id: str,
        page: int = 1,
        page_size: int = 20,
        search_query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get paginated samples with optional search.
        Returns: {
            "dataset_id", "page", "page_size", "total_pages",
            "total_samples", "samples": [{"idx", "question", "answer"}],
            "search_query"
        }
        """
        pass
    
    @staticmethod
    def get_sample(dataset_id: str, sample_idx: int) -> Dict[str, Any]:
        """
        Get single sample by index.
        Returns: {"dataset_id", "sample_idx", "question", "expected_answer", "metadata"}
        """
        pass
    
    @staticmethod
    def get_total_samples(dataset_id: str) -> int:
        """Get total sample count for a dataset."""
        pass
```

#### `managers/session_manager.py`
```python
"""
Manages current session state (inference results).
Single session - new inference replaces old data.
"""
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

@dataclass
class SessionState:
    """Current inference session state."""
    model_id: str
    question: str
    expected_answer: Optional[str]
    generated_answer: str
    is_correct: Optional[bool]
    
    # Extracted data
    hidden_states: Dict[int, Any] = field(default_factory=dict)  # layer_idx -> data
    attention_patterns: Dict[int, Dict[int, Any]] = field(default_factory=dict)  # layer -> head -> data
    probe_predictions: Dict[int, float] = field(default_factory=dict)  # layer_idx -> confidence
    
    # Tokens
    input_tokens: list = field(default_factory=list)
    output_tokens: list = field(default_factory=list)

class SessionManager:
    _instance: Optional['SessionManager'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.session: Optional[SessionState] = None
        return cls._instance
    
    def create_session(self, **kwargs) -> SessionState:
        """Create new session, replacing any existing one."""
        pass
    
    def get_session(self) -> Optional[SessionState]:
        """Get current session or None."""
        pass
    
    def clear_session(self):
        """Clear current session."""
        pass
    
    def has_active_session(self) -> bool:
        """Check if there's an active session."""
        pass


def get_session_manager() -> SessionManager:
    return SessionManager()
```

#### `extractors/layer_extractor.py`
```python
"""
Extract and process layer representations.
Integrates with existing probing_utils.py
"""
from typing import Dict, Any, List
import torch

class LayerExtractor:
    
    @staticmethod
    def extract_hidden_states(
        model,
        input_ids: torch.Tensor,
        layers: List[int] = None  # None = all layers
    ) -> Dict[int, torch.Tensor]:
        """
        Extract hidden states from specified layers.
        Returns: {layer_idx: Tensor[seq_len, hidden_size]}
        """
        pass
    
    @staticmethod
    def compute_layer_statistics(
        hidden_state: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute statistics for a layer's hidden state.
        Returns: {"mean", "std", "max", "min", "norm"}
        """
        pass
    
    @staticmethod
    def get_token_representation(
        hidden_states: Dict[int, torch.Tensor],
        token_idx: int,
        layer_idx: int
    ) -> torch.Tensor:
        """Get representation for specific token at specific layer."""
        pass
```

#### `extractors/attention_extractor.py`
```python
"""
Extract and process attention patterns.
"""
from typing import Dict, Any, List, Tuple
import torch
import numpy as np

class AttentionExtractor:
    
    @staticmethod
    def extract_attention_patterns(
        model,
        input_ids: torch.Tensor,
        layers: List[int] = None,
        heads: List[int] = None
    ) -> Dict[int, Dict[int, np.ndarray]]:
        """
        Extract attention patterns.
        Returns: {layer_idx: {head_idx: attention_matrix[seq, seq]}}
        """
        pass
    
    @staticmethod
    def compute_head_statistics(
        attention: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute statistics for attention head.
        Returns: {"avg", "max", "entropy", "sparsity"}
        """
        pass
    
    @staticmethod
    def compute_attention_flow(
        attention_patterns: Dict[int, Dict[int, np.ndarray]],
        threshold: float = 0.1
    ) -> List[Dict[str, Any]]:
        """
        Compute cross-layer attention flow (head-to-head connections).
        Returns: [{"source_layer", "target_layer", "head_connections": [...]}]
        """
        pass
    
    @staticmethod
    def get_top_attended_tokens(
        attention: np.ndarray,
        token_idx: int,
        token_texts: List[str],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get tokens that a specific token attends to most.
        Returns: [{"token_idx", "token_text", "weight"}, ...]
        """
        pass
```

#### `extractors/dimension_reducer.py`
```python
"""
PCA-based dimension reduction for dimension flow view.
"""
from typing import Dict, List, Tuple
import numpy as np
from sklearn.decomposition import PCA

class DimensionReducer:
    
    def __init__(self, n_components: int = 10):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
    
    def fit_transform_layers(
        self,
        hidden_states: Dict[int, np.ndarray]
    ) -> Dict[int, np.ndarray]:
        """
        Apply PCA across all layers.
        Returns: {layer_idx: reduced_representation[n_components]}
        """
        pass
    
    def get_explained_variance(self) -> List[float]:
        """Get explained variance ratio per component."""
        pass
    
    def get_component_correlations_with_correctness(
        self,
        reduced_states: Dict[int, np.ndarray],
        correctness_labels: np.ndarray
    ) -> List[float]:
        """Compute correlation of each PC with correctness."""
        pass
```

#### `scanners/availability_scanner.py`
```python
"""
Scan output/ and checkpoints/ for available combinations.
"""
from typing import List, Dict, Any
from pathlib import Path
import re

class AvailabilityScanner:
    
    @staticmethod
    def scan_all_combinations() -> List[Dict[str, Any]]:
        """
        Scan for all model/dataset combinations and their status.
        Returns: [{
            "model_id", "model_name", "dataset_id", "dataset_name",
            "has_answers", "has_input_output_ids", "has_probe",
            "probe_config", "samples_processed", "samples_total",
            "samples_coverage", "accuracy",
            "ready_for_visualization", "ready_for_probe_predictions"
        }, ...]
        """
        pass
    
    @staticmethod
    def get_combinations_for_model(model_id: str) -> List[Dict[str, Any]]:
        """Filter combinations by model."""
        pass
    
    @staticmethod
    def get_ready_combinations() -> List[Dict[str, Any]]:
        """Get only fully ready combinations (has probe)."""
        pass
    
    @staticmethod
    def _parse_probe_filename(filename: str) -> Dict[str, Any]:
        """Parse probe checkpoint filename for config."""
        pass
    
    @staticmethod
    def _count_processed_samples(answers_file: Path) -> Tuple[int, float]:
        """Count samples and accuracy from answers CSV."""
        pass
```

#### `runners/probe_runner.py`
```python
"""
Load and run probe classifiers.
Integrates with existing probe.py
"""
from typing import Dict, Any, Optional, List
import torch

class ProbeRunner:
    
    def __init__(self):
        self.loaded_probes: Dict[str, Any] = {}  # key: "model_dataset_layer_token"
    
    def load_probe(
        self,
        model_id: str,
        dataset_id: str,
        layer: int,
        token_type: str
    ) -> bool:
        """
        Load probe classifier from checkpoint.
        Returns: True if loaded successfully, False otherwise
        """
        pass
    
    def predict(
        self,
        hidden_state: torch.Tensor,
        model_id: str,
        dataset_id: str,
        layer: int,
        token_type: str
    ) -> Dict[str, Any]:
        """
        Run probe prediction on hidden state.
        Returns: {"confidence": float, "prediction": str, "probabilities": [float, float]}
        """
        pass
    
    def predict_all_layers(
        self,
        hidden_states: Dict[int, torch.Tensor],
        model_id: str,
        dataset_id: str,
        token_type: str
    ) -> Dict[int, Dict[str, Any]]:
        """
        Run probe on all layers.
        Returns: {layer_idx: prediction_dict}
        """
        pass
    
    def is_probe_available(
        self,
        model_id: str,
        dataset_id: str
    ) -> bool:
        """Check if any probe exists for model/dataset combo."""
        pass
```

#### `app.py`
```python
"""
FastAPI application with all routes.
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="LLMsKnow Visualization API",
    version="1.0.0",
    description="Backend API for LLM layer visualization tool"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model Routes ---
@app.get("/api/models")
def list_models(): ...

@app.post("/api/models/{model_id}/load")
def load_model(model_id: str): ...

@app.get("/api/models/status")
def get_model_status(): ...

# --- Dataset Routes ---
@app.get("/api/datasets")
def list_datasets(): ...

@app.get("/api/datasets/{dataset_id}/samples")
def get_samples(dataset_id: str, page: int = 1, page_size: int = 20, search: str = None): ...

@app.get("/api/datasets/{dataset_id}/sample/{sample_idx}")
def get_sample(dataset_id: str, sample_idx: int): ...

# --- Availability Routes ---
@app.get("/api/combinations")
def list_combinations(model_id: str = None, ready_only: bool = False): ...

# --- Inference Routes ---
@app.post("/api/generate")
def generate(request: GenerateRequest): ...

# --- Session Data Routes ---
@app.get("/api/session")
def get_session(): ...

@app.get("/api/session/layer/{layer_idx}")
def get_layer_data(layer_idx: int): ...

@app.get("/api/session/attention/{layer_idx}/{head_idx}")
def get_attention_data(layer_idx: int, head_idx: int): ...

@app.get("/api/session/attention/flow")
def get_attention_flow(threshold: float = 0.1): ...

@app.get("/api/session/probe")
def get_probe_predictions(): ...

@app.get("/api/session/dimensions")
def get_dimension_flow(n_components: int = 10): ...
```

---

## 3. Frontend Design

### 3.1 Component Hierarchy

```
App
├── Header
│   ├── Logo
│   ├── ModelSelector
│   └── ModelLoadingIndicator
│
├── MainLayout
│   ├── InputSection
│   │   ├── InputPanel
│   │   │   ├── InputSourceToggle (Dataset | Custom)
│   │   │   ├── DatasetBrowser (when source=dataset)
│   │   │   │   ├── DatasetDropdown
│   │   │   │   ├── CoverageInfo
│   │   │   │   ├── SampleSearch
│   │   │   │   └── SampleTable
│   │   │   │       ├── SampleRow (selectable)
│   │   │   │       └── Pagination
│   │   │   ├── CustomInput (when source=custom)
│   │   │   │   ├── QuestionInput
│   │   │   │   └── ExpectedAnswerInput
│   │   │   └── SelectedInputDisplay
│   │   └── RunButton
│   │
│   ├── OutputSection
│   │   ├── OutputPanel
│   │   │   ├── CorrectnessIndicator
│   │   │   └── TokenDisplay
│   │   │       └── TokenChip (clickable)
│   │   └── LoadingOverlay (conditional)
│   │
│   └── VisualizationSection
│       ├── VisualizationPanel
│       │   ├── DisabledOverlay (when !hasRun)
│       │   ├── ViewModeSelector
│       │   ├── VisualizationArea
│       │   │   ├── LayerFlowView (D3)
│       │   │   ├── AttentionFlowView (D3)
│       │   │   └── DimensionFlowView (D3)
│       │   └── VisualizationControls
│       │       ├── ConnectionThresholdSlider
│       │       ├── CrossHeadToggle
│       │       └── AnimationToggle
│       │
│       └── InsightPanel
│           ├── SelectionInfo
│           ├── StatisticsDisplay
│           ├── AttentionHeatmap
│           ├── HeadRankingChart
│           └── TokenJourneyChart
│
└── ErrorBoundary
    └── ErrorDisplay
```

### 3.2 State Management

Using **Zustand** for simplicity and TypeScript support:

```typescript
// stores/appStore.ts

interface AppState {
  // Model state
  selectedModelId: string | null;
  modelStatus: 'idle' | 'loading' | 'loaded' | 'error';
  modelLoadProgress: number;
  modelError: string | null;
  
  // Input state
  inputSource: 'dataset' | 'custom';
  selectedDatasetId: string | null;
  selectedSample: Sample | null;
  customQuestion: string;
  customExpectedAnswer: string;
  
  // Dataset browser state
  datasetPage: number;
  datasetPageSize: number;
  searchQuery: string;
  
  // Run state
  hasRun: boolean;
  isRunning: boolean;
  runError: string | null;
  
  // Session data (after run)
  session: SessionData | null;
  
  // Visualization state
  viewMode: 'layer_overview' | 'attention_flow' | 'dimension_flow';
  selectedTokenIdx: number | null;
  selectedLayerIdx: number | null;
  selectedHeadIdx: number | null;
  connectionThreshold: number;
  showCrossHeadConnections: boolean;
  animateFlow: boolean;
  
  // Actions
  setModel: (modelId: string) => void;
  loadModel: () => Promise<void>;
  setInputSource: (source: 'dataset' | 'custom') => void;
  setSelectedDataset: (datasetId: string) => void;
  setSelectedSample: (sample: Sample) => void;
  setCustomQuestion: (question: string) => void;
  setCustomExpectedAnswer: (answer: string) => void;
  runInference: () => Promise<void>;
  clearSession: () => void;
  setViewMode: (mode: ViewMode) => void;
  selectToken: (idx: number | null) => void;
  selectLayer: (idx: number | null) => void;
  selectHead: (idx: number | null) => void;
  setConnectionThreshold: (value: number) => void;
  toggleCrossHeadConnections: () => void;
  toggleAnimateFlow: () => void;
}
```

### 3.3 Hooks

```typescript
// hooks/useModels.ts
function useModels() {
  // Fetch available models with their ready dataset counts
  return { models, isLoading, error };
}

// hooks/useCombinations.ts
function useCombinations(modelId: string | null) {
  // Fetch available combinations for selected model
  return { combinations, isLoading, error, refetch };
}

// hooks/useDatasets.ts
function useDatasets() {
  // Fetch available datasets
  return { datasets, isLoading, error };
}

// hooks/useSamples.ts
function useSamples(datasetId: string | null, page: number, search: string) {
  // Fetch paginated samples with search
  return { samples, totalPages, totalSamples, isLoading, error };
}

// hooks/useSession.ts
function useSession() {
  // Fetch current session data after run
  return { session, isLoading, error, refetch };
}

// hooks/useLayerData.ts
function useLayerData(layerIdx: number | null) {
  // Fetch detailed layer data when selected
  return { layerData, isLoading, error };
}

// hooks/useAttentionData.ts
function useAttentionData(layerIdx: number | null, headIdx: number | null) {
  // Fetch attention pattern for specific head
  return { attentionData, isLoading, error };
}

// hooks/useAttentionFlow.ts
function useAttentionFlow(threshold: number) {
  // Fetch computed attention flow
  return { flowData, isLoading, error };
}
```

### 3.4 File Structure

```
visualization/
├── frontend/
│   ├── public/
│   │   └── index.html
│   │
│   ├── src/
│   │   ├── App.tsx
│   │   ├── main.tsx
│   │   │
│   │   ├── components/
│   │   │   ├── layout/
│   │   │   │   ├── Header.tsx
│   │   │   │   ├── MainLayout.tsx
│   │   │   │   └── Section.tsx
│   │   │   │
│   │   │   ├── input/
│   │   │   │   ├── InputPanel.tsx
│   │   │   │   ├── InputSourceToggle.tsx
│   │   │   │   ├── DatasetBrowser.tsx
│   │   │   │   ├── DatasetDropdown.tsx
│   │   │   │   ├── CoverageInfo.tsx
│   │   │   │   ├── SampleSearch.tsx
│   │   │   │   ├── SampleTable.tsx
│   │   │   │   ├── SampleRow.tsx
│   │   │   │   ├── Pagination.tsx
│   │   │   │   ├── CustomInput.tsx
│   │   │   │   ├── SelectedInputDisplay.tsx
│   │   │   │   └── RunButton.tsx
│   │   │   │
│   │   │   ├── output/
│   │   │   │   ├── OutputPanel.tsx
│   │   │   │   ├── CorrectnessIndicator.tsx
│   │   │   │   ├── TokenDisplay.tsx
│   │   │   │   └── TokenChip.tsx
│   │   │   │
│   │   │   ├── visualization/
│   │   │   │   ├── VisualizationPanel.tsx
│   │   │   │   ├── ViewModeSelector.tsx
│   │   │   │   ├── VisualizationArea.tsx
│   │   │   │   ├── VisualizationControls.tsx
│   │   │   │   ├── LayerFlowView.tsx          # D3
│   │   │   │   ├── AttentionFlowView.tsx      # D3
│   │   │   │   ├── DimensionFlowView.tsx      # D3
│   │   │   │   └── shared/
│   │   │   │       ├── NodeRenderer.tsx
│   │   │   │       ├── ConnectionRenderer.tsx
│   │   │   │       └── Tooltip.tsx
│   │   │   │
│   │   │   ├── insight/
│   │   │   │   ├── InsightPanel.tsx
│   │   │   │   ├── SelectionInfo.tsx
│   │   │   │   ├── StatisticsDisplay.tsx
│   │   │   │   ├── AttentionHeatmap.tsx       # D3
│   │   │   │   ├── HeadRankingChart.tsx       # D3
│   │   │   │   └── TokenJourneyChart.tsx      # D3
│   │   │   │
│   │   │   ├── common/
│   │   │   │   ├── Button.tsx
│   │   │   │   ├── Dropdown.tsx
│   │   │   │   ├── Slider.tsx
│   │   │   │   ├── Toggle.tsx
│   │   │   │   ├── LoadingOverlay.tsx
│   │   │   │   ├── DisabledOverlay.tsx
│   │   │   │   ├── ErrorDisplay.tsx
│   │   │   │   └── ProgressBar.tsx
│   │   │   │
│   │   │   └── model/
│   │   │       ├── ModelSelector.tsx
│   │   │       └── ModelLoadingIndicator.tsx
│   │   │
│   │   ├── hooks/
│   │   │   ├── useModels.ts
│   │   │   ├── useCombinations.ts
│   │   │   ├── useDatasets.ts
│   │   │   ├── useSamples.ts
│   │   │   ├── useSession.ts
│   │   │   ├── useLayerData.ts
│   │   │   ├── useAttentionData.ts
│   │   │   └── useAttentionFlow.ts
│   │   │
│   │   ├── stores/
│   │   │   └── appStore.ts
│   │   │
│   │   ├── api/
│   │   │   ├── client.ts           # Axios instance
│   │   │   ├── models.ts           # Model API calls
│   │   │   ├── datasets.ts         # Dataset API calls
│   │   │   ├── combinations.ts     # Availability API calls
│   │   │   ├── inference.ts        # Inference API calls
│   │   │   └── session.ts          # Session data API calls
│   │   │
│   │   ├── types/
│   │   │   ├── index.ts            # Re-exports
│   │   │   ├── models.ts           # Model types
│   │   │   ├── datasets.ts         # Dataset types
│   │   │   ├── session.ts          # Session types
│   │   │   └── visualization.ts    # Visualization types
│   │   │
│   │   ├── utils/
│   │   │   ├── colorScale.ts       # Correctness color mapping
│   │   │   ├── formatters.ts       # Number/text formatters
│   │   │   └── d3Helpers.ts        # D3 utility functions
│   │   │
│   │   └── styles/
│   │       ├── globals.css
│   │       └── variables.css
│   │
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   └── tailwind.config.js
```

---

## 4. Data Flow

### 4.1 Initial Load Flow

```
┌─────────────┐     GET /api/models      ┌─────────────┐
│   App.tsx   │ ──────────────────────▶  │   Backend   │
│ (useModels) │ ◀──────────────────────  │             │
└─────────────┘    [{id, name, ready}]   └─────────────┘
      │
      │ User selects model
      ▼
┌─────────────┐   GET /api/combinations   ┌─────────────┐
│  useCombi-  │      ?model_id=...        │   Backend   │
│  nations    │ ──────────────────────▶   │ (Scanner)   │
│             │ ◀──────────────────────   │             │
└─────────────┘   [{dataset, status}]     └─────────────┘
      │
      │ Populate dataset dropdown
      ▼
┌─────────────┐   POST /api/models/load   ┌─────────────┐
│  loadModel  │ ──────────────────────▶   │ModelManager │
│  (action)   │ ◀──────────────────────   │             │
└─────────────┘    {status, progress}     └─────────────┘
```

### 4.2 Dataset Selection Flow

```
┌─────────────┐     GET /api/datasets     ┌─────────────┐
│  useData-   │      /{id}/samples        │  Dataset    │
│  sets       │ ──────────────────────▶   │  Manager    │
│             │ ◀──────────────────────   │             │
└─────────────┘    {samples, pages}       └─────────────┘
      │
      │ User selects sample row
      ▼
┌─────────────┐
│ setSelected │  Update store.selectedSample
│  Sample     │  Display in SelectedInputDisplay
└─────────────┘
      │
      │ Run button becomes enabled
      ▼
```

### 4.3 Inference Flow

```
┌─────────────┐     POST /api/generate    ┌─────────────┐
│ runInfer-   │ ──────────────────────▶   │   Backend   │
│  ence       │    {question, expected,   │             │
│             │     model_id}             │             │
└─────────────┘                           └─────────────┘
      │                                         │
      │                                         │
      │                                         ▼
      │                                   ┌─────────────┐
      │                                   │ Model       │
      │                                   │ Manager     │
      │                                   │  .generate()│
      │                                   └──────┬──────┘
      │                                          │
      │                                          ▼
      │                                   ┌─────────────┐
      │                                   │ Layer &     │
      │                                   │ Attention   │
      │                                   │ Extractors  │
      │                                   └──────┬──────┘
      │                                          │
      │                                          ▼
      │                                   ┌─────────────┐
      │                                   │ Probe       │
      │                                   │ Runner      │
      │                                   └──────┬──────┘
      │                                          │
      │                                          ▼
      │                                   ┌─────────────┐
      │                                   │ Session     │
      │                                   │ Manager     │
      │                                   │ (store all) │
      │                                   └──────┬──────┘
      │                                          │
      │ ◀────────────────────────────────────────┘
      │    {session_id, answer, is_correct, tokens}
      ▼
┌─────────────┐
│ setSession  │  Store session, enable visualization
└─────────────┘
```

### 4.4 Visualization Data Fetch Flow

```
┌─────────────┐   GET /api/session/layer/16  ┌─────────────┐
│ useLayer    │ ─────────────────────────▶   │  Session    │
│  Data(16)   │ ◀─────────────────────────   │  Manager    │
└─────────────┘    {activations, probe}      └─────────────┘

┌─────────────┐   GET /api/session/attention ┌─────────────┐
│ useAtten-   │        /16/12                │  Session    │
│  tionData   │ ─────────────────────────▶   │  Manager    │
│  (16, 12)   │ ◀─────────────────────────   │             │
└─────────────┘    {pattern, stats, top}     └─────────────┘

┌─────────────┐   GET /api/session/attention ┌─────────────┐
│ useAtten-   │       /flow?threshold=0.1    │  Attention  │
│  tionFlow   │ ─────────────────────────▶   │  Extractor  │
│             │ ◀─────────────────────────   │             │
└─────────────┘    [{source, target, conns}] └─────────────┘
```

---

## 5. Type Definitions

### 5.1 Backend Types (Pydantic)

```python
# schemas/requests.py
from pydantic import BaseModel
from typing import Optional

class GenerateRequest(BaseModel):
    question: str
    expected_answer: Optional[str] = None
    model_id: str
    extract_hidden_states: bool = True
    extract_attentions: bool = True

class LoadModelRequest(BaseModel):
    model_id: str
```

```python
# schemas/responses.py
from pydantic import BaseModel
from typing import List, Dict, Optional, Any

class ModelInfo(BaseModel):
    id: str
    name: str
    num_layers: int
    num_heads: int
    ready_datasets: int
    total_datasets: int

class DatasetInfo(BaseModel):
    id: str
    name: str
    total_samples: int
    category: str
    columns: List[str]

class SampleInfo(BaseModel):
    idx: int
    question: str
    answer: str

class DatasetPage(BaseModel):
    dataset_id: str
    page: int
    page_size: int
    total_pages: int
    total_samples: int
    samples: List[SampleInfo]
    search_query: Optional[str]

class CombinationInfo(BaseModel):
    model_id: str
    model_name: str
    dataset_id: str
    dataset_name: str
    has_answers: bool
    has_input_output_ids: bool
    has_probe: bool
    probe_config: Optional[Dict[str, Any]]
    samples_processed: int
    samples_total: int
    samples_coverage: float
    accuracy: Optional[float]
    ready_for_visualization: bool
    ready_for_probe_predictions: bool

class TokenInfo(BaseModel):
    idx: int
    text: str
    token_id: int

class LayerData(BaseModel):
    layer_idx: int
    statistics: Dict[str, float]
    probe_prediction: Optional[Dict[str, Any]]
    top_neurons: List[Dict[str, Any]]

class AttentionHeadData(BaseModel):
    layer_idx: int
    head_idx: int
    pattern: List[List[float]]  # seq x seq
    statistics: Dict[str, float]
    top_attended_tokens: List[Dict[str, Any]]

class AttentionFlowConnection(BaseModel):
    source_head: int
    target_head: int
    weight: float

class AttentionFlowLayer(BaseModel):
    source_layer: int
    target_layer: int
    head_connections: List[AttentionFlowConnection]

class SessionResponse(BaseModel):
    session_id: str
    model_id: str
    question: str
    expected_answer: Optional[str]
    generated_answer: str
    is_correct: Optional[bool]
    input_tokens: List[TokenInfo]
    output_tokens: List[TokenInfo]
    num_layers: int
    num_heads: int

class GenerateResponse(BaseModel):
    session: SessionResponse
    # Detailed layer/attention data fetched separately
```

### 5.2 Frontend Types (TypeScript)

```typescript
// types/models.ts
export interface Model {
  id: string;
  name: string;
  numLayers: number;
  numHeads: number;
  readyDatasets: number;
  totalDatasets: number;
}

export interface ModelStatus {
  status: 'idle' | 'loading' | 'loaded' | 'error';
  progress: number;
  error?: string;
}
```

```typescript
// types/datasets.ts
export interface Dataset {
  id: string;
  name: string;
  totalSamples: number;
  category: string;
  columns: string[];
}

export interface Sample {
  idx: number;
  question: string;
  answer: string;
}

export interface DatasetPage {
  datasetId: string;
  page: number;
  pageSize: number;
  totalPages: number;
  totalSamples: number;
  samples: Sample[];
  searchQuery: string | null;
}

export interface Combination {
  modelId: string;
  modelName: string;
  datasetId: string;
  datasetName: string;
  hasAnswers: boolean;
  hasProbe: boolean;
  probeConfig: { layer: number; token: string } | null;
  samplesProcessed: number;
  samplesTotal: number;
  samplesCoverage: number;
  accuracy: number | null;
  readyForVisualization: boolean;
  readyForProbePredictions: boolean;
}
```

```typescript
// types/session.ts
export interface Token {
  idx: number;
  text: string;
  tokenId: number;
}

export interface Session {
  sessionId: string;
  modelId: string;
  question: string;
  expectedAnswer: string | null;
  generatedAnswer: string;
  isCorrect: boolean | null;
  inputTokens: Token[];
  outputTokens: Token[];
  numLayers: number;
  numHeads: number;
}

export interface LayerData {
  layerIdx: number;
  statistics: {
    mean: number;
    std: number;
    max: number;
    min: number;
    norm: number;
  };
  probePrediction: {
    confidence: number;
    prediction: 'correct' | 'incorrect';
    probabilities: [number, number];
  } | null;
  topNeurons: Array<{ idx: number; activation: number }>;
}

export interface AttentionHeadData {
  layerIdx: number;
  headIdx: number;
  pattern: number[][];
  statistics: {
    avg: number;
    max: number;
    entropy: number;
    sparsity: number;
  };
  topAttendedTokens: Array<{
    tokenIdx: number;
    tokenText: string;
    weight: number;
  }>;
}

export interface AttentionFlowConnection {
  sourceHead: number;
  targetHead: number;
  weight: number;
}

export interface AttentionFlowLayer {
  sourceLayer: number;
  targetLayer: number;
  headConnections: AttentionFlowConnection[];
}
```

```typescript
// types/visualization.ts
export type ViewMode = 'layer_overview' | 'attention_flow' | 'dimension_flow';

export interface VisualizationState {
  isEnabled: boolean;
  viewMode: ViewMode;
  selectedTokenIdx: number | null;
  selectedLayerIdx: number | null;
  selectedHeadIdx: number | null;
  connectionThreshold: number;
  showCrossHeadConnections: boolean;
  animateFlow: boolean;
}
```

---

## 6. API Contracts

### 6.1 Model Endpoints

```yaml
GET /api/models:
  description: List available models
  response:
    models: ModelInfo[]

POST /api/models/{model_id}/load:
  description: Load model into GPU memory
  response:
    status: "loading" | "loaded" | "error"
    progress: number (0-100)
    error?: string

GET /api/models/status:
  description: Get current model loading status
  response:
    currentModelId: string | null
    status: "idle" | "loading" | "loaded" | "error"
    progress: number
    error?: string
```

### 6.2 Dataset Endpoints

```yaml
GET /api/datasets:
  description: List available datasets
  response:
    datasets: DatasetInfo[]

GET /api/datasets/{dataset_id}/samples:
  params:
    page: int (default: 1)
    page_size: int (default: 20, max: 100)
    search: string (optional)
  response:
    DatasetPage

GET /api/datasets/{dataset_id}/sample/{sample_idx}:
  response:
    datasetId: string
    sampleIdx: number
    question: string
    expectedAnswer: string
    metadata: object
```

### 6.3 Availability Endpoints

```yaml
GET /api/combinations:
  params:
    model_id: string (optional, filter by model)
    ready_only: bool (default: false)
  response:
    combinations: CombinationInfo[]
```

### 6.4 Inference Endpoints

```yaml
POST /api/generate:
  body:
    question: string
    expected_answer: string | null
    model_id: string
  response:
    SessionResponse
  errors:
    400: Invalid request
    503: Model not loaded
    500: Inference failed
```

### 6.5 Session Data Endpoints

```yaml
GET /api/session:
  description: Get current session summary
  response:
    SessionResponse | null

GET /api/session/layer/{layer_idx}:
  response:
    LayerData

GET /api/session/attention/{layer_idx}/{head_idx}:
  response:
    AttentionHeadData

GET /api/session/attention/flow:
  params:
    threshold: float (default: 0.1)
  response:
    flow: AttentionFlowLayer[]

GET /api/session/probe:
  response:
    predictions: { [layerIdx: number]: ProbePrediction }

GET /api/session/dimensions:
  params:
    n_components: int (default: 10)
  response:
    layers: { [layerIdx: number]: number[] }
    explainedVariance: number[]
```

---

## 7. Integration with Existing Codebase

### 7.1 Files to Import From

| Existing File | What to Use | Backend Module |
|---------------|-------------|----------------|
| `probing_utils.py` | `load_model_and_validate_gpu()` | `model_manager.py` |
| `probing_utils.py` | `tokenize()`, `generate()` | `model_manager.py` |
| `probing_utils.py` | `extract_internal_reps_single_sample()` | `layer_extractor.py` |
| `probing_utils.py` | `MODEL_FRIENDLY_NAMES`, `LIST_OF_MODELS` | `config.py` |
| `probe.py` | `get_saved_clf_if_exists()` | `probe_runner.py` |
| `probe.py` | `get_layer_reps()` | `probe_runner.py` |
| `compute_correctness.py` | `compute_correctness_movies()` etc. | `utils/correctness.py` |

### 7.2 Import Pattern

```python
# In visualization/backend/managers/model_manager.py

import sys
from pathlib import Path

# Add src/ to path for imports
SRC_DIR = Path(__file__).parent.parent.parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from probing_utils import (
    load_model_and_validate_gpu,
    tokenize,
    generate,
    MODEL_FRIENDLY_NAMES,
    LIST_OF_MODELS,
)
```

### 7.3 Functions to Wrap

```python
# utils/correctness.py
"""Wrapper for compute_correctness functions."""

from compute_correctness import (
    compute_correctness_movies,
    compute_correctness_triviaqa,
    # ... other dataset-specific functions
)

def compute_correctness(
    model_output: str,
    expected_answer: str,
    dataset_category: str
) -> bool:
    """
    Unified correctness computation.
    Dispatches to dataset-specific function.
    """
    if dataset_category == "factual":
        # Use fuzzy matching for factual QA
        return compute_correctness_movies(model_output, expected_answer)
    elif dataset_category == "math":
        return compute_correctness_math(model_output, expected_answer)
    # ... etc
    else:
        # Default: exact match (case-insensitive)
        return model_output.strip().lower() == expected_answer.strip().lower()
```

---

## 8. Error Handling Strategy

### 8.1 Backend Error Classes

```python
# utils/exceptions.py

class VisualizationError(Exception):
    """Base exception for visualization backend."""
    pass

class ModelNotLoadedError(VisualizationError):
    """Model needs to be loaded before inference."""
    status_code = 503

class ModelLoadError(VisualizationError):
    """Failed to load model (CUDA OOM, etc.)."""
    status_code = 500

class InferenceError(VisualizationError):
    """Inference failed (timeout, generation error)."""
    status_code = 500

class DatasetNotFoundError(VisualizationError):
    """Dataset file not found."""
    status_code = 404

class ProbeNotFoundError(VisualizationError):
    """Probe checkpoint not found."""
    status_code = 404

class SessionNotFoundError(VisualizationError):
    """No active session."""
    status_code = 404
```

### 8.2 FastAPI Error Handlers

```python
# app.py

from fastapi import Request
from fastapi.responses import JSONResponse

@app.exception_handler(VisualizationError)
async def visualization_error_handler(request: Request, exc: VisualizationError):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.__class__.__name__,
            "message": str(exc),
            "retryable": exc.status_code >= 500
        }
    )
```

### 8.3 Frontend Error Handling

```typescript
// api/client.ts

import axios from 'axios';

const client = axios.create({
  baseURL: 'http://localhost:8000/api',
  timeout: 120000, // 2 min for inference
});

// Response interceptor for error handling
client.interceptors.response.use(
  (response) => response,
  (error) => {
    const apiError: ApiError = {
      type: error.response?.data?.error || 'NetworkError',
      message: error.response?.data?.message || error.message,
      retryable: error.response?.data?.retryable ?? true,
    };
    return Promise.reject(apiError);
  }
);
```

---

## 9. Caching Strategy

### 9.1 Backend Caching

```python
# Dataset caching (in-memory LRU)
@lru_cache(maxsize=8)
def _load_dataset_cached(dataset_id: str) -> pd.DataFrame:
    ...

# Probe caching (keep loaded probes)
class ProbeRunner:
    def __init__(self):
        self.loaded_probes: Dict[str, Any] = {}  # Persists across requests

# Model caching (singleton, only one model at a time)
class ModelManager:
    _instance = None  # Singleton
    
    def load_model(self, model_id: str):
        if self.current_model_id == model_id:
            return  # Already loaded, skip
        # Unload previous, load new
```

### 9.2 Frontend Caching

```typescript
// Using React Query or SWR for data fetching with caching

// hooks/useSamples.ts
function useSamples(datasetId: string | null, page: number, search: string) {
  return useQuery({
    queryKey: ['samples', datasetId, page, search],
    queryFn: () => fetchSamples(datasetId!, page, search),
    enabled: !!datasetId,
    staleTime: 5 * 60 * 1000, // 5 minutes
    cacheTime: 30 * 60 * 1000, // 30 minutes
  });
}

// Session data - no cache (always fresh after run)
function useSession() {
  return useQuery({
    queryKey: ['session'],
    queryFn: fetchSession,
    staleTime: 0, // Always refetch
  });
}
```

---

## 10. Performance Considerations

### 10.1 Attention Data Size
- 32 layers × 32 heads × (seq_len × seq_len) attention matrices
- For seq_len=100: 32 × 32 × 100 × 100 × 4 bytes ≈ 40MB per inference
- **Mitigations**: 
  - Store only diagonal/important values
  - Compute statistics on-the-fly, don't store full matrices
  - Allow lazy loading of full attention for selected layer/head

### 10.2 Rendering Performance
- 32 × 32 = 1024 nodes
- Up to 32 × 32 × 31 = 31,744 connections (cross-head enabled)
- **Mitigations**:
  - Use connection threshold to limit visible connections
  - Consider WebGL-based rendering (Three.js) if D3 SVG is too slow
  - Virtualize: only render visible layers in viewport

### 10.3 Memory Management
- Clear previous session data when running new question
- Use Web Workers for heavy frontend computation
- Stream large responses from backend (chunked transfer)

### 10.4 Backend Optimization
- Model loaded once, kept in memory (singleton)
- Dataset caching with LRU (max 8 datasets)
- Probe caching (loaded probes persist across requests)
- Compute attention statistics lazily on demand

---

## 11. Implementation Checklist

### Phase 1: Project Setup (1 hour)
- [ ] Create `visualization/` directory structure
- [ ] Backend: Create `requirements.txt` (fastapi, uvicorn, pydantic)
- [ ] Backend: Create `config.py` with paths and constants
- [ ] Frontend: Initialize React+Vite+TypeScript project
- [ ] Frontend: Install dependencies (d3, zustand, axios, tailwind)
- [ ] Create `run.py` to start both servers

### Phase 2: Backend Core (3 hours)
- [ ] Implement `app.py` with FastAPI boilerplate + CORS
- [ ] Implement `schemas/requests.py` and `schemas/responses.py`
- [ ] Implement `managers/model_manager.py`
  - [ ] `load_model()` with existing `probing_utils.py`
  - [ ] `generate()` with hidden state extraction
  - [ ] `is_loaded()`, `unload()`, `get_model_config()`
- [ ] Implement `managers/session_manager.py`
- [ ] Add routes: `/api/models`, `/api/models/{id}/load`, `/api/models/status`
- [ ] Test: Load model, generate text

### Phase 3: Dataset & Availability (2 hours)
- [ ] Implement `managers/dataset_manager.py`
  - [ ] `get_available_datasets()`
  - [ ] `get_samples()` with pagination and search
  - [ ] `get_sample()`
- [ ] Implement `scanners/availability_scanner.py`
  - [ ] `scan_all_combinations()`
  - [ ] Parse output files and checkpoints
- [ ] Add routes: `/api/datasets`, `/api/datasets/{id}/samples`, `/api/combinations`
- [ ] Test: List datasets, paginate, search, check combinations

### Phase 4: Extractors (2 hours)
- [ ] Implement `extractors/layer_extractor.py`
  - [ ] `extract_hidden_states()`
  - [ ] `compute_layer_statistics()`
- [ ] Implement `extractors/attention_extractor.py`
  - [ ] `extract_attention_patterns()`
  - [ ] `compute_head_statistics()`
  - [ ] `compute_attention_flow()`
- [ ] Implement `extractors/dimension_reducer.py`
  - [ ] PCA on layer representations
- [ ] Test: Extract from real model inference

### Phase 5: Probe Runner (1 hour)
- [ ] Implement `runners/probe_runner.py`
  - [ ] `load_probe()` from checkpoints
  - [ ] `predict()` single layer
  - [ ] `predict_all_layers()`
  - [ ] `is_probe_available()`
- [ ] Test: Load probe, run prediction

### Phase 6: Full Inference Pipeline (1 hour)
- [ ] Implement `/api/generate` route
  - [ ] Load model if needed
  - [ ] Generate answer
  - [ ] Extract hidden states & attention
  - [ ] Run probes
  - [ ] Compute correctness
  - [ ] Store in session
- [ ] Add session data routes: `/api/session/*`
- [ ] Test: Full inference pipeline

### Phase 7: Frontend Foundation (3 hours)
- [ ] Create `stores/appStore.ts` with Zustand
- [ ] Create `api/client.ts` with Axios
- [ ] Create API modules: `api/models.ts`, `api/datasets.ts`, etc.
- [ ] Create type definitions in `types/`
- [ ] Create basic layout components: `Header`, `MainLayout`
- [ ] Create common components: `Button`, `Dropdown`, `LoadingOverlay`

### Phase 8: Input Components (2 hours)
- [ ] Create `ModelSelector` with loading indicator
- [ ] Create `InputSourceToggle`
- [ ] Create `DatasetBrowser` with:
  - [ ] `DatasetDropdown`
  - [ ] `CoverageInfo`
  - [ ] `SampleTable` + `Pagination`
  - [ ] `SampleSearch`
- [ ] Create `CustomInput` (question + expected answer)
- [ ] Create `SelectedInputDisplay`
- [ ] Create `RunButton` with state management
- [ ] Wire up to store and API

### Phase 9: Output Components (1 hour)
- [ ] Create `OutputPanel`
- [ ] Create `CorrectnessIndicator`
- [ ] Create `TokenDisplay` with `TokenChip`
- [ ] Create `DisabledOverlay` for visualization

### Phase 10: Layer Flow Visualization (2 hours)
- [ ] Create `VisualizationPanel` container
- [ ] Create `ViewModeSelector`
- [ ] Create `LayerFlowView` with D3
  - [ ] Layer nodes
  - [ ] Connection lines
  - [ ] Correctness colors
  - [ ] Click handler

### Phase 11: Attention Flow Visualization (3 hours)
- [ ] Create `AttentionFlowView` with D3
  - [ ] Grid layout (layers × heads)
  - [ ] Head nodes with correctness colors
  - [ ] Horizontal connections (same head)
  - [ ] Diagonal connections (cross-head)
  - [ ] Threshold slider
  - [ ] Cross-head toggle
- [ ] Create `VisualizationControls`

### Phase 12: Insight Panel (2 hours)
- [ ] Create `InsightPanel` container
- [ ] Create `SelectionInfo`
- [ ] Create `StatisticsDisplay`
- [ ] Create `AttentionHeatmap` with D3
- [ ] Create `HeadRankingChart` with D3
- [ ] Wire to selection state

### Phase 13: Dimension Flow Visualization (1 hour)
- [ ] Create `DimensionFlowView` with D3
- [ ] PCA component display
- [ ] Importance lines
- [ ] Slider for n_components

### Phase 14: Polish & Testing (2 hours)
- [ ] Error states and messages
- [ ] Loading states everywhere
- [ ] Responsive layout
- [ ] Keyboard navigation
- [ ] Performance optimization
- [ ] End-to-end testing

---

## Summary

| Component | Files | Est. Hours |
|-----------|-------|------------|
| Backend Core | config, app, schemas, model_manager, session_manager | 4 |
| Backend Data | dataset_manager, availability_scanner | 2 |
| Backend Extract | layer_extractor, attention_extractor, dimension_reducer | 2 |
| Backend Probe | probe_runner, correctness wrapper | 1 |
| Frontend Setup | store, api client, types | 2 |
| Frontend Input | model selector, dataset browser, custom input | 3 |
| Frontend Output | output panel, tokens, correctness | 1 |
| Frontend Viz | layer flow, attention flow, dimension flow | 6 |
| Frontend Insight | insight panel, heatmap, charts | 2 |
| Polish | error handling, loading states, testing | 2 |
| **TOTAL** | | **~25 hours** |

---

## Appendix A: Full Implementation - `availability_scanner.py`

```python
"""
Scans output/ and checkpoints/ to determine which model/dataset combinations
have trained probes and pre-computed data.
"""
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd

from .config import (
    OUTPUT_DIR, CHECKPOINTS_DIR, DATA_DIR,
    SUPPORTED_MODELS, DATASET_CONFIG
)

def get_available_combinations() -> List[Dict[str, Any]]:
    """
    Scan output/ and checkpoints/ to find all available model/dataset combinations.
    Returns list of combinations with their availability status.
    
    IMPORTANT: The existing pipeline uses "output_id" (e.g., "movies") in output filenames,
    not "dataset_id" (e.g., "movie_qa_train"). This function maps between them.
    """
    combinations = []
    
    for model_id, model_config in SUPPORTED_MODELS.items():
        model_friendly = model_config["friendly_name"]
        
        for dataset_id, dataset_config in DATASET_CONFIG.items():
            # IMPORTANT: Use output_id for matching output files, not dataset_id
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
            
            # Parse probe config from filename if exists
            probe_config = _parse_probe_filename(probe_files[0].name) if has_probe else None
            
            # Get sample count and accuracy if answers file exists
            samples_processed, accuracy = (
                _count_processed_samples(answers_file) if has_answers else (0, None)
            )
            
            # Get total samples available in raw dataset
            samples_total = get_total_samples_in_dataset(dataset_id)
            
            combinations.append({
                "model_id": model_id,
                "model_name": model_friendly,
                "dataset_id": dataset_id,
                "dataset_name": DATASET_CONFIG[dataset_id]["display_name"],
                "has_answers": has_answers,
                "has_input_output_ids": has_ids,
                "has_probe": has_probe,
                "probe_config": probe_config,
                "samples_processed": samples_processed,
                "samples_total": samples_total,
                "samples_coverage": (
                    round(samples_processed / samples_total, 2) 
                    if samples_total > 0 else 0
                ),
                "accuracy": round(accuracy, 3) if accuracy else None,
                "ready_for_visualization": has_answers and has_ids,
                "ready_for_probe_predictions": has_probe
            })
    
    return combinations


def _parse_probe_filename(filename: str) -> Optional[Dict[str, Any]]:
    """Parse probe checkpoint filename for config."""
    # Pattern: clf_mistral-7b-instruct_movies_layer-15_token-exact_answer_last_token.pkl
    match = re.search(r'layer-(\d+)_token-(.+)\.pkl', filename)
    if match:
        return {
            "layer": int(match.group(1)),
            "token": match.group(2)
        }
    return None


def _count_processed_samples(answers_file: Path) -> Tuple[int, Optional[float]]:
    """Count samples and accuracy from answers CSV."""
    try:
        df = pd.read_csv(answers_file)
        sample_count = len(df)
        accuracy = (
            df['automatic_correctness'].mean() 
            if 'automatic_correctness' in df.columns else None
        )
        return sample_count, accuracy
    except Exception:
        return 0, None


def get_total_samples_in_dataset(dataset_id: str) -> int:
    """Get total number of samples available in raw dataset CSV."""
    if dataset_id not in DATASET_CONFIG:
        return 0
    
    filename = DATASET_CONFIG[dataset_id]["filename"]
    filepath = DATA_DIR / filename
    
    if not filepath.exists():
        return 0
    
    try:
        df = pd.read_csv(filepath)
        return len(df)
    except Exception:
        return 0


def get_combinations_for_model(model_id: str) -> List[Dict[str, Any]]:
    """Get combinations filtered by model."""
    all_combos = get_available_combinations()
    return [c for c in all_combos if c["model_id"] == model_id]


def get_ready_combinations() -> List[Dict[str, Any]]:
    """Get only combinations that are ready for full visualization."""
    all_combos = get_available_combinations()
    return [c for c in all_combos if c["ready_for_probe_predictions"]]
```

---

## Appendix B: Full Implementation - `dataset_manager.py`

```python
"""
Dataset loading and pagination with LRU caching.
"""
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from functools import lru_cache

from .config import DATA_DIR, DATASET_CONFIG


@lru_cache(maxsize=8)
def _load_dataset_cached(dataset_id: str) -> pd.DataFrame:
    """Internal cached loader."""
    if dataset_id not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset: {dataset_id}")
    
    config = DATASET_CONFIG[dataset_id]
    filepath = DATA_DIR / config["filename"]
    
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found: {filepath}")
    
    return pd.read_csv(filepath)


class DatasetManager:
    
    @staticmethod
    def get_available_datasets() -> List[Dict[str, Any]]:
        """List all available datasets with metadata."""
        datasets = []
        
        for dataset_id, config in DATASET_CONFIG.items():
            filepath = DATA_DIR / config["filename"]
            if filepath.exists():
                df = _load_dataset_cached(dataset_id)
                datasets.append({
                    "id": dataset_id,
                    "name": config["display_name"],
                    "filename": config["filename"],
                    "total_samples": len(df),
                    "category": config["category"],
                    "columns": list(df.columns)
                })
        
        return datasets
    
    @staticmethod
    def get_samples(
        dataset_id: str,
        page: int = 1,
        page_size: int = 20,
        search_query: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get paginated samples with optional search."""
        df = _load_dataset_cached(dataset_id)
        config = DATASET_CONFIG[dataset_id]
        
        question_col = config["question_col"]
        answer_col = config["answer_col"]
        
        # Apply search filter
        if search_query:
            mask = df[question_col].str.contains(search_query, case=False, na=False)
            df = df[mask].reset_index(drop=True)
        
        total_samples = len(df)
        total_pages = (total_samples + page_size - 1) // page_size
        
        # Paginate
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total_samples)
        page_df = df.iloc[start_idx:end_idx]
        
        samples = []
        for i, (_, row) in enumerate(page_df.iterrows()):
            samples.append({
                "idx": start_idx + i,
                "question": str(row[question_col]),
                "answer": str(row[answer_col])
            })
        
        return {
            "dataset_id": dataset_id,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
            "total_samples": total_samples,
            "samples": samples,
            "search_query": search_query
        }
    
    @staticmethod
    def get_sample(dataset_id: str, sample_idx: int) -> Dict[str, Any]:
        """Get single sample by index."""
        df = _load_dataset_cached(dataset_id)
        config = DATASET_CONFIG[dataset_id]
        
        if sample_idx < 0 or sample_idx >= len(df):
            raise IndexError(
                f"Sample index {sample_idx} out of range for dataset {dataset_id}"
            )
        
        row = df.iloc[sample_idx]
        
        return {
            "dataset_id": dataset_id,
            "sample_idx": sample_idx,
            "question": str(row[config["question_col"]]),
            "expected_answer": str(row[config["answer_col"]]),
            "metadata": {
                "category": config["category"],
                "source": dataset_id
            }
        }
    
    @staticmethod
    def get_total_samples(dataset_id: str) -> int:
        """Get total sample count for a dataset."""
        df = _load_dataset_cached(dataset_id)
        return len(df)
```

---

## Appendix C: Full Implementation - `attention_extractor.py`

```python
"""
Extract and process attention patterns from model forward pass.

NOTE: This uses model's native `output_attentions=True` parameter rather than
baukit's TraceDict (used in probing_utils.py for layer representation extraction).
This is intentional: we need raw attention weights (softmax outputs), not the
projected outputs traced by probing_utils.

The two approaches are complementary:
- probing_utils.py TraceDict: Extracts layer outputs/hidden states for probing
- AttentionExtractor: Extracts raw attention patterns for visualization
"""
import torch
import numpy as np
from typing import Dict, Any, List, Optional

class AttentionExtractor:
    
    @staticmethod
    def extract_attention_patterns(
        model,
        input_ids: torch.Tensor,
        layers: Optional[List[int]] = None,
        heads: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Extract attention patterns from all layers and heads.
        
        Returns:
            {
                "patterns": {layer_idx: {head_idx: attention_matrix}},
                "statistics": {layer_idx: {head_idx: stats_dict}}
            }
        """
        model.eval()
        
        with torch.no_grad():
            outputs = model(
                input_ids,
                output_attentions=True,
                return_dict=True
            )
        
        # outputs.attentions is tuple of (batch, num_heads, seq_len, seq_len)
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
                
                # (seq_len, seq_len)
                attn = layer_attention[0, head_idx].cpu().numpy()
                
                attention_patterns[layer_idx][head_idx] = attn.tolist()
                attention_stats[layer_idx][head_idx] = (
                    AttentionExtractor.compute_head_statistics(attn)
                )
        
        return {
            "patterns": attention_patterns,
            "statistics": attention_stats
        }
    
    @staticmethod
    def compute_head_statistics(attention: np.ndarray) -> Dict[str, float]:
        """Compute statistics for attention head."""
        return {
            "avg": float(attention.mean()),
            "max": float(attention.max()),
            "entropy": float(AttentionExtractor._compute_entropy(attention)),
            "sparsity": float(AttentionExtractor._compute_sparsity(attention))
        }
    
    @staticmethod
    def _compute_entropy(attention: np.ndarray) -> float:
        """Compute entropy of attention distribution."""
        attn_flat = attention.flatten()
        attn_flat = attn_flat[attn_flat > 1e-10]  # avoid log(0)
        return -np.sum(attn_flat * np.log(attn_flat))
    
    @staticmethod
    def _compute_sparsity(attention: np.ndarray, threshold: float = 0.1) -> float:
        """Compute sparsity (fraction of weights below threshold)."""
        return np.mean(attention < threshold)
    
    @staticmethod
    def compute_attention_flow(
        attention_patterns: Dict[int, Dict[int, np.ndarray]],
        threshold: float = 0.1
    ) -> List[Dict[str, Any]]:
        """
        Compute cross-layer attention flow (head-to-head connections).
        
        Returns: [{"source_layer", "target_layer", "head_connections": [...]}]
        """
        flow = []
        layer_indices = sorted(attention_patterns.keys())
        
        for i in range(len(layer_indices) - 1):
            src_layer = layer_indices[i]
            tgt_layer = layer_indices[i + 1]
            
            layer_flow = {
                "source_layer": src_layer,
                "target_layer": tgt_layer,
                "head_connections": []
            }
            
            src_heads = list(attention_patterns[src_layer].keys())
            tgt_heads = list(attention_patterns[tgt_layer].keys())
            
            for src_head in src_heads:
                for tgt_head in tgt_heads:
                    src_attn = np.array(attention_patterns[src_layer][src_head])
                    tgt_attn = np.array(attention_patterns[tgt_layer][tgt_head])
                    
                    weight = AttentionExtractor._compute_similarity(src_attn, tgt_attn)
                    
                    if weight > threshold:
                        layer_flow["head_connections"].append({
                            "source_head": src_head,
                            "target_head": tgt_head,
                            "weight": weight
                        })
            
            flow.append(layer_flow)
        
        return flow
    
    @staticmethod
    def _compute_similarity(attn1: np.ndarray, attn2: np.ndarray) -> float:
        """Compute cosine similarity between two attention patterns."""
        a1 = attn1.flatten()
        a2 = attn2.flatten()
        
        norm1 = np.linalg.norm(a1)
        norm2 = np.linalg.norm(a2)
        
        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0
        
        return float(np.dot(a1, a2) / (norm1 * norm2))
    
    @staticmethod
    def get_top_attended_tokens(
        attention: np.ndarray,
        token_idx: int,
        token_texts: List[str],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Get tokens that a specific token attends to most."""
        attn_row = attention[token_idx]
        top_indices = np.argsort(attn_row)[-top_k:][::-1]
        
        return [
            {
                "token_idx": int(idx),
                "token_text": token_texts[idx] if idx < len(token_texts) else "",
                "weight": float(attn_row[idx])
            }
            for idx in top_indices
        ]
```

---

## Appendix D: Frontend Example - `AttentionFlowView.tsx`

```tsx
import React, { useRef, useEffect } from 'react';
import * as d3 from 'd3';
import { AttentionFlowLayer, HeadData } from '../types';

interface AttentionFlowViewProps {
  layers: number;
  heads: number;
  headData: HeadData[][];  // [layer][head]
  flowData: AttentionFlowLayer[];
  selectedToken: number | null;
  connectionThreshold: number;
  showCrossHead: boolean;
  onHeadClick: (layer: number, head: number) => void;
}

export const AttentionFlowView: React.FC<AttentionFlowViewProps> = ({
  layers,
  heads,
  headData,
  flowData,
  selectedToken,
  connectionThreshold,
  showCrossHead,
  onHeadClick,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  
  // Layout constants
  const LAYER_SPACING = 120;
  const HEAD_SPACING = 20;
  const NODE_RADIUS = 6;
  const MARGIN = { top: 40, right: 40, bottom: 40, left: 60 };
  
  useEffect(() => {
    if (!svgRef.current) return;
    
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    
    const width = layers * LAYER_SPACING + MARGIN.left + MARGIN.right;
    const height = heads * HEAD_SPACING + MARGIN.top + MARGIN.bottom;
    
    svg.attr('width', width).attr('height', height);
    
    const g = svg.append('g')
      .attr('transform', `translate(${MARGIN.left}, ${MARGIN.top})`);
    
    // Color scale for correctness (red -> yellow -> green)
    const colorScale = d3.scaleLinear<string>()
      .domain([0, 0.5, 1])
      .range(['#EF4444', '#EAB308', '#22C55E']);
    
    // Draw connections first (behind nodes)
    const connectionsGroup = g.append('g').attr('class', 'connections');
    
    flowData.forEach(layerFlow => {
      layerFlow.headConnections
        .filter(conn => conn.weight >= connectionThreshold)
        .filter(conn => showCrossHead || conn.sourceHead === conn.targetHead)
        .forEach(conn => {
          const x1 = layerFlow.sourceLayer * LAYER_SPACING;
          const y1 = conn.sourceHead * HEAD_SPACING;
          const x2 = layerFlow.targetLayer * LAYER_SPACING;
          const y2 = conn.targetHead * HEAD_SPACING;
          
          const isCrossHead = conn.sourceHead !== conn.targetHead;
          
          connectionsGroup.append('path')
            .attr('d', isCrossHead 
              ? bezierPath(x1, y1, x2, y2)
              : `M ${x1} ${y1} L ${x2} ${y2}`)
            .attr('stroke', isCrossHead ? '#8B5CF6' : '#94A3B8')
            .attr('stroke-width', 1 + conn.weight * 3)
            .attr('stroke-opacity', 0.3 + conn.weight * 0.5)
            .attr('fill', 'none');
        });
    });
    
    // Draw nodes for each layer/head
    for (let layer = 0; layer < layers; layer++) {
      for (let head = 0; head < heads; head++) {
        const data = headData[layer]?.[head];
        const x = layer * LAYER_SPACING;
        const y = head * HEAD_SPACING;
        const confidence = data?.probePrediction?.confidence ?? 0.5;
        
        g.append('circle')
          .attr('cx', x)
          .attr('cy', y)
          .attr('r', NODE_RADIUS)
          .attr('fill', colorScale(confidence))
          .attr('stroke', '#fff')
          .attr('stroke-width', 1)
          .attr('cursor', 'pointer')
          .on('click', () => onHeadClick(layer, head))
          .on('mouseenter', function() {
            d3.select(this)
              .transition()
              .duration(100)
              .attr('r', NODE_RADIUS * 1.5);
          })
          .on('mouseleave', function() {
            d3.select(this)
              .transition()
              .duration(100)
              .attr('r', NODE_RADIUS);
          });
      }
    }
    
    // Draw layer labels
    for (let layer = 0; layer < layers; layer++) {
      g.append('text')
        .attr('x', layer * LAYER_SPACING)
        .attr('y', -20)
        .attr('text-anchor', 'middle')
        .attr('font-size', '12px')
        .attr('fill', '#666')
        .text(`L${layer}`);
    }
    
    // Draw head labels (first few only)
    for (let head = 0; head < Math.min(heads, 5); head++) {
      g.append('text')
        .attr('x', -20)
        .attr('y', head * HEAD_SPACING + 4)
        .attr('text-anchor', 'end')
        .attr('font-size', '10px')
        .attr('fill', '#666')
        .text(`H${head}`);
    }
    
  }, [layers, heads, headData, flowData, connectionThreshold, showCrossHead]);
  
  return (
    <svg 
      ref={svgRef}
      className="attention-flow-view"
      style={{ overflow: 'visible' }}
    />
  );
};

function bezierPath(x1: number, y1: number, x2: number, y2: number): string {
  const midX = (x1 + x2) / 2;
  return `M ${x1} ${y1} C ${midX} ${y1}, ${midX} ${y2}, ${x2} ${y2}`;
}
```

---

## Next Steps

1. Review this document for completeness
2. Approve architecture decisions
3. Begin implementation following the checklist
4. Track progress in checklist

---

## Document Cross-Reference

| Need to know... | See document |
|-----------------|--------------|
| What features to build | `VISUALIZATION_DESIGN.md` |
| How the UI should look | `VISUALIZATION_DESIGN.md` (ASCII mockups) |
| User interaction flows | `VISUALIZATION_DESIGN.md` |
| Data structures/API schemas | `VISUALIZATION_DESIGN.md` (Data Structures section) |
| Module architecture | `VISUALIZATION_CODE_DESIGN.md` |
| Class/function signatures | `VISUALIZATION_CODE_DESIGN.md` |
| Implementation order | `VISUALIZATION_CODE_DESIGN.md` (Checklist) |
| Full code examples | `VISUALIZATION_CODE_DESIGN.md` (Appendices)
