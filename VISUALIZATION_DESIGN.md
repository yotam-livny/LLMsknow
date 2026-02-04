# LLMsKnow Layer Visualization Tool

> Interactive tool for exploring LLM internal representations and probe predictions.
> Based on the paper "LLMs Know More Than They Show" (arXiv:2410.02707)

---

## Quick Start

```bash
# Start both frontend and backend
cd visualization && bash run.sh

# Or start separately:
# Backend: cd visualization/backend && python -m api.app
# Frontend: cd visualization/frontend && npm run dev
```

**URLs:**
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

---

## Overview

This visualization tool provides an interactive interface for:

1. **Running inference** on questions from datasets or custom input
2. **Visualizing attention patterns** across layers and heads
3. **Logit Lens analysis** - how token predictions evolve through layers
4. **Correctness Evolution** - how the model's internal "correctness belief" changes across layers
5. **Exploring trained probes** at multiple layers (0, 5, 10, 14, 15, 16, 20, 25, 30)

---

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                      FRONTEND (React + Vite)                    │
│                        http://localhost:5173                    │
├────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌───────────────────────┐   │
│  │   Model     │  │  Dataset    │  │   Sample Browser      │   │
│  │  Selector   │  │  Selector   │  │ (pagination, search)  │   │
│  └─────────────┘  └─────────────┘  └───────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   Inference Panel                        │   │
│  │  - Custom input OR dataset sample                        │   │
│  │  - Run inference with layer + attention extraction       │   │
│  │  - Token display (clickable)                             │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                 Visualization Panel                      │   │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────────────┐   │   │
│  │  │ Attention  │ │ Logit Lens │ │ Correctness Evol.  │   │   │
│  │  │   View     │ │    View    │ │       View         │   │   │
│  │  └────────────┘ └────────────┘ └────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP/REST
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                     BACKEND (FastAPI)                          │
│                      http://localhost:8000                      │
├────────────────────────────────────────────────────────────────┤
│  Endpoints:                                                     │
│  - /api/models               List supported models              │
│  - /api/datasets             List datasets with samples         │
│  - /api/combinations         Model+dataset availability         │
│  - /api/model/load           Load model to GPU/MPS              │
│  - /api/inference            Run inference + extract layers     │
│  - /api/inference/attention  Get attention patterns             │
│  - /api/inference/logit-lens Logit lens analysis               │
│  - /api/inference/correctness-evolution  Probe across layers   │
├────────────────────────────────────────────────────────────────┤
│  Core Modules:                                                  │
│  - model_manager.py          Singleton model loader             │
│  - dataset_manager.py        CSV loading with pagination        │
│  - availability_scanner.py   Scans output/ and checkpoints/     │
│  - layer_extractor.py        Hidden state extraction            │
│  - attention_extractor.py    Attention pattern extraction       │
│  - probe_runner.py           Load and run trained probes        │
│  - exact_answer_extractor.py Extract answer tokens              │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                    Existing Codebase (src/)                    │
│  - probing_utils.py  (model loading, tokenization)             │
│  - probe.py          (probe training)                           │
│  - compute_correctness.py (answer matching)                     │
└────────────────────────────────────────────────────────────────┘
```

---

## Supported Models

| Model | Layers | Heads | Hidden Size |
|-------|--------|-------|-------------|
| Mistral-7B-Instruct-v0.2 | 32 | 32 | 4096 |
| Mistral-7B-v0.3 | 32 | 32 | 4096 |
| LLaMA-3-8B | 32 | 32 | 4096 |
| LLaMA-3-8B-Instruct | 32 | 32 | 4096 |

---

## Supported Datasets

| Dataset | Category | Question Column | Answer Column |
|---------|----------|-----------------|---------------|
| Movie QA (Train/Test) | Factual | `Question` | `Answer` |
| Answerable Math | Math | `question` | `answer` |
| MNLI (Train/Validation) | NLI | `Question` | `Answer` |
| Winogrande (Train/Test) | Commonsense | `sentence` | `answer` |
| Winobias (Dev/Test) | Bias | `sentence` | `answer` |
| Natural Questions | Factual | `question` | `answer` |

---

## Visualization Modes

### 1. Attention View

Shows where a selected token "looks" in the sequence.

```
┌────────────────────────────────────────────────────────────┐
│  🔍 Attention Pattern                                       │
│                                                             │
│  Source token: [Dropdown: select token]                     │
│  Layer: [Slider 0-31]   Head: [Dropdown: avg / 0-31]       │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  0: "Who"        ████████████████░░░░░░  65%          │  │
│  │  1: " directed"  ████████░░░░░░░░░░░░░░  32%          │  │
│  │  2: " the"       ████░░░░░░░░░░░░░░░░░░  15%          │  │
│  │  3: " movie"     ██████████████████████░  89%          │  │
│  │  4: " Titanic"   ████████████████████████  95%  ← max  │  │
│  │  5: "?"          ██░░░░░░░░░░░░░░░░░░░░░  8%           │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  [Click a bar to select that token as source]              │
└────────────────────────────────────────────────────────────┘
```

**Features:**
- Select source token (which token's attention to visualize)
- Layer slider (0-31)
- Head selector (individual head or average across all heads)
- Bar chart showing attention weights to all tokens
- Click bars to navigate to different source tokens

---

### 2. Logit Lens View

Shows how token predictions evolve through layers.

```
┌────────────────────────────────────────────────────────────┐
│  🔬 Logit Lens                                              │
│                                                             │
│  Predicting: "Cameron" (position 8)                         │
│  Using hidden state at: "James" (position 7)               │
│                                                             │
│  ┌─────────────────────┐  ┌─────────────────────┐          │
│  │ Layer 0             │  │ Layer 15            │          │
│  │ Target: #2451       │  │ Target: #8          │          │
│  │ ─────────────────── │  │ ─────────────────── │          │
│  │ #1 "the"    12.3%   │  │ #1 "Cameron" 67.2%  │          │
│  │ #2 "and"     8.1%   │  │ #2 "Smith"   15.4%  │          │
│  │ #3 "James"   5.2%   │  │ #3 "Brown"    8.9%  │          │
│  │ #4 "a"       4.8%   │  │ #4 "Jones"    4.1%  │          │
│  │ #5 "is"      3.2%   │  │ #5 "the"      2.3%  │          │
│  └─────────────────────┘  └─────────────────────┘          │
│                                                             │
│  (... more layers ...)                                      │
│                                                             │
│  💡 Key insight: The actual token "Cameron" starts as       │
│     #2451 at layer 0 and rises to #1 by layer 15           │
└────────────────────────────────────────────────────────────┘
```

**Features:**
- Shows top-K predictions at each layer
- Highlights the actual generated token
- Shows target token rank at each layer (watch it rise!)
- Reveals when the model "decides" on its answer

---

### 3. Correctness Evolution View

Shows how the model's internal "belief" about correctness evolves.

```
┌────────────────────────────────────────────────────────────┐
│  📈 Correctness Evolution                                   │
│                                                             │
│  🎯 Exact Answer Tokens                                     │
│  "James <mark>Cameron</mark> directed the movie."          │
│  Method: LLM extraction | Token positions: 6, 7            │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Expected: James Cameron                               │  │
│  │ Ground Truth: ✓ Correct                              │  │
│  │ Before Generation: ✓ 72% at L15, tok 5               │  │
│  │ After Generation: ✓ 89% at L15, tok 12               │  │
│  │ ✓ Model's final self-assessment matches reality      │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  📊 Confidence Across Layers                                │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  100% ─                              ●────●────●     │  │
│  │       │                      ●──●────                │  │
│  │   50% ├ ─ ─ ─ ─ ─ ─ ─ ─●────                        │  │
│  │       │          ●──●──                              │  │
│  │    0% └──────────────────────────────────────────▶   │  │
│  │        L0   L5  L10  L14 L15 L16 L20 L25 L30         │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  💡 First confident layer: L16                             │
│  Peak: 92% at L30                                          │
└────────────────────────────────────────────────────────────┘
```

**Features:**
- Extracts "exact answer" tokens from generated response
- Runs probes at all available layers (currently 9 layers)
- D3 line chart showing P(correct) evolution
- Highlights first confident layer (>70%)
- Compares probe prediction vs ground truth
- Shows calibration (does model know what it knows?)

---

## User Flow

### Complete Workflow

```
1. SELECT MODEL
   └─> Dropdown shows available models with ready dataset counts
       e.g., "Mistral 7B Instruct (1 ready, 10 partial)"

2. SELECT DATASET  
   └─> Dropdown shows datasets with status
       ✓ READY = has probe + answers
       ⚠ PARTIAL = has answers but no probe
       ○ NOT_PROCESSED = raw CSV only

3. BROWSE SAMPLES
   └─> Paginated table with search
       Click row to select sample

4. RUN INFERENCE
   └─> Click "▶ Run Inference"
       - Model loads if not cached
       - Generates answer
       - Extracts layer representations
       - Extracts attention patterns
       - Runs probes at all available layers

5. EXPLORE VISUALIZATIONS
   └─> Switch between tabs:
       - Attention (layer/head attention patterns)
       - Logit Lens (token prediction evolution)
       - Correctness (probe predictions across layers)
       
       Click tokens to analyze different positions
```

---

## Trained Probes

Currently trained probes for **Mistral-7B-Instruct + Movie QA**:

| Layer | Status | Token Position |
|-------|--------|----------------|
| 0 | ✅ | exact_answer_last_token |
| 5 | ✅ | exact_answer_last_token |
| 10 | ✅ | exact_answer_last_token |
| 14 | ✅ | exact_answer_last_token |
| 15 | ✅ | exact_answer_last_token |
| 16 | ✅ | exact_answer_last_token |
| 20 | ✅ | exact_answer_last_token |
| 25 | ✅ | exact_answer_last_token |
| 30 | ✅ | exact_answer_last_token |

**To train more probes:**
```bash
cd src && export WANDB_MODE=offline
python3 probe.py --model mistralai/Mistral-7B-Instruct-v0.2 \
  --dataset movies --layer 12 --token exact_answer_last_token \
  --probe_at mlp --seeds 42 --save_clf
```

---

## API Endpoints

### Models & Datasets

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/models` | GET | List supported models with availability |
| `/api/models/{id}/combinations` | GET | Get dataset status for model |
| `/api/datasets` | GET | List all datasets |
| `/api/datasets/{id}/samples` | GET | Get paginated samples (supports `?search=`) |
| `/api/combinations` | GET | All model+dataset combinations |

### Model Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/model/status` | GET | Current model status |
| `/api/model/load` | POST | Load model to GPU/MPS |
| `/api/model/unload` | POST | Unload model |

### Inference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/inference` | POST | Run inference with layer/attention extraction |
| `/api/inference/layers` | GET | Get layer representations from last inference |
| `/api/inference/attention` | GET | Get attention patterns from last inference |
| `/api/inference/logit-lens` | POST | Logit lens analysis for token position |
| `/api/inference/correctness-evolution` | POST | Probe predictions across layers |

---

## File Structure

```
visualization/
├── backend/
│   ├── api/
│   │   ├── app.py           # FastAPI routes
│   │   └── schemas.py       # Pydantic models
│   ├── core/
│   │   ├── model_manager.py        # Singleton model loader
│   │   ├── dataset_manager.py      # CSV pagination/search
│   │   ├── availability_scanner.py # Scan output/ and checkpoints/
│   │   ├── layer_extractor.py      # Hidden state extraction
│   │   ├── attention_extractor.py  # Attention pattern extraction
│   │   ├── probe_runner.py         # Load and run probes
│   │   ├── exact_answer_extractor.py # Extract answer tokens
│   │   └── correctness.py          # Correctness computation
│   ├── utils/
│   │   └── logging.py       # Centralized logging
│   ├── config.py            # Configuration constants
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── App.tsx          # Main application
│   │   ├── store/
│   │   │   └── useStore.ts  # Zustand state management
│   │   ├── api/
│   │   │   └── client.ts    # Axios API client
│   │   └── components/
│   │       ├── ModelSelector.tsx
│   │       ├── DatasetSelector.tsx
│   │       ├── SampleBrowser.tsx
│   │       ├── CombinationDetails.tsx
│   │       ├── InferencePanel.tsx
│   │       ├── TokenDisplay.tsx
│   │       ├── VisualizationPanel.tsx
│   │       ├── AttentionView.tsx        # D3 attention chart
│   │       ├── LogitLensView.tsx        # Logit lens analysis
│   │       └── CorrectnessEvolutionView.tsx  # D3 correctness chart
│   ├── package.json
│   └── vite.config.ts
│
└── run.sh                   # Start both servers
```

---

## Key Insights from the Paper

The visualization tool helps explore key findings from "LLMs Know More Than They Show":

1. **Internal Correctness Encoding**: LLMs encode whether their answer is correct in their hidden states, even when they express uncertainty externally.

2. **Layer-wise Evolution**: The "correctness signal" typically emerges in middle layers and strengthens through later layers.

3. **Exact Answer Tokens**: Truthfulness information is concentrated in the last tokens of the exact answer (hence `exact_answer_last_token` probe position).

4. **Probe Predictions vs Ground Truth**: The visualization shows whether the model "knows" it's right or wrong, and compares to actual correctness.

---

## Color Scheme

| Element | Color | Usage |
|---------|-------|-------|
| Background | `#0f1419` | Dark theme base |
| Panel | `#1a2332` | Card backgrounds |
| Border | `#38444d` | Panel borders |
| Accent | `#1d9bf0` | Buttons, highlights |
| Correct | `#4CAF50` | Green for correct predictions |
| Incorrect | `#ff4444` | Red for incorrect predictions |
| Warning | `#fbbf24` | Yellow for highlights |
| Text | `#fff` / `#aaa` | Primary/secondary text |

---

## Requirements

### Backend
```
fastapi>=0.100.0
uvicorn>=0.22.0
pandas>=2.0.0
numpy>=1.24.0
torch>=2.0.0
transformers>=4.30.0
scikit-learn>=1.2.0
```

### Frontend
```
react ^18
zustand (state management)
axios (HTTP client)
d3 (visualizations)
vite (build tool)
```

---

## Troubleshooting

### Backend won't start
```bash
# Check if port 8000 is in use
lsof -i :8000 | grep LISTEN

# Kill existing process
lsof -ti:8000 | xargs kill -9
```

### CORS errors
The backend allows origins on ports 5173-5176. If frontend runs on a different port, add it to `config.py` `CORS_ORIGINS`.

### Model loading fails
- Check GPU/MPS memory (Mistral 7B needs ~14GB)
- Try `use_quantization=true` in load request
- Check HuggingFace token for gated models

### No probe predictions
- Ensure probes are trained: `ls checkpoints/clf_*.pkl`
- Check dataset_id matches output_id in probe filenames
- Run probe training if needed (see "Trained Probes" section)

---

## Future Enhancements

- [ ] Comparison mode: side-by-side analysis of two questions
- [ ] Batch analysis: patterns across multiple samples
- [ ] Export visualizations as images/PDFs
- [ ] 3D layer visualization
- [ ] Attention head importance ranking
- [ ] Neuron-level analysis
