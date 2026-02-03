# LLMsKnow Layer Visualization Tool - Design Document

## Overview

An interactive, explorative visualization tool for understanding how LLMs process information through their layers, with focus on:
- **Token generation process**: How each output token is "created" across layers
- **Correctness detection**: When/where the model "knows" it's right or wrong
- **Attention head flow**: Visualize data flow through attention heads across layers
- **Model-agnostic**: Works with any supported model (Mistral, LLaMA, etc.)

---

## Supported Models

The tool automatically adapts to any model in the codebase:

| Model | Layers | Hidden Size | Attention Heads |
|-------|--------|-------------|-----------------|
| Mistral-7B-Instruct-v0.2 | 32 | 4096 | 32 |
| Mistral-7B-v0.3 | 32 | 4096 | 32 |
| Meta-Llama-3-8B | 32 | 4096 | 32 |
| Meta-Llama-3-8B-Instruct | 32 | 4096 | 32 |

Adding new models requires only updating `probing_utils.py` - the visualization adapts automatically.

---

## Visualization Modes

The tool offers **three complementary visualization modes**:

### Mode 1: Layer Overview (High-Level)
- Left → Right: Layer depth (32 layers)
- Each layer as a single node
- Color indicates correctness confidence
- Click to drill down

### Mode 2: Attention Head Flow (Recommended - TensorFlow Playground Style)
- Left → Right: Layer depth
- Top → Bottom: Attention heads (32 heads per layer)
- Connections show attention weight flow between layers
- Line width = attention strength
- Color = correctness at that layer/head
- Supports drill-down to specific heads

### Mode 3: Dimension Flow (Advanced)
- PCA-reduced view of top-K important dimensions
- Shows how information transforms through layers
- For advanced users wanting dimension-level insights

---

## UI Layout

### Main Interface with Mode Selector

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  🧠 LLMsKnow Layer Explorer                    [Model: Mistral-7B ▼]        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  INPUT                                                               │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐│   │
│  │  │ Who directed the movie Titanic?                            [▶]  ││   │
│  │  └─────────────────────────────────────────────────────────────────┘│   │
│  │                                                                      │   │
│  │  OUTPUT                                                              │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐│   │
│  │  │ [James]  [Cameron]  [directed]  [Titanic]  [.]                  ││   │
│  │  │    ↑         ↑                                                  ││   │
│  │  │  click tokens to explore their generation process               ││   │
│  │  └─────────────────────────────────────────────────────────────────┘│   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  VIEW MODE: [◉ Layer Overview] [○ Attention Flow] [○ Dimension Flow]       │
├─────────────────────────────────────────────────────────────────────────────┤
```

### View Mode 1: Layer Overview

```
│  LAYER FLOW (Left → Right = Layer Depth)                                   │
│  ════════════════════════════════════════════════════════════════════════  │
│                                                                             │
│   L0        L4        L8        L12       L16       L20       L24      L31  │
│   ┌─┐       ┌─┐       ┌─┐       ┌─┐       ┌─┐       ┌─┐       ┌─┐      ┌─┐  │
│   │ │──────▶│░│──────▶│▒│──────▶│▓│──────▶│▓│──────▶│█│──────▶│█│─────▶│█│  │
│   └─┘       └─┘       └─┘       └─┘       └─┘       └─┘       └─┘      └─┘  │
│    │         │         │         │         │         │         │        │   │
│   ───────────────────────────────────────────────────────────────────────   │
│   🔴 0.12   🔴 0.23   🟡 0.45   🟡 0.58   🟢 0.72   🟢 0.85   🟢 0.91  🟢 0.96│
│                                                                             │
│   Click any layer to drill down into attention heads →                      │
```

### View Mode 2: Attention Head Flow (TensorFlow Playground Style)

```
│  ATTENTION HEAD FLOW (Left → Right = Layers, Top → Bottom = Heads)         │
│  ════════════════════════════════════════════════════════════════════════  │
│                                                                             │
│      Layer 0      Layer 8      Layer 16     Layer 24     Layer 31          │
│         │            │            │            │            │               │
│    ┌────┴────┐  ┌────┴────┐  ┌────┴────┐  ┌────┴────┐  ┌────┴────┐         │
│    │         │  │         │  │         │  │         │  │         │         │
│   ┌─┐       ┌─┐       ┌─┐       ┌─┐       ┌─┐                              │
│ H0│🔴│══════│🔴│══════│🟡│══════│🟢│══════│🟢│  Head 0                     │
│   └─┘   ╲   └─┘   ╲   └─┘   ╲   └─┘   ╲   └─┘                              │
│          ╲         ╲         ╲         ╲                                    │
│   ┌─┐     ╲ ┌─┐     ╲ ┌─┐     ╲ ┌─┐     ╲ ┌─┐                              │
│ H1│🔴│═════╳│🔴│═════╳│🟡│═════╳│🟢│═════╳│🟢│  Head 1                     │
│   └─┘     ╱ └─┘     ╱ └─┘     ╱ └─┘     ╱ └─┘                              │
│          ╱         ╱         ╱         ╱                                    │
│   ┌─┐   ╱   ┌─┐   ╱   ┌─┐   ╱   ┌─┐   ╱   ┌─┐                              │
│ H2│🔴│══════│🟡│══════│🟡│══════│🟢│══════│🟢│  Head 2                     │
│   └─┘       └─┘       └─┘       └─┘       └─┘                              │
│    :         :         :         :         :                               │
│   ┌─┐       ┌─┐       ┌─┐       ┌─┐       ┌─┐                              │
│H31│🔴│══════│🔴│══════│🟢│══════│🟢│══════│🟢│  Head 31                    │
│   └─┘       └─┘       └─┘       └─┘       └─┘                              │
│                                                                             │
│  ═══ Thick line = High attention weight    ─── Thin line = Low weight      │
│  Line color transitions: 🔴→🟡→🟢 based on correctness flow                 │
│                                                                             │
│  [Slider: Show top N connections] ══════════●══ 50%                        │
│  [□ Show cross-head connections]  [□ Animate flow]  [□ Highlight path]     │
```

### View Mode 3: Dimension Flow (PCA-Reduced)

```
│  DIMENSION FLOW (Top-K Principal Components)                                │
│  ════════════════════════════════════════════════════════════════════════  │
│                                                                             │
│         Layer 0    Layer 8    Layer 16   Layer 24   Layer 31               │
│                                                                             │
│    PC1    ●─────────●═════════●══════════●══════════●                      │
│                  ╲       ╲         ╲          ╲                             │
│    PC2    ●═════════●─────────●══════════●══════════●                      │
│                  ╱       ╱         ╲          ╱                             │
│    PC3    ●─────────●═════════●══════════●══════════●                      │
│                  ╲       ╲         ╱          ╲                             │
│    PC4    ●═════════●─────────●──────────●══════════●                      │
│                                                                             │
│    ...                                                                      │
│                                                                             │
│    PC10   ●─────────●─────────●──────────●──────────●                      │
│                                                                             │
│  ═══ High importance    ─── Low importance                                  │
│  [Slider: Number of PCs] ════●════════════ 10                              │
```

### Insight Panel (Context-Aware)

```
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INSIGHT PANEL                                                              │
│  ════════════════════════════════════════════════════════════════════════  │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  📍 Selected: Layer 16, Head 12, Token "Cameron"                    │   │
│  │                                                                      │   │
│  │  ┌──────────────────┐  ┌──────────────────────────────────────────┐│   │
│  │  │ Head Statistics  │  │ Attention Pattern (Heatmap)              ││   │
│  │  │                  │  │ ┌────────────────────────────────────┐   ││   │
│  │  │ Avg Attn: 0.342  │  │ │ Who   dir   the   mov   Tit   ?    │   ││   │
│  │  │ Max Attn: 0.891  │  │ │ Jam ░░░░░ ░░░░░ ░░░░░ ░░░░░ ▓▓▓▓▓ │   ││   │
│  │  │ Entropy: 1.23    │  │ │ Cam ░░░░░ ░░░░░ ░░░░░ ▓▓▓▓▓ █████ │   ││   │
│  │  │ Sparsity: 0.78   │  │ │                                    │   ││   │
│  │  │                  │  │ └────────────────────────────────────┘   ││   │
│  │  │ Correctness:     │  │ ░ Low attention  ▓ Medium  █ High        ││   │
│  │  │   🟢 0.85        │  │                                          ││   │
│  │  └──────────────────┘  └──────────────────────────────────────────┘│   │
│  │                                                                      │   │
│  │  ┌────────────────────────────────────────────────────────────────┐│   │
│  │  │ Head Importance Ranking (for selected token)                   ││   │
│  │  │                                                                ││   │
│  │  │ H12 ████████████████████████████████████████████████ 0.89      ││   │
│  │  │ H8  ██████████████████████████████████████           0.76      ││   │
│  │  │ H15 ████████████████████████████████                 0.71      ││   │
│  │  │ H3  ██████████████████████                           0.52      ││   │
│  │  │ H27 ████████████████                                 0.41      ││   │
│  │  └────────────────────────────────────────────────────────────────┘│   │
│  │                                                                      │   │
│  │  ┌────────────────────────────────────────────────────────────────┐│   │
│  │  │ Correctness Evolution (All Heads at Layer 16)                  ││   │
│  │  │                                                                ││   │
│  │  │ Correct ▲     ●  ●     ●●●●●   ●●●●●●●●●●●●●●●●               ││   │
│  │  │         │   ●●●●●●●●●●●                                       ││   │
│  │  │         │ ●●                                                   ││   │
│  │  │  Wrong  └──────────────────────────────────────────▶          ││   │
│  │  │           H0  H4  H8  H12 H16 H20 H24 H28 H31                 ││   │
│  │  └────────────────────────────────────────────────────────────────┘│   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Visual Elements

### 1. Layer Nodes
```
   Normal          Hovered         Selected        Correct         Wrong
   ┌───┐           ┌───┐           ┏━━━┓           ┌───┐           ┌───┐
   │ ░ │           │▒▒▒│           ┃███┃           │🟢 │           │🔴 │
   └───┘           └───┘           ┗━━━┛           └───┘           └───┘
   
   Color intensity = activation magnitude
   Border = selection state
   Inner icon = correctness prediction
```

### 2. Attention Head Nodes (New)
```
   Small (default)     Medium (hovered)      Large (selected)
        ●                    ◉                    ◉
       5px                  8px                  12px
   
   Color: Correctness gradient (🔴 → 🟡 → 🟢)
   Opacity: Attention weight strength (0.2 - 1.0)
   Border: White glow when selected
```

### 3. Connection Lines

```
   Standard Flow (same head across layers):
   ═══════════════════════════════════════════════════════════════
   Thickness: 1-5px based on attention weight
   Color: Gradient from source correctness to target correctness
   
   Cross-Head Flow (attention redistributes to different heads):
   ─────────────────────────────╲
                                 ╲
                                  ╲─────────────────────────────────
   Thickness: 1-3px (thinner than same-head)
   Color: Lighter, with gradient
   Style: Slightly curved bezier
   
   Highlighted Path (when token selected):
   ████████████████████████████████████████████████████████████████
   Thickness: 3-7px
   Color: Bright blue with glow effect
   Animation: Pulse/flow animation
```

### 4. Token Chips (Output)
```
   Unselected              Selected                 Correct              Wrong
   ┌─────────┐            ┏━━━━━━━━━┓             ┌─────────┐          ┌─────────┐
   │ Cameron │            ┃ Cameron ┃             │✓Cameron │          │✗ wrong  │
   └─────────┘            ┗━━━━━━━━━┛             └─────────┘          └─────────┘
   gray bg                 blue border             green tint           red tint
```

### 5. Correctness Gradient Bar
```
   ┌────────────────────────────────────────────────────────────────────┐
   │░░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒▒▒▒▓▓▓▓▓▓▓▓▓▓▓▓████████████████████████████│
   └────────────────────────────────────────────────────────────────────┘
   🔴 0.0            🟡 0.5                                    🟢 1.0
   
   Smooth gradient: Red → Yellow → Green
   Marker shows current layer's prediction
```

### 6. Attention Heatmap (in Insight Panel)
```
   ┌────────────────────────────────────────────────────┐
   │         Input Tokens (columns)                     │
   │        Who  dir  the  mov  Tit  ?                 │
   │ Output ┌───────────────────────────────┐          │
   │ Tokens │░░░ ░░░ ░░░ ░░░ ███ ░░░│ James  │          │
   │ (rows) │░░░ ░░░ ░░░ ▓▓▓ ███ ░░░│ Cameron│          │
   │        └───────────────────────────────┘          │
   └────────────────────────────────────────────────────┘
   ░ = 0.0-0.3    ▓ = 0.3-0.7    █ = 0.7-1.0
```

---

## Interaction Flows

### Flow 1: Run New Question
```
User types question → Clicks [▶] Run
                           ↓
              Backend loads model (if needed)
                           ↓
              Generate answer token by token
                           ↓
              Extract layer representations
                           ↓
              Extract attention patterns (NEW)
                           ↓
              Run probe at each layer
                           ↓
              Return all data to frontend
                           ↓
              Animate layer flow visualization
                           ↓
              Display tokens with correctness colors
```

### Flow 2: Explore Token Generation
```
User clicks output token (e.g., "Cameron")
                           ↓
              Highlight token's "path" through layers
                           ↓
              Show attention connections for this token
                           ↓
              In Attention Head Flow mode:
                - Highlight which heads contributed most
                - Show cross-head information flow
                - Animate the path through layers
                           ↓
              Update insight panel with:
                - Activation statistics
                - Top contributing attention heads
                - Layer-by-layer probe predictions
                - Attention heatmap for selected head
```

### Flow 3: Explore Layer (in Layer Overview mode)
```
User clicks layer node (e.g., Layer 16)
                           ↓
              Highlight layer
                           ↓
              Show option to drill down to Attention Head view
                           ↓
              Update insight panel with:
                - Layer-wide statistics
                - All 32 heads' correctness predictions
                - Comparison to previous/next layers
```

### Flow 4: Explore Attention Head (in Attention Flow mode)
```
User clicks attention head node (e.g., Layer 16, Head 12)
                           ↓
              Highlight head and its connections
                           ↓
              Show attention pattern heatmap
                           ↓
              Update insight panel with:
                - Head-specific attention statistics
                - Which input tokens it attends to
                - Correctness contribution of this head
                - Connection to other heads in adjacent layers
```

### Flow 5: Adjust Connection Visibility
```
User moves "Show top N connections" slider
                           ↓
              Filter connections by weight threshold
                           ↓
              Smoothly animate visible connections
                           ↓
              Show only the strongest attention paths

User toggles "Show cross-head connections"
                           ↓
              Show/hide diagonal connections between different heads
                           ↓
              Useful for seeing information redistribution
```

### Flow 6: Switch View Mode
```
User clicks different view mode radio button
                           ↓
              Animate transition between views
                           ↓
              Maintain selected token/layer context
                           ↓
              Update insight panel for new view
```

---

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FRONTEND (React + D3)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌────────────┐  │
│   │ Input       │    │ View Mode   │    │ Token       │    │ Insight    │  │
│   │ Component   │───▶│ Visualizers │◀──▶│ Display     │───▶│ Panel      │  │
│   └─────────────┘    └──────┬──────┘    └─────────────┘    └────────────┘  │
│                             │                                               │
│                    ┌────────┴────────┐                                     │
│                    │                  │                                     │
│             ┌──────▼──────┐    ┌──────▼──────┐    ┌──────────────┐         │
│             │ LayerFlow   │    │ AttentionFlow│   │ DimensionFlow│         │
│             │ (D3)        │    │ (D3) - NEW   │   │ (D3)         │         │
│             └─────────────┘    └──────────────┘   └──────────────┘         │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                        State Management                              │  │
│   │   • currentModel       • selectedToken     • viewMode                │  │
│   │   • question/answer    • selectedLayer     • selectedHead (NEW)      │  │
│   │   • layerData[]        • attentionData[]   • connectionThreshold     │  │
│   │   • insightData        • showCrossHead     • animateFlow             │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                      │                                      │
│                                      ▼                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                              API CLIENT                                     │
│                         WebSocket / REST API                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           BACKEND (FastAPI + Python)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────┐                                                       │
│   │ /api/models     │ → List available models                              │
│   ├─────────────────┤                                                       │
│   │ /api/generate   │ → Run model, return answer + layer data + attention  │
│   ├─────────────────┤                                                       │
│   │ /api/probe      │ → Get probe predictions for all layers               │
│   ├─────────────────┤                                                       │
│   │ /api/layer/{n}  │ → Get detailed data for specific layer               │
│   ├─────────────────┤                                                       │
│   │ /api/attention  │ → Get attention patterns (NEW)                       │
│   │    /{layer}/{head}│                                                     │
│   ├─────────────────┤                                                       │
│   │ /api/attention/ │ → Get aggregated head-to-head attention flow (NEW)   │
│   │    flow         │                                                       │
│   └─────────────────┘                                                       │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                        Model Manager                                 │   │
│   │   • load_model(name)           - Load model into memory             │   │
│   │   • generate(prompt)           - Generate + extract representations │   │
│   │   • get_probe_preds()          - Run probe classifier at each layer │   │
│   │   • get_layer_data(n)          - Get activations for layer n        │   │
│   │   • get_attention_patterns()   - Extract all attention weights (NEW)│   │
│   │   • get_head_flow()            - Compute head-to-head flow (NEW)    │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                      │
│                                      ▼                                      │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    Existing Codebase Integration                     │   │
│   │                                                                      │   │
│   │   probing_utils.py  →  load_model_and_validate_gpu()                │   │
│   │                    →  extract_internal_reps_*()                     │   │
│   │   probe.py         →  trained probe classifiers                     │   │
│   │   compute_*.py     →  correctness calculation                       │   │
│   │   attention_extractor.py  → NEW FILE for attention extraction       │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Structures

### LayerData (per layer)
```json
{
  "layer_idx": 16,
  "layer_name": "model.layers.16.mlp",
  "activations": {
    "mean": 0.342,
    "std": 0.156,
    "max": 2.341,
    "min": -1.892
  },
  "probe_prediction": {
    "confidence": 0.72,
    "prediction": "correct",
    "probabilities": [0.28, 0.72]
  },
  "top_neurons": [
    {"idx": 1542, "activation": 2.341},
    {"idx": 892, "activation": 2.156}
  ]
}
```

### AttentionHeadData (NEW - per layer per head)
```json
{
  "layer_idx": 16,
  "head_idx": 12,
  "attention_pattern": {
    "shape": [seq_len, seq_len],
    "data": [[0.1, 0.2, ...], ...]  // attention weights matrix
  },
  "statistics": {
    "avg_attention": 0.342,
    "max_attention": 0.891,
    "entropy": 1.23,
    "sparsity": 0.78
  },
  "probe_prediction": {
    "confidence": 0.85,
    "prediction": "correct"
  },
  "top_attended_tokens": [
    {"token_idx": 5, "token_text": "Titanic", "weight": 0.891},
    {"token_idx": 4, "token_text": "movie", "weight": 0.456}
  ]
}
```

### AttentionFlowData (NEW - cross-layer head connections)
```json
{
  "source_layer": 15,
  "target_layer": 16,
  "head_connections": [
    {
      "source_head": 12,
      "target_head": 12,
      "weight": 0.89,
      "correctness_delta": 0.05
    },
    {
      "source_head": 12,
      "target_head": 8,
      "weight": 0.34,
      "correctness_delta": 0.02
    }
  ]
}
```

### TokenData (per output token - extended)
```json
{
  "token_idx": 0,
  "token_text": "James",
  "token_id": 5765,
  "is_correct": true,
  "generation_probability": 0.89,
  "layer_journey": [
    {"layer": 0, "activation_norm": 0.12},
    {"layer": 8, "activation_norm": 0.34},
    {"layer": 16, "activation_norm": 0.67},
    {"layer": 31, "activation_norm": 0.95}
  ],
  "attention_path": [
    {
      "layer": 0,
      "dominant_heads": [{"head": 5, "weight": 0.45}, {"head": 12, "weight": 0.32}]
    },
    {
      "layer": 8,
      "dominant_heads": [{"head": 12, "weight": 0.67}, {"head": 3, "weight": 0.21}]
    }
  ]
}
```

### SessionData (per question - extended)
```json
{
  "session_id": "abc123",
  "model": "mistralai/Mistral-7B-Instruct-v0.2",
  "model_config": {
    "num_layers": 32,
    "num_heads": 32,
    "hidden_size": 4096
  },
  "question": "Who directed Titanic?",
  "answer": "James Cameron",
  "tokens": [TokenData, ...],
  "layers": [LayerData, ...],
  "attention_heads": [[AttentionHeadData, ...], ...],  // [layer][head]
  "attention_flow": [AttentionFlowData, ...],         // between consecutive layers
  "overall_correctness": 0.96,
  "ground_truth": "James Cameron",
  "is_correct": true
}
```

### ViewState (NEW - frontend state)
```json
{
  "viewMode": "attention_flow",  // "layer_overview" | "attention_flow" | "dimension_flow"
  "selectedToken": 1,
  "selectedLayer": 16,
  "selectedHead": 12,
  "connectionThreshold": 0.3,   // only show connections above this weight
  "showCrossHeadConnections": true,
  "animateFlow": false,
  "highlightedPath": [
    {"layer": 0, "head": 5},
    {"layer": 8, "head": 12},
    {"layer": 16, "head": 12},
    {"layer": 24, "head": 12},
    {"layer": 31, "head": 12}
  ]
}
```

---

## Color Scheme

### Primary Palette
```
Background:       #FAFAFA (light gray)
Card Background:  #FFFFFF (white)
Text Primary:     #1A1A1A (near black)
Text Secondary:   #666666 (gray)
Border:           #E0E0E0 (light gray)
```

### Semantic Colors
```
Correct (high):   #22C55E (green)
Correct (mid):    #84CC16 (lime)
Neutral:          #EAB308 (yellow)
Wrong (mid):      #F97316 (orange)
Wrong (high):     #EF4444 (red)
```

### Gradient for Correctness
```css
background: linear-gradient(
  to right,
  #EF4444 0%,      /* red */
  #F97316 25%,     /* orange */
  #EAB308 50%,     /* yellow */
  #84CC16 75%,     /* lime */
  #22C55E 100%     /* green */
);
```

### Layer Node Colors (by activation intensity)
```
Low activation:   #E5E7EB (gray-200)
Medium:           #93C5FD (blue-300)
High:             #3B82F6 (blue-500)
Very high:        #1D4ED8 (blue-700)
```

### Attention Flow Colors (NEW)
```
Connection base:          #94A3B8 (gray-400)
Connection highlighted:   #3B82F6 (blue-500)
Cross-head connection:    #8B5CF6 (purple-500)
Path highlight:           #0EA5E9 (cyan-500)
Path glow:               rgba(14, 165, 233, 0.3)
```

### Head Node Colors (NEW - by correctness)
```css
/* Computed from correctness value 0-1 */
function getHeadColor(correctness) {
  if (correctness < 0.33) return '#EF4444';  // red
  if (correctness < 0.66) return '#EAB308';  // yellow  
  return '#22C55E';  // green
}

/* Opacity based on attention weight */
opacity: 0.2 + (attentionWeight * 0.8);  // range: 0.2 - 1.0
```

---

## File Structure

```
visualization/
├── backend/
│   ├── __init__.py
│   ├── app.py                    # FastAPI application
│   ├── model_manager.py          # Model loading & inference
│   ├── layer_extractor.py        # Layer representation extraction
│   ├── attention_extractor.py    # NEW: Attention pattern extraction
│   ├── attention_flow.py         # NEW: Head-to-head flow computation
│   ├── probe_runner.py           # Probe predictions
│   └── dimension_reducer.py      # NEW: PCA for dimension flow view
│
├── frontend/
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/
│   │   │   ├── InputPanel.tsx
│   │   │   ├── ViewModeSelector.tsx       # NEW: Radio buttons for view modes
│   │   │   ├── visualizations/
│   │   │   │   ├── LayerFlow.tsx          # D3 layer overview
│   │   │   │   ├── AttentionFlow.tsx      # NEW: D3 attention head grid
│   │   │   │   ├── DimensionFlow.tsx      # NEW: D3 PCA dimension view
│   │   │   │   ├── ConnectionLines.tsx    # NEW: Shared connection rendering
│   │   │   │   └── FlowAnimator.tsx       # NEW: Animation controller
│   │   │   ├── TokenDisplay.tsx
│   │   │   ├── InsightPanel.tsx
│   │   │   ├── AttentionHeatmap.tsx       # NEW: Attention matrix heatmap
│   │   │   ├── HeadRanking.tsx            # NEW: Head importance bar chart
│   │   │   └── CorrectnessBar.tsx
│   │   ├── hooks/
│   │   │   ├── useLayerData.ts
│   │   │   ├── useAttentionData.ts        # NEW
│   │   │   └── useViewState.ts            # NEW
│   │   ├── api/
│   │   │   └── client.ts
│   │   ├── utils/
│   │   │   ├── colorScale.ts              # NEW: Color computation utilities
│   │   │   └── pathHighlight.ts           # NEW: Path tracing utilities
│   │   └── styles/
│   │       └── globals.css
│   ├── package.json
│   └── tsconfig.json
│
├── requirements.txt
├── run.py                        # Single command to start both
└── README.md
```

---

## Implementation Phases

### Phase 1: Core Backend (3-4 hours)
- [ ] FastAPI server setup
- [ ] Model loading endpoint
- [ ] Generation + layer extraction endpoint
- [ ] **Attention extraction endpoint (NEW)**
- [ ] Probe prediction endpoint
- [ ] Integration with existing `probing_utils.py`

### Phase 2: Attention Extraction (2-3 hours) - NEW
- [ ] Create `attention_extractor.py`
- [ ] Extract attention weights from model forward pass
- [ ] Compute head-level statistics
- [ ] Create `attention_flow.py` for cross-layer flow
- [ ] Optimize for memory (don't store full seq×seq for long sequences)

### Phase 3: Basic Frontend (3-4 hours)
- [ ] React app scaffold
- [ ] Input component (question box, model selector)
- [ ] **View mode selector (NEW)**
- [ ] Token display (clickable output tokens)
- [ ] Basic layer flow (static boxes)
- [ ] API integration

### Phase 4: Layer Overview Visualization (2-3 hours)
- [ ] Layer node visualization with D3
- [ ] Connection lines with weight-based thickness
- [ ] Correctness gradient bar
- [ ] Click to expand/drill-down hint

### Phase 5: Attention Head Flow Visualization (4-5 hours) - NEW
- [ ] Grid layout: layers (x) × heads (y)
- [ ] Head nodes with correctness coloring
- [ ] Same-head connections (horizontal)
- [ ] Cross-head connections (diagonal bezier curves)
- [ ] Connection weight → line thickness
- [ ] Connection threshold slider
- [ ] Toggle for cross-head visibility
- [ ] Hover effects (highlight connected nodes)

### Phase 6: Path Highlighting & Animation (2-3 hours) - NEW
- [ ] Token selection highlights attention path
- [ ] Compute dominant path through layers
- [ ] Animate flow along path (optional toggle)
- [ ] Glow effect on highlighted connections

### Phase 7: Insight Panel (3-4 hours)
- [ ] Context-aware content (changes based on selection)
- [ ] Layer detail view
- [ ] **Attention heatmap component (NEW)**
- [ ] **Head ranking bar chart (NEW)**
- [ ] Token journey visualization
- [ ] Activation statistics display
- [ ] Probe prediction history chart

### Phase 8: Dimension Flow View (2-3 hours) - NEW
- [ ] PCA computation on backend
- [ ] Top-K PC visualization
- [ ] Connection based on PC importance
- [ ] Slider for number of PCs

### Phase 9: Polish (2-3 hours)
- [ ] Loading states
- [ ] Error handling
- [ ] Keyboard shortcuts (arrow keys for navigation)
- [ ] Responsive layout
- [ ] Performance optimization for 32×32 grid
- [ ] Documentation

---

## New Code Required for Attention Extraction

### `attention_extractor.py` (Backend)

```python
import torch
from typing import List, Dict, Any

def extract_attention_patterns(
    model,
    inputs: torch.Tensor,
    layers_to_extract: List[int] = None
) -> Dict[str, Any]:
    """
    Extract attention patterns from all layers and heads.
    
    Returns:
        {
            "patterns": {
                layer_idx: {
                    head_idx: attention_matrix (seq_len x seq_len)
                }
            },
            "statistics": {
                layer_idx: {
                    head_idx: {
                        "avg": float,
                        "max": float,
                        "entropy": float,
                        "sparsity": float
                    }
                }
            }
        }
    """
    model.eval()
    
    with torch.no_grad():
        outputs = model(
            inputs,
            output_attentions=True,
            return_dict=True
        )
    
    # outputs.attentions is tuple of (batch, num_heads, seq_len, seq_len)
    attention_patterns = {}
    attention_stats = {}
    
    for layer_idx, layer_attention in enumerate(outputs.attentions):
        if layers_to_extract and layer_idx not in layers_to_extract:
            continue
            
        # layer_attention shape: (batch, num_heads, seq_len, seq_len)
        attention_patterns[layer_idx] = {}
        attention_stats[layer_idx] = {}
        
        for head_idx in range(layer_attention.shape[1]):
            attn = layer_attention[0, head_idx].cpu().numpy()  # (seq_len, seq_len)
            
            attention_patterns[layer_idx][head_idx] = attn.tolist()
            attention_stats[layer_idx][head_idx] = {
                "avg": float(attn.mean()),
                "max": float(attn.max()),
                "entropy": float(compute_entropy(attn)),
                "sparsity": float(compute_sparsity(attn))
            }
    
    return {
        "patterns": attention_patterns,
        "statistics": attention_stats
    }


def compute_attention_flow(
    attention_patterns: Dict,
    num_layers: int,
    num_heads: int,
    threshold: float = 0.1
) -> List[Dict]:
    """
    Compute information flow between heads across layers.
    
    This estimates how much information from each head in layer L
    contributes to each head in layer L+1.
    """
    flow = []
    
    for layer_idx in range(num_layers - 1):
        layer_flow = {
            "source_layer": layer_idx,
            "target_layer": layer_idx + 1,
            "head_connections": []
        }
        
        # For each head in source layer
        for src_head in range(num_heads):
            # For each head in target layer
            for tgt_head in range(num_heads):
                # Compute correlation/similarity between attention patterns
                src_attn = attention_patterns[layer_idx][src_head]
                tgt_attn = attention_patterns[layer_idx + 1][tgt_head]
                
                weight = compute_attention_similarity(src_attn, tgt_attn)
                
                if weight > threshold:
                    layer_flow["head_connections"].append({
                        "source_head": src_head,
                        "target_head": tgt_head,
                        "weight": weight
                    })
        
        flow.append(layer_flow)
    
    return flow


def compute_entropy(attention: np.ndarray) -> float:
    """Compute entropy of attention distribution."""
    # Flatten and normalize
    attn_flat = attention.flatten()
    attn_flat = attn_flat[attn_flat > 1e-10]  # avoid log(0)
    return -np.sum(attn_flat * np.log(attn_flat))


def compute_sparsity(attention: np.ndarray, threshold: float = 0.1) -> float:
    """Compute sparsity (fraction of weights below threshold)."""
    return np.mean(attention < threshold)


def compute_attention_similarity(attn1: np.ndarray, attn2: np.ndarray) -> float:
    """Compute similarity between two attention patterns."""
    # Use cosine similarity on flattened attention
    a1 = np.array(attn1).flatten()
    a2 = np.array(attn2).flatten()
    
    norm1 = np.linalg.norm(a1)
    norm2 = np.linalg.norm(a2)
    
    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0
    
    return float(np.dot(a1, a2) / (norm1 * norm2))
```

### Frontend: AttentionFlow.tsx (Simplified)

```tsx
import React, { useRef, useEffect } from 'react';
import * as d3 from 'd3';

interface AttentionFlowProps {
  layers: number;
  heads: number;
  headData: HeadData[][];  // [layer][head]
  flowData: FlowData[];
  selectedToken: number | null;
  connectionThreshold: number;
  showCrossHead: boolean;
  onHeadClick: (layer: number, head: number) => void;
}

export const AttentionFlow: React.FC<AttentionFlowProps> = ({
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
    
    // Color scale for correctness
    const colorScale = d3.scaleLinear<string>()
      .domain([0, 0.5, 1])
      .range(['#EF4444', '#EAB308', '#22C55E']);
    
    // Draw connections first (behind nodes)
    const connectionsGroup = g.append('g').attr('class', 'connections');
    
    flowData.forEach(layerFlow => {
      layerFlow.head_connections
        .filter(conn => conn.weight >= connectionThreshold)
        .filter(conn => showCrossHead || conn.source_head === conn.target_head)
        .forEach(conn => {
          const x1 = layerFlow.source_layer * LAYER_SPACING;
          const y1 = conn.source_head * HEAD_SPACING;
          const x2 = layerFlow.target_layer * LAYER_SPACING;
          const y2 = conn.target_head * HEAD_SPACING;
          
          const isCrossHead = conn.source_head !== conn.target_head;
          
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
        
        g.append('circle')
          .attr('cx', x)
          .attr('cy', y)
          .attr('r', NODE_RADIUS)
          .attr('fill', colorScale(data?.probe_prediction?.confidence ?? 0.5))
          .attr('stroke', '#fff')
          .attr('stroke-width', 1)
          .attr('cursor', 'pointer')
          .on('click', () => onHeadClick(layer, head))
          .on('mouseenter', function() {
            d3.select(this).attr('r', NODE_RADIUS * 1.5);
          })
          .on('mouseleave', function() {
            d3.select(this).attr('r', NODE_RADIUS);
          });
      }
    }
    
    // Draw axis labels
    // ... (layer and head labels)
    
  }, [layers, heads, headData, flowData, connectionThreshold, showCrossHead]);
  
  return <svg ref={svgRef} />;
};

function bezierPath(x1: number, y1: number, x2: number, y2: number): string {
  const midX = (x1 + x2) / 2;
  return `M ${x1} ${y1} C ${midX} ${y1}, ${midX} ${y2}, ${x2} ${y2}`;
}
```

---

## Performance Considerations

### Attention Data Size
- 32 layers × 32 heads × (seq_len × seq_len) attention matrices
- For seq_len=100: 32 × 32 × 100 × 100 × 4 bytes ≈ 40MB per inference
- **Mitigation**: 
  - Store only diagonal/important values
  - Compute statistics on-the-fly, don't store full matrices
  - Allow lazy loading of full attention for selected layer/head

### Rendering Performance
- 32 × 32 = 1024 nodes
- Up to 32 × 32 × 31 = 31,744 connections (cross-head enabled)
- **Mitigation**:
  - Use connection threshold to limit visible connections
  - Use WebGL-based rendering (Three.js) if D3 SVG is too slow
  - Virtualize: only render visible layers in viewport

### Memory Management
- Clear previous session data when running new question
- Use Web Workers for heavy computation
- Stream large responses from backend

---

## Usage

### Start the visualization tool:
```bash
cd visualization
python run.py
```

This will:
1. Start the FastAPI backend on `http://localhost:8000`
2. Start the React frontend on `http://localhost:3000`
3. Open your browser automatically

### First run:
1. Select a model from the dropdown
2. Wait for model to load (first time may take ~30s)
3. Enter a question and click Run
4. **Select view mode** (Layer Overview, Attention Flow, or Dimension Flow)
5. Explore the visualization!

### Attention Flow Exploration:
1. Switch to "Attention Flow" view mode
2. Adjust connection threshold slider to show more/fewer connections
3. Toggle "Show cross-head connections" for information redistribution view
4. Click any head node to see its attention heatmap
5. Click a token to highlight its path through heads

---

## Future Enhancements

- [ ] **Comparison mode**: Side-by-side view of two different questions
- [ ] **Batch mode**: Visualize patterns across multiple questions
- [ ] **Export**: Save visualizations as images/PDFs
- [x] **Attention heads**: Visualize attention patterns (DESIGNED)
- [ ] **Neuron search**: Find neurons that activate for specific concepts
- [ ] **Training mode**: See how probe accuracy changes with more training data
- [ ] **3D visualization**: WebGL-based 3D view of layer×head×token space
- [ ] **Attention diff**: Compare attention patterns between correct/incorrect answers
