# SE(3)-VLA Training Guide

> **Complete guide to training, evaluating, and publishing the SE(3)-VLA model.**

---

## Quick Start (TL;DR)

```bash
# 1. Setup environment
bash scripts/setup_environment.sh

# 2. Smoke test (5 min, synthetic data)
bash scripts/run_pipeline.sh --smoke

# 3. Full training (all heads, 3 seeds)
bash scripts/run_pipeline.sh

# 4. Individual training
python src/train_smolvla.py --config configs/smolvla_se3.yaml --head-type flow --seed 0
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    SE(3)-VLA Architecture                        │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │   SmolVLA    │    │  SE(3) Flow  │    │    Geodesic      │  │
│  │   VLM-450M   │───▶│    Head      │───▶│    Actions       │  │
│  │  (frozen)    │    │  (~20M)      │    │   T ∈ SE(3)      │  │
│  └──────────────┘    └──────────────┘    └──────────────────┘  │
│                                                                 │
│  Vision (SigLIP)     Flow Matching      Riemannian             │
│  Language (SmolLM)    on SE(3)           Interpolation          │
└─────────────────────────────────────────────────────────────────┘
```

**Parameter Budget:**
| Component | Params | Trainable |
|-----------|--------|-----------|
| SmolVLA VLM | ~430M | ❌ frozen |
| SE(3) flow head | ~20M | ✅ |
| Geodesic chunking head | ~5M | ✅ |
| **Total** | **~455M** | **~25M** |

---

## Three Head Types

### 1. Flow Head (`--head-type flow`)
**Standard SE(3) flow matching — single action prediction.**

- Learns a conditional flow on SE(3) from source distribution to target action
- Predicts velocity field in the Lie algebra se(3)
- Integrates flow via Euler method (10-20 steps)
- Best for: single-step action prediction

### 2. Chunk Head (`--head-type chunk`)
**Geodesic action chunking — temporally coherent action sequences.**

- Predicts K anchor poses on SE(3) (K << H)
- Interpolates H actions along geodesics between anchors
- Gives temporal consistency for free
- Best for: smooth trajectory prediction, reducing jerk

### 3. Uncertainty Head (`--head-type uncertainty`)
**Uncertainty-aware flow matching — calibrated confidence.**

- Draws N samples from the flow (different noise seeds)
- Computes Fréchet mean and geodesic variance on SE(3)
- Provides conformal prediction sets with coverage guarantees
- Best for: safety-critical applications, knowing when the model is wrong

---

## Training Configuration

### Config Files

| Config | Description | Use Case |
|--------|-------------|----------|
| `configs/smolvla_se3.yaml` | SmolVLA + SE(3) (main) | Primary experiments |
| `configs/smolvla_flow.yaml` | Flow head only | Single-action tasks |
| `configs/smolvla_chunk.yaml` | Chunk head only | Trajectory tasks |
| `configs/smolvla_uncertainty.yaml` | Uncertainty head | Safety-critical |
| `configs/novel_se3.yaml` | Synthetic experiments | Development/debugging |

### Key Parameters

```yaml
model:
  hidden_dim: 768          # SmolVLA's VLM hidden size
  head_hidden_dim: 256     # Flow head MLP hidden dim
  n_layers: 4              # Flow head MLP depth
  source_scale: 0.1        # Source distribution scale

training:
  learning_rate: 1.0e-4    # AdamW LR
  batch_size: 32           # Batch size
  n_epochs: 50             # Training epochs
  gradient_clip_norm: 1.0  # Gradient clipping
```

---

## Step-by-Step Training

### Step 1: Environment Setup

```bash
# Basic setup
bash scripts/setup_environment.sh

# With SmolVLA backbone
bash scripts/setup_environment.sh --smolvla

# Full setup (SmolVLA + benchmarks)
bash scripts/setup_environment.sh --full
```

**Requirements:**
- Python 3.9+
- PyTorch 2.0+
- CUDA GPU (A100 recommended, works on RTX 3090+)
- ~16GB VRAM for full training

### Step 2: Verify Installation

```bash
# Quick sanity check
python -c "
from src.models.se3_action_head import SE3ActionPredictor
from src.utils.se3_utils import exp_se3, log_se3
import torch
print('SE(3) utilities: OK')
xi = torch.randn(4, 6) * 0.1
T = exp_se3(xi)
xi_recovered = log_se3(T)
print(f'Exp/Log roundtrip error: {(xi - xi_recovered).abs().max():.6f}')
"
```

### Step 3: Smoke Test

```bash
# 5-minute smoke test with synthetic data
bash scripts/run_pipeline.sh --smoke
```

This will:
- Use synthetic data (no real benchmark needed)
- Train all 3 head types for 5 epochs
- Evaluate and generate a report
- Verify the entire pipeline works

### Step 4: Full Training

```bash
# Train all heads, 3 seeds each (9 total runs)
bash scripts/run_pipeline.sh

# Or train individually:
python src/train_smolvla.py --config configs/smolvla_se3.yaml --head-type flow --seed 0
python src/train_smolvla.py --config configs/smolvla_se3.yaml --head-type chunk --seed 0
python src/train_smolvla.py --config configs/smolvla_se3.yaml --head-type uncertainty --seed 0
```

### Step 5: Evaluation

```bash
# Evaluate a specific checkpoint
python src/evaluate_smolvla.py \
    --config configs/smolvla_se3.yaml \
    --checkpoint checkpoints/smolvla_se3/smolvla_flow_seed0_best.pt \
    --head-type flow

# Evaluate uncertainty (includes conformal coverage)
python src/evaluate_smolvla.py \
    --config configs/smolvla_se3.yaml \
    --checkpoint checkpoints/smolvla_se3/smolvla_uncertainty_seed0_best.pt \
    --head-type uncertainty
```

---

## Training with Real Benchmarks

### LIBERO (Recommended)

```bash
# 1. Install LIBERO
pip install libero

# 2. Update config
# Set data.benchmark: "libero_spatial" in config

# 3. Train
python src/train_smolvla.py --config configs/smolvla_se3.yaml --head-type flow --seed 0
```

### MetaWorld MT-10

```bash
# 1. Install MetaWorld
pip install git+https://github.com/Farama-Foundation/Metaworld.git

# 2. Update config
# Set data.benchmark: "metaworld_mt10" in config

# 3. Train
python src/train_smolvla.py --config configs/smolvla_se3.yaml --head-type flow --seed 0
```

---

## Output Structure

```
checkpoints/smolvla_se3/
├── smolvla_flow_seed0_best.pt          # Best flow head (seed 0)
├── smolvla_flow_seed0_history.json     # Training history
├── smolvla_flow_seed0_results.json     # Final metrics
├── smolvla_chunk_seed0_best.pt         # Best chunk head
├── smolvla_uncertainty_seed0_best.pt   # Best uncertainty head
└── ...

results/
├── smolvla_flow_seed0_eval.json        # Evaluation results
├── smolvla_chunk_seed0_eval.json
├── smolvla_uncertainty_seed0_eval.json
└── ...

reports/
└── PIPELINE_REPORT_*.md                # Auto-generated report
```

---

## Metrics Explained

| Metric | Description | Lower is better |
|--------|-------------|-----------------|
| **G-RMSE** | Geodesic RMSE on SE(3) | ✅ |
| **R-RMSE** | Rotation RMSE (SO(3) only) | ✅ |
| **T-RMSE** | Translation RMSE | ✅ |
| **Temporal Smoothness** | Mean geodesic distance between consecutive actions | ✅ |
| **Conformal Coverage** | Empirical coverage of prediction sets (target: 1-α) | ≈ target |
| **Gripper Accuracy** | Binary gripper prediction accuracy | ❌ higher |

---

## Hyperparameter Guide

### Flow Steps
- **Training:** 10 steps (default, good balance)
- **Evaluation:** 10-20 steps (more steps = more accurate)
- **Speed/accuracy tradeoff:** 1 step ≈ 2× faster, slightly worse

### Source Scale
- **0.1** (default) — good for most tasks
- **0.05** — tighter source, better for fine manipulation
- **0.2** — wider source, better for large motions

### Head Hidden Dim
- **256** (default) — good balance
- **128** — faster, slightly worse
- **512** — slower, slightly better (diminishing returns)

### Chunk Size (chunk head only)
- **8** (default) — standard action chunk
- **4** — shorter chunks, less temporal smoothing
- **16** — longer chunks, more temporal smoothing

### N Anchors (chunk head only)
- **2** (default) — start + end, linear geodesic
- **4** — 4 waypoints, piecewise geodesic (curved trajectories)

---

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
python src/train_smolvla.py --config configs/smolvla_se3.yaml --seed 0
# Edit config: training.batch_size: 16  (or 8)
```

### SmolVLA Not Found
```bash
# The script will fall back to mock backbone automatically
# To use real SmolVLA:
bash scripts/setup_environment.sh --smolvla
```

### NaN Loss
- Check learning rate (try 1e-5)
- Check gradient clipping (should be 1.0)
- Reduce source_scale to 0.05

### Slow Training
- Use GPU (CUDA)
- Reduce n_flow_steps_train to 5
- Use fewer dataloader workers

---

## Publishing

### Model Card

After training, create a model card:

```python
# Generate model card
python scripts/generate_model_card.py \
    --results results/ \
    --output MODEL_CARD.md
```

### HuggingFace Hub

```bash
# Upload to HuggingFace
pip install huggingface_hub
huggingface-cli login

python scripts/push_to_hub.py \
    --checkpoint checkpoints/smolvla_se3/smolvla_flow_seed0_best.pt \
    --repo-name se3-vla-smolvla
```

### Paper Results

The pipeline generates all tables and figures needed for the paper:
- Per-family metrics (rotation_heavy vs translation_heavy)
- Per-seed statistics (mean ± std)
- Conformal coverage curves
- Temporal smoothness analysis

---

## Command Reference

```bash
# Training
python src/train_smolvla.py --config CONFIG --head-type HEAD --seed SEED [--epochs N] [--resume CKPT]

# Evaluation
python src/evaluate_smolvla.py --config CONFIG --checkpoint CKPT --head-type HEAD [--output OUT]

# Pipeline
bash scripts/run_pipeline.sh [--head HEAD] [--seed SEED] [--smoke] [--dry-run]

# Setup
bash scripts/setup_environment.sh [--smolvla] [--benchmarks] [--full]
```
