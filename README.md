# SE(3)-VLA: Geodesic Action Chunking & Uncertainty-Aware Flow Matching

> **Replacing the Euclidean action head of SmolVLA (450M) with Riemannian flow matching on SE(3), adding geodesic action chunking and calibrated conformal prediction — all under 500M parameters.**

[![Paper](https://img.shields.io/badge/paper-coming%20soon-blue)]()
[![Code](https://img.shields.io/badge/code-v0.3-green)]()
[![License](https://img.shields.io/badge/license-MIT-yellow)]()
[![Params](https://img.shields.io/badge/params-<500M-orange)]()

---

## The Problem

All published VLAs predict robot actions in **flat Euclidean space** (R⁶ axis-angle ⊕ R³ translation):

- **Antipodal discontinuity**: axis-angle has a 2-cover at ||θ|| = π
- **Temporal incoherence**: H independent predictions can cross the antipodal boundary
- **No uncertainty**: the robot can't know when it's wrong

SmolVLA (450M) uses flow matching for action prediction, but in Euclidean space — it inherits all three problems.

## Our Solution (3 Novel Contributions)

### 1. SE(3) Flow Matching Head for SmolVLA

Replace SmolVLA's Euclidean action expert with a Riemannian flow matching head on the SE(3) Lie group. Actions are predicted as geodesics on the manifold, respecting the true geometry of rigid body motions.

```
SmolVLA VLM (frozen, 430M) → hidden state h → SE(3) Flow Head (trainable, 20M) → T ∈ SE(3)
```

### 2. Geodesic Action Chunking

Instead of predicting H independent SE(3) poses, predict **K anchor poses** and interpolate H actions along geodesics:

```
h → MLP → K se(3) anchors → exp → SE(3) → geodesic interpolation → H actions
```

This gives temporal consistency for free. K=2 (start+end) handles straight motions; K=4 handles curved trajectories.

### 3. Uncertainty-Aware Flow with Conformal Prediction

Flow matching is generative — sampling is nearly free. Draw N samples, compute Fréchet mean and geodesic variance on SE(3), and calibrate conformal prediction sets:

```
N flow samples → Fréchet mean → geodesic variance σ² → conformal set {T : d(T, T̄) ≤ q_α}
```

**Coverage guarantee** (distribution-free): P(T* ∈ C_α) ≥ 1 − α

**First VLA with principled uncertainty on SE(3).**

## Parameter Budget (< 500M)

| Component | Params | Trainable |
|-----------|--------|-----------|
| SmolVLA VLM (SigLIP + SmolLM) | ~430M | ❌ frozen |
| SE(3) flow head | ~20M | ✅ |
| Geodesic chunking head | ~5M | ✅ |
| **Total** | **~455M** | **~25M** |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Install SmolVLA (requires lerobot)
git clone https://github.com/huggingface/lerobot.git
cd lerobot && pip install -e ".[smolvla]" && cd ..

# Train SmolVLA + SE(3) flow head on LIBERO
python src/train_smolvla.py --config configs/smolvla_se3.yaml --seed 0

# Train with geodesic chunking
python src/train_smolvla.py --config configs/smolvla_se3.yaml --seed 0 --head-type chunk

# Train with uncertainty-aware flow
python src/train_smolvla.py --config configs/smolvla_se3.yaml --seed 0 --head-type uncertainty

# Evaluate
python src/evaluate_smolvla.py --config configs/smolvla_se3.yaml \
    --checkpoint checkpoints/smolvla_se3/best.pt
```

## Repository Structure

```
s3-godsec/
├── configs/
│   ├── smolvla_se3.yaml           # ★ SmolVLA + SE(3) (main config)
│   ├── novel_se3.yaml             # Synthetic experiment config
│   ├── octo_se3.yaml              # Octo + SE(3) (legacy)
│   ├── octo_baseline.yaml         # Octo Euclidean baseline (legacy)
│   └── ...
├── src/
│   ├── backbones/
│   │   └── smolvla_backbone.py    # ★ SmolVLA adapter (frozen VLM)
│   ├── models/
│   │   ├── se3_action_head.py     # SE(3) flow matching head
│   │   ├── geodesic_chunking.py   # ★ Geodesic action chunking
│   │   ├── uncertainty_head.py    # ★ Uncertainty + conformal prediction
│   │   ├── geodesic_loss.py       # Geodesic loss functions
│   │   ├── octo_adapter.py        # Octo adapter (legacy)
│   │   └── ...
│   ├── training/
│   │   └── data_loader.py         # Synthetic + real datasets
│   ├── utils/
│   │   ├── se3_utils.py           # SE(3) operations (exp, log, geodesic)
│   │   ├── metrics.py             # Geodesic RMSE, rotation RMSE
│   │   └── visualization.py
│   ├── train_smolvla.py           # ★ SmolVLA training script
│   ├── evaluate_smolvla.py        # ★ SmolVLA evaluation script
│   └── ...
├── experiments/
│   ├── run_novel_experiment.py    # Novel pipeline experiment
│   └── run_all.sh                 # Full sweep
├── reports/
├── docs/
├── scratch/
├── implementation_plan.md
└── README.md
```

## Why SmolVLA?

| Property | Benefit |
|----------|---------|
| **450M params** | Fits under 500M budget with our SE(3) head |
| **PyTorch native** | No JAX/CUDA install hell (Octo has this problem) |
| **Flow matching** | Already uses flow matching — clean A/B test (Euclidean vs SE(3) flow) |
| **Pretrained** | Trained on LeRobot community data — real manipulation demos |
| **Open source** | Full training recipe available |

## Experimental Design

### A/B Test: Euclidean vs SE(3) Flow (same backbone, same budget)

| Model | Backbone | Action Head | Params |
|-------|----------|-------------|--------|
| SmolVLA-Euclidean (baseline) | SmolVLA-450M frozen | Euclidean flow (original) | 450M |
| SmolVLA-SE(3) (ours) | SmolVLA-450M frozen | SE(3) flow matching | 455M |
| SmolVLA-SE(3)-Chunk (ours) | SmolVLA-450M frozen | Geodesic chunking | 455M |
| SmolVLA-SE(3)-Unc (ours) | SmolVLA-450M frozen | Uncertainty flow | 455M |

### Benchmarks

- **LIBERO-Spatial**: 10 tasks, spatial reasoning
- **LIBERO-Object**: 10 tasks, object manipulation
- **Meta-World MT-10**: 10 tasks, diverse manipulation

### Metrics

- **Rotation RMSE** (SO(3) geodesic) — primary metric
- **Translation RMSE** — secondary metric
- **Geodesic RMSE** (SE(3)) — combined metric
- **Temporal smoothness** — mean geodesic distance between consecutive actions
- **Conformal coverage** — empirical coverage of prediction sets
- **Success rate** — task completion on benchmarks

## Key Results

### Synthetic Diagnostic (Phase 0)

SE(3) head retains advantage over Euclidean baseline across all 3 seeds:

| Backbone | Seed | Euclid R-RMSE | SE(3) R-RMSE | Δ |
|----------|------|---------------|--------------|---|
| scene_id | 0 | 2.4330 | 2.3178 | +0.1152 |
| scene_id | 1 | 2.3443 | 2.2519 | +0.0925 |
| scene_id | 2 | 2.4394 | 2.3176 | +0.1218 |

### SmolVLA Experiments

TBD — pending `python src/train_smolvla.py --config configs/smolvla_se3.yaml`

## Citation

```bibtex
@article{se3vla2026,
  title={SE(3)-VLA: Geodesic Action Chunking and Uncertainty-Aware Flow Matching on SmolVLA},
  author={TBD},
  journal={arXiv preprint},
  year={2026}
}
```

## Related Work

- **SmolVLA** (HuggingFace, 2025): 450M VLA with Euclidean flow matching
- **RFMP** (Braun et al., IROS 2024): Riemannian flow matching for single-pose robot policies
- **Octo** (UC Berkeley, 2024): Open-source generalist robot policy (27M/93M)
- **OpenVLA** (Stanford, 2024): 7B open-source VLA

## License

MIT License
