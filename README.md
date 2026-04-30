# SE(3)-VLA: Geodesic Action Chunking & Uncertainty-Aware Flow Matching

> **Novel contributions over Braun et al. (RFMP, IROS 2024):**
> 1. **Geodesic Action Chunking** — predict temporally-coherent action *chunks* on SE(3) via K anchor poses + geodesic interpolation (vs. single-pose prediction in RFMP)
> 2. **Uncertainty-Aware Flow Matching** — N-sample flow matching with geodesic variance + conformal prediction sets on SE(3) (vs. point estimates in all existing VLAs)
> 3. **First VLA with calibrated coverage guarantees** on the SE(3) manifold

[![Paper](https://img.shields.io/badge/paper-coming%20soon-blue)]()
[![Code](https://img.shields.io/badge/code-v0.2-green)]()
[![License](https://img.shields.io/badge/license-MIT-yellow)]()

---

## The Problem

All published VLAs (Octo, OpenVLA, π0, SmolVLA, RDT-1B) predict robot actions as:
- **Independent** H-step action chunks in flat Euclidean space (R⁶ axis-angle ⊕ R³ translation)
- **Point estimates** — no uncertainty quantification

This has three known failure modes:
1. **Antipodal discontinuity**: axis-angle has a 2-cover at ||θ|| = π
2. **Temporal incoherence**: H independent predictions can cross the antipodal boundary → discontinuous execution
3. **No uncertainty**: the robot can't know when it's wrong → no safe fallback

RFMP (Braun et al. 2024) addressed #1 via Riemannian flow matching on SE(3), but only for **single-pose** prediction — it doesn't address #2 or #3.

## Our Solution

### 1. Geodesic Action Chunking (Novel)

Instead of predicting H independent SE(3) poses, we predict **K anchor poses** (K << H) and interpolate H actions along geodesics:

```
h (VLA hidden) → MLP → K se(3) anchors → exp → SE(3)
SE(3) anchors → geodesic interpolation → H action poses
```

**Why this matters:**
- Robot manipulation trajectories are smooth on SE(3)
- Geodesic interpolation gives temporal consistency for free
- K=2 (start+end) is often sufficient; K=4 handles curved motions

**Comparison to prior work:**

| Method | Temporal structure | Manifold |
|--------|-------------------|----------|
| Octo/OpenVLA | None (H independent) | Euclidean R⁷ |
| Diffusion Policy | Iterative denoising | Euclidean |
| RFMP (Braun 2024) | Single pose | SE(3) |
| **Ours** | **Geodesic interpolation** | **SE(3)** |

### 2. Uncertainty-Aware Flow Matching (Novel)

Flow matching is generative — sampling is nearly free (different noise seeds, same ODE). We exploit this:

1. Draw N samples from the flow → N candidate SE(3) actions
2. Compute **Fréchet mean** on SE(3) (iterative log-map averaging)
3. Compute **geodesic variance** σ² = (1/N) Σ d(T̄, Tᵢ)²
4. Calibrate **conformal prediction sets**: {T : d(T, T̄) ≤ q_α}

**Coverage guarantee** (distribution-free): P(T* ∈ C_α) ≥ 1 - α

**This is the first VLA with principled uncertainty on SE(3).**

### 3. Combined Pipeline

```
VLA Backbone (frozen)
       ↓
   Hidden State h
       ↓
┌──────────────────────┐     ┌─────────────────────────┐
│ Geodesic Chunking    │     │ Uncertainty Flow Head    │
│ K anchors → H poses  │     │ N samples → σ² + C_α    │
└──────────────────────┘     └─────────────────────────┘
       ↓                              ↓
  Action Chunk [H, SE(3)]      Mean + Uncertainty + Conformal Set
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the novel experiment (geodesic chunking + uncertainty)
python experiments/run_novel_experiment.py --config configs/novel_se3.yaml --seed 0

# Run the original pilot sweep (12 runs)
bash experiments/run_all.sh

# Evaluate a checkpoint
python src/evaluate.py --config configs/octo_se3.yaml \
    --checkpoint checkpoints/OctoSE3_scene_id_seed0_best.pt
```

## Repository Structure

```
s3-godsec/
├── configs/
│   ├── octo_se3.yaml              # SE(3) flow head + scene_id
│   ├── octo_baseline.yaml         # Euclidean baseline + scene_id
│   ├── octo_se3_cnn.yaml          # SE(3) + mock CNN
│   ├── octo_baseline_cnn.yaml     # Euclidean + mock CNN
│   └── novel_se3.yaml             # Novel: chunking + uncertainty config
├── src/
│   ├── models/
│   │   ├── se3_action_head.py     # SE(3) flow matching head
│   │   ├── geodesic_chunking.py   # ★ Novel: geodesic action chunking
│   │   ├── uncertainty_head.py    # ★ Novel: uncertainty-aware flow
│   │   ├── geodesic_loss.py       # Geodesic loss functions
│   │   ├── octo_adapter.py        # Adapter for Octo VLA
│   │   ├── scene_id_backbone.py   # Scene-ID backbone for A/B testing
│   │   └── mock_backbone.py       # Mock CNN backbone
│   ├── training/
│   │   ├── data_loader.py         # Synthetic SE(3) dataset
│   │   └── trainer.py             # Training loop
│   ├── utils/
│   │   ├── se3_utils.py           # SE(3) operations (exp, log, geodesic)
│   │   ├── metrics.py             # Geodesic RMSE, rotation RMSE
│   │   └── visualization.py       # Geodesic visualization
│   ├── train.py                   # Main training script
│   └── evaluate.py                # Main evaluation script
├── experiments/
│   ├── run_novel_experiment.py    # ★ Novel: full pipeline experiment
│   ├── run_all.sh                 # 12-run sweep
│   ├── ablation_euclidean_vs_se3.py
│   ├── ablation_rotation_tasks.py
│   └── PREREGISTRATION.md
├── reports/
│   ├── SUMMARY.md                 # Pilot results
│   ├── PILOT_POSTLEAK.md          # Post-leakage-fix results
│   ├── BLOCKER.md                 # Phase 1 blocker (CuDNN/JAX)
│   └── PROGRESS.md                # Progress tracker
├── docs/                          # Research documentation
├── scratch/                       # Dev scripts
├── implementation_plan.md         # Phase-by-phase execution plan
├── requirements.txt
└── LICENSE
```

## Key Results

### Synthetic Diagnostic (Phase 0)

SE(3) head retains advantage over Euclidean baseline across all 3 seeds:

| Backbone | Seed | Euclid R-RMSE | SE(3) R-RMSE | Δ |
|----------|------|---------------|--------------|---|
| scene_id | 0    | 2.4330        | 2.3178       | +0.1152 |
| scene_id | 1    | 2.3443        | 2.2519       | +0.0925 |
| scene_id | 2    | 2.4394        | 2.3176       | +0.1218 |

### Novel Experiment

TBD — pending `python experiments/run_novel_experiment.py`

## Citation

```bibtex
@article{se3vla2026,
  title={SE(3)-VLA: Geodesic Action Chunking and Uncertainty-Aware Flow Matching for Vision-Language-Action Models},
  author={TBD},
  journal={arXiv preprint},
  year={2026}
}
```

## Acknowledgments

- Braun et al. for Riemannian Flow Matching Policy (RFMP, IROS 2024)
- Octo Model Team for the open-source VLA baseline
- `geoopt` library for Riemannian optimization in PyTorch

## License

MIT License
