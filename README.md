# SE(3)-VLA: Geodesic Action Head for Vision-Language-Action Models

> *This is a synthetic diagnostic. Main results are in `reports/MAIN_RESULTS.md` (real benchmark).*
> **Replacing Euclidean action prediction with Riemannian flow matching on SE(3) for rotationally-correct robot manipulation.**

[![Paper](https://img.shields.io/badge/paper-coming%20soon-blue)]()
[![Code](https://img.shields.io/badge/code-coming%20soon-green)]()
[![License](https://img.shields.io/badge/license-MIT-yellow)]()

---

## The Problem

All published VLAs (Octo, OpenVLA, π0, SmolVLA, RDT-1B) predict robot actions in **flat Euclidean space** (R⁶ axis-angle ⊕ R³ translation). This is theoretically suboptimal for rotations:

- **Antipodal discontinuity**: axis-angle has a 2-cover at ||θ|| = π
- **Double-cover waste**: quaternions q and -q represent the same rotation, but Euclidean MSE penalizes both
- **Chunked prediction breaks**: action chunks (H=8) can cross the antipodal boundary → discontinuous execution

## The Solution

**SE(3)-VLA** replaces the Euclidean action head with a **Riemannian flow matching head on SE(3)** — the Lie group of rigid body motions. Actions are predicted as geodesics on the manifold, respecting the true geometry of robot motions.

## Key Results

TBD — pending Phase 4 results

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download pretrained Octo checkpoint
python scripts/download_octo.py

# Train SE(3) action head
python src/train.py --config configs/octo_se3.yaml

# Evaluate
python src/evaluate.py --config configs/octo_se3.yaml --checkpoint checkpoints/best.pt
```

## Repository Structure

```
se3-vla/
├── README.md                    # This file
├── LICENSE
├── requirements.txt
├── docs/
│   ├── MOTIVATION.md            # Why this research direction
│   ├── SELECTION_PROCESS.md     # How we selected this over alternatives
│   ├── TECHNICAL_DETAILS.md     # Full mathematical formulation
│   ├── EXPERIMENT_PLAN.md       # Detailed experiment protocol
│   ├── PAPER_OUTLINE.md         # Paper structure and writing plan
│   ├── RISK_ANALYSIS.md         # What could go wrong + mitigations
│   └── TIMELINE.md              # Week-by-week execution plan
├── src/
│   ├── models/
│   │   ├── se3_action_head.py   # Core SE(3) flow matching head
│   │   ├── se3_layers.py        # SE(3)-specific neural network layers
│   │   ├── geodesic_loss.py     # Geodesic distance and loss functions
│   │   └── octo_adapter.py      # Adapter to plug SE(3) head into Octo
│   ├── utils/
│   │   ├── se3_utils.py         # SE(3) operations (exp, log, adjoint)
│   │   ├── visualization.py     # Geodesic visualization tools
│   │   └── metrics.py           # Action-ECE and geodesic metrics
│   ├── training/
│   │   ├── trainer.py           # Training loop
│   │   └── data_loader.py       # Data loading utilities
│   ├── train.py                 # Main training script
│   └── evaluate.py              # Main evaluation script
├── experiments/
│   ├── ablation_euclidean_vs_se3.py
│   ├── ablation_rotation_tasks.py
│   └── run_all.sh
├── configs/
│   ├── octo_se3.yaml            # SE(3) action head config
│   └── octo_baseline.yaml       # Baseline Euclidean config
├── paper/
│   └── (draft coming)
└── references/
    └── (key papers)
```

## Citation

```bibtex
@article{se3vla2026,
  title={SE(3)-VLA: Riemannian Flow Matching on Lie Groups for Vision-Language-Action Models},
  author={TBD},
  journal={arXiv preprint},
  year={2026}
}
```

## Acknowledgments

- Octo Model Team for the open-source VLA baseline
- `geoopt` library for Riemannian optimization in PyTorch
- `theseus` library for SE(3) operations

## License

MIT License
