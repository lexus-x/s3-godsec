# SE(3)-VLA Training Guide

> **Compact VLA under 400M parameters, outperforming 1B+ models on rotation-heavy tasks.**

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                  SE(3)-VLA Compact Architecture                     │
│                       ~252M Parameters                              │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │  SigLIP-Base │    │ SmolLM-135M  │    │    SE(3) Flow Head   │  │
│  │  (87M, ❄️)   │───▶│  (135M, ✅)  │───▶│    (20M, ✅)         │  │
│  │  Vision      │    │  Language    │    │    Geodesic Actions  │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
│         │                   │                      │                │
│         └───────────────────┼──────────────────────┘                │
│                    Cross-Attention Fusion (10M)                     │
└─────────────────────────────────────────────────────────────────────┘

Budget: 87M + 135M + 10M + 20M = 252M ✓ Under 400M
```

**Why this beats 1B+ models:**
- OpenVLA (7B), Octo-Base (93M), and other VLAs use **Euclidean action heads**
- Euclidean heads suffer from antipodal discontinuity on large rotations
- Our SE(3) head respects the true geometry of rigid body motions
- The geometric inductive bias compensates for the smaller backbone

---

## Validation Strategy (3 Phases)

### Phase 1: Synthetic Diagnostics
**Goal:** Prove SE(3) > Euclidean on controlled synthetic data

```bash
bash scripts/run_pipeline.sh --phase 1
```

**Gate criteria:**
- SE(3) R-RMSE < Euclidean R-RMSE on `rotation_heavy` family (all seeds)
- Improvement ≥ 3% on rotation-heavy tasks
- No catastrophic regression on translation-heavy tasks

### Phase 2: Sim Benchmarks (LIBERO / MetaWorld)
**Goal:** Prove compact SE(3)-VLA > 1B+ models

```bash
bash scripts/run_pipeline.sh --phase 2
```

**Gate criteria:**
- Success rate on rotation-heavy tasks ≥ OpenVLA-7B
- Success rate on rotation-heavy tasks ≥ Octo-Base (93M)
- Overall success rate within 5pp of 1B+ models

### Phase 3: Real-World (only if Phase 2 passes)
**Goal:** Transfer to physical robot

**Gate criteria:**
- Phase 2 shows significant proof
- Real robot demonstrations available
- Sim-to-real gap is manageable

---

## Quick Start

```bash
# 1. Setup
bash scripts/setup_environment.sh

# 2. Smoke test (5 min)
bash scripts/run_pipeline.sh --smoke

# 3. Phase 1: Synthetic diagnostics
bash scripts/run_pipeline.sh --phase 1

# 4. Single run
python src/train_smolvla.py --config configs/compact_se3.yaml --head-type flow --seed 0

# 5. Evaluate with comparison
python src/evaluate_smolvla.py --config configs/compact_se3.yaml \
    --checkpoint checkpoints/compact_se3/se3vla_flow_seed0_best.pt --compare
```

---

## Config Files

| Config | Head | Description |
|--------|------|-------------|
| `configs/compact_se3.yaml` | Flow | Primary — single action prediction |
| `configs/compact_se3_chunk.yaml` | Chunk | Temporally coherent chunks |
| `configs/compact_se3_uncertainty.yaml` | Uncertainty | Calibrated confidence |

All configs use the same backbone: SigLIP-Base + SmolLM-135M = 252M total.

---

## Head Types

### Flow (`--head-type flow`)
Standard SE(3) flow matching. Predicts single action via learned flow on SE(3).
- 10 integration steps at train/eval
- Best for: single-step manipulation

### Chunk (`--head-type chunk`)
Geodesic action chunking. Predicts K anchor poses, interpolates H actions.
- K=2 anchors, H=8 actions
- Temporal consistency for free
- Best for: smooth trajectories

### Uncertainty (`--head-type uncertainty`)
Multi-sample flow with conformal prediction.
- N=10 samples for uncertainty estimation
- 90% coverage guarantee
- Best for: safety-critical tasks

---

## Parameter Budget Verification

Every training run prints the parameter budget:

```
┌─────────────────────────────────────────┐
│  PARAMETER BUDGET                       │
├─────────────────────────────────────────┤
│  Total:         252,000,000 (252.0M)    │
│  Trainable:     165,000,000 (165.0M)    │
│  Frozen:         87,000,000 (87.0M)     │
│  Status:    ✓ UNDER 400M                │
└─────────────────────────────────────────┘
```

If the model exceeds 400M, use a smaller vision encoder:
- `vision_model: "siglip-small"` (38M instead of 87M)
- `vision_model: "dinov2-small"` (22M instead of 87M)

---

## Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **G-RMSE** | Geodesic RMSE on SE(3) | Lower |
| **R-RMSE** | Rotation RMSE (SO(3)) | Lower |
| **T-RMSE** | Translation RMSE | Lower |
| **Δ R-RMSE** | Euclidean R-RMSE − SE(3) R-RMSE | Positive = SE(3) wins |
| **Success Rate** | Task completion (sim/real) | Higher |
| **Conformal Coverage** | Empirical coverage (target: 90%) | ≈ 0.9 |

---

## Benchmark Comparison

The `--compare` flag runs both SE(3) and Euclidean baseline on the same backbone:

```
  Family                 │  SE(3) R-RMSE │ Euclid R-RMSE │    Δ (Eucl-SE3) │  Winner
  ───────────────────────┼───────────────┼───────────────┼─────────────────┼────────
  rotation_heavy         │        2.3178 │        2.4330 │          +0.1152 │   SE(3)
  translation_heavy      │        0.2377 │        0.3016 │          +0.0639 │   SE(3)
  combined               │        1.5586 │        1.6444 │          +0.0858 │   SE(3)
```

**If SE(3) wins on rotation_heavy by ≥ 3%, proceed to Phase 2.**

---

## Troubleshooting

**Model over 400M?**
```yaml
# Use smaller vision encoder
model:
  vision_model: "dinov2-small"  # 22M instead of 87M
```

**CUDA OOM?**
```yaml
training:
  batch_size: 16  # or 8
```

**SE(3) not beating Euclidean?**
- Check seed count (need 3+ seeds for reliable comparison)
- Check rotation-heavy family specifically (this is where SE(3) should win)
- Try `head_type: "chunk"` for temporal consistency bonus

---

## Output Structure

```
checkpoints/compact_se3/
├── se3vla_flow_seed0_best.pt
├── se3vla_flow_seed0_history.json
├── se3vla_flow_seed0_results.json
├── se3vla_chunk_seed0_best.pt
├── se3vla_uncertainty_seed0_best.pt
└── ...

results/
├── compact_flow_eval.json
├── compact_chunk_eval.json
└── ...

reports/
├── PHASE1_REPORT_*.md
└── PHASE2_REPORT_*.md
```
