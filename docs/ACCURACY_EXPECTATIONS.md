# Accuracy Expectations & Realistic Projections

## Benchmark Baselines (Existing VLAs on LIBERO)

| Model | Params | LIBERO-Spatial | LIBERO-Object | LIBERO-Long |
|-------|--------|---------------|---------------|-------------|
| Octo-Base | 93M | ~55% | ~50% | ~35% |
| SmolVLA | 450M | ~75% | ~70% | ~55% |
| OpenVLA | 7B | ~85% | ~80% | ~65% |
| Evo-1 | 770M | ~95% | ~90% | ~80% |

## Our Realistic Projections

### Rotation Metrics (our strength)

| Metric | SmolVLA baseline | Our expected | Delta |
|--------|-----------------|-------------|-------|
| Rotation RMSE | ~2.3 rad | ~2.1 rad | **-8 to 10%** |
| Geodesic RMSE | ~1.7 | ~1.6 | **-5 to 8%** |

Evidence: Synthetic diagnostic showed consistent 4-5% rotation improvement across all seeds.

### Translation Metrics (our weakness)

| Metric | SmolVLA baseline | Our expected | Delta |
|--------|-----------------|-------------|-------|
| Translation RMSE | ~0.32 m | ~0.35 m | **+5 to 15% worse** |

Evidence: Synthetic diagnostic showed 10-41% translation regression. The SE(3) flow head couples rotation and translation, which can hurt pure translation tasks.

### Task Success Rate (what benchmarks measure)

| Benchmark | SmolVLA baseline | Our realistic range | Notes |
|-----------|-----------------|-------------------|-------|
| LIBERO-Spatial | ~75% | **72-78%** | Spatial tasks need rotation — small help |
| LIBERO-Object | ~70% | **68-73%** | Mixed rotation/translation — neutral |
| LIBERO-Long | ~55% | **52-58%** | Long-horizon compounds errors — risky |

**Expected delta: ±3% on success rate** — within noise.

### Temporal Smoothness (geodesic chunking)

| Metric | SmolVLA (independent) | Ours (geodesic) | Delta |
|--------|---------------------|-----------------|-------|
| Consecutive action distance | High (independent) | Low (geodesic) | **-30 to 50%** |
| Antipodal crossings | Possible | Impossible | **-100%** |

### Uncertainty Calibration (conformal prediction)

| Metric | Target | Expected |
|--------|--------|----------|
| Coverage (α=0.1) | 90% | 88-92% |
| Variance-error correlation | >0.5 | 0.4-0.7 |

## Why We Won't Beat SmolVLA's Success Rate

1. **Data gap**: SmolVLA trained on 30k+ episodes; we fine-tune on ~2.5k
2. **Co-adaptation**: SmolVLA's head co-adapted with VLM during pretraining; ours is frozen + from scratch
3. **Translation regression**: SE(3) coupling hurts translation-heavy tasks
4. **Benchmark design**: LIBERO measures success rate, not rotation accuracy

## What Gets Published

| Result | Venue | Likelihood |
|--------|-------|-----------|
| Rotation RMSE ↓8%, success rate ±2%, + uncertainty | RA-L short / CoRL workshop | **60%** |
| Rotation RMSE ↓5%, success rate neutral, + smoothness + uncertainty | Workshop paper | **80%** |
| Null result (no improvement) | arXiv negative result | **20%** |

## How to Maximize Chances

1. **End-to-end fine-tuning**: Unfreeze backbone, fine-tune whole model on LIBERO
2. **More data**: Train on full LeRobot community dataset
3. **Per-task tuning**: Different hyperparams for spatial vs object vs long tasks
4. **Ensemble**: Combine SE(3) head with Euclidean head, select per-task

## Paper Framing

> We demonstrate that SE(3) Riemannian geometry improves rotation prediction accuracy by X% over Euclidean flow matching on the same SmolVLA backbone under a 500M parameter budget. We further introduce geodesic action chunking for temporal consistency and conformal prediction for calibrated uncertainty — capabilities absent from all existing VLAs.

The paper is about **geometry + uncertainty**, not about beating the leaderboard.
