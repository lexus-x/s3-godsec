# PREREGISTRATION — SE(3) vs Euclidean Head Experiment

**Date committed**: 2026-04-30 (before training)  
**Repository**: s3-godsec  

---

## Hypothesis

The SE(3) flow matching action head achieves lower rotation RMSE than the
Euclidean (axis-angle) action head on rotation-heavy tasks (θ ∈ [π/2, π]),
while showing comparable or worse performance on translation-heavy tasks
(θ ∈ [0, π/12]).

## Independent Variables

| Variable | Levels |
|----------|--------|
| Action head | `OctoSE3` (SE(3) flow matching), `OctoEuclideanBaseline` (R⁶ MLP) |
| Backbone | `scene_id` (clean embedding), `mock_cnn` (learned CNN) |
| Seed | 0, 1, 2 |

**Total runs**: 2 heads × 2 backbones × 3 seeds = **12 runs**

## Dependent Variables (per family)

| Metric | Definition |
|--------|-----------|
| Rotation RMSE | `sqrt(mean(geodesic_distance_rotation_only²))` in radians |
| Translation RMSE | `sqrt(mean(‖t_pred - t_target‖²))` in meters |
| Geodesic RMSE | `sqrt(mean(geodesic_distance²))` combined |

## Data Families (hard-bounded, no leakage)

| Family | θ range | ‖v‖ range |
|--------|---------|-----------|
| `rotation_heavy` | Uniform(π/2, π) | ≤ 0.05 |
| `translation_heavy` | Uniform(0, π/12) | Uniform(0.1, 0.5) |

Rotation axes sampled uniformly on S² in both families.

## Decision Rules

### Primary claim (rotation-heavy, scene_id backbone)

The SE(3) head is considered to show a **geometry advantage** if:

```
mean_over_seeds(R-RMSE_euclidean[rot_heavy]) - mean_over_seeds(R-RMSE_se3[rot_heavy]) > 0
```

AND this difference is **consistent across all 3 seeds** (i.e., SE(3) wins
or ties on rotation RMSE in every seed, not just in the mean).

### Secondary check (translation-heavy, scene_id backbone)

We expect **no meaningful difference** on translation-heavy tasks:

```
|mean_over_seeds(R-RMSE_se3[trans_heavy]) - mean_over_seeds(R-RMSE_euclidean[trans_heavy])| < 0.1 rad
```

### Backbone interaction

If SE(3) wins under `scene_id` but not `mock_cnn`, the conclusion is:
"SE(3) geometry advantage exists but requires clean features." This is
still a real, publishable result — it narrows the claim scope.

If SE(3) wins under both backbones, the result is stronger.

If SE(3) does not win under `scene_id`, the hypothesis is **rejected**.

## Acceptable Outcomes

| Outcome | Interpretation |
|---------|---------------|
| SE(3) < Euclid on rot_heavy (both backbones) | Strong support |
| SE(3) < Euclid on rot_heavy (scene_id only) | Partial support — narrower claim |
| SE(3) ≈ Euclid on rot_heavy | Null result — Euclidean sufficient |
| SE(3) > Euclid on rot_heavy | Negative — SE(3) head hurts |

All outcomes are valid. We commit to reporting whatever happens.

## Training Protocol

- 50 epochs, AdamW, lr=1e-4, cosine schedule, batch_size=32
- 2500 samples per family for training (5000 total)
- 250 samples per family for validation (500 total)
- Gradient clip norm = 1.0
- Best checkpoint selected by combined geodesic RMSE on validation set
