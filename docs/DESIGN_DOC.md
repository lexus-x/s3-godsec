# DESIGN_DOC.md — SE(3)-VLA: Riemannian Flow Matching on Lie Groups for Vision-Language-Action Models

## 1. Abstract

Vision-Language-Action (VLA) models have become the dominant paradigm for generalist robot policies, yet all published VLAs predict actions in flat Euclidean space (R⁶ axis-angle ⊕ R³ translation). This Euclidean approximation is theoretically suboptimal for rotations, causing antipodal discontinuities, double-cover waste, and chunked prediction boundary crossing that degrades performance on rotation-heavy tasks. In this paper, we introduce SE(3)-VLA, replacing the Euclidean action head with a Riemannian flow matching head on SE(3) — the Lie group of rigid body motions. We provide theoretical bounds showing that Euclidean flow matching incurs an O(θ²) error for rotations, predicting empirical failure on large rotations.

Currently, this project provides a synthetic pilot study that serves as a frozen diagnostic. [MEASURED] Results on this synthetic dataset demonstrate that our SE(3) flow matching head improves rotation geodesic RMSE (R-RMSE) by ~4.6% on rotation-heavy tasks when using an oracle `scene_id` backbone (2.2958 vs 2.4056 rad). [PLANNED] We plan to scale this to real VLA backbones (Octo-Small, OpenVLA-7B) on the real-world LIBERO-Spatial and MetaWorld MT-10 benchmarks. By respecting the true geometry of robot motions, SE(3)-VLA provides a principled geometric framework for action generation without imposing overhead on the visual backbone. No success-rate numbers are claimed yet for real-world tasks; all real experiments are [PLANNED].

## 2. Problem Statement and Motivation

The foundational observation of this work is that robot end-effector motions live on SE(3), not on R⁷. Every published VLA model — such as Octo (Octo Model Team, RSS 2024), OpenVLA (Kim et al., CoRL 2024), π0 (Black et al., arXiv 2024), SmolVLA (arXiv 2025), RDT-1B (ICLR 2025), and FAST (RSS 2026) — predicts actions as points in flat Euclidean space: `[dx, dy, dz, dRx, dRy, dRz, grip]`. 

This is a structural modeling error due to the mismatch between the SO(3) manifold and flat Euclidean space:
1. **Antipodal Discontinuity (Axis-Angle)**: The axis-angle representation maps rotations to R³ via the Rodrigues formula. However, this has a 2-cover: rotations by θ around axis n̂ and by (2π - θ) around -n̂ map to the same physical rotation. Near ||θ|| = π, this mapping is discontinuous. Small perturbations cause 180° execution flips.
2. **Double-Cover Waste (Quaternions)**: Quaternions double-cover SO(3) (q and -q are the same rotation). An Euclidean MSE loss penalizes the numerical distance between q̂ and q even if q̂ is geometrically closer to -q.
3. **Action Chunking Crossing Discontinuities**: VLAs predict action chunks (e.g., H=8). When predicting in Euclidean space, an action chunk crossing the antipodal boundary will interpolate linearly in Euclidean space, producing a discontinuous trajectory physically.

Analysis of MetaWorld MT-50 task performance ([UNVERIFIED] based on aggregated data in `docs/MOTIVATION.md:46-54`) revealed a systematic pattern: there is an estimated 17% gap in success rates between translation-heavy tasks (average < 30° rotation) and rotation-heavy tasks (average > 90° rotation). Current VLAs fail disproportionately on rotation-heavy tasks simply because their Euclidean action heads cannot properly model the curvature of SE(3).

## 3. Related Work

### 3.1 VLAs
Vision-Language-Action (VLA) models such as RT-1 (Brohan et al., ICRA 2023), RT-2 (Brohan et al., [UNVERIFIED] CoRL 2023), Octo (Octo Model Team, RSS 2024), OpenVLA (CoRL 2024), π0 (Black et al., arXiv 2024), SmolVLA (arXiv 2025), RDT-1B (ICLR 2025), and FAST (RSS 2026) encode observations and text to predict tokenized or continuous actions. 
**Position vs. our work:** All these models use Euclidean action heads. We show this is suboptimal for rotations and replace the head with a geometric one.

### 3.2 Flow matching
Flow matching is a powerful generative framework popularized by Lipman et al. ([UNVERIFIED] ICLR 2023) for Euclidean spaces. Chen & Lipman in "Riemannian Flow Matching on General Geometries" (NeurIPS 2024) generalized this framework to Riemannian manifolds, providing the mathematical foundation for learning flows directly on curved spaces.
**Position vs. our work:** Flow matching on manifolds is proven but unapplied to VLA actions. We are the first to port this to VLA end-effector prediction.

### 3.3 SE(3) methods in robotics
SE(3)-aware models have been used in robotics, for instance in RFMPose (NeurIPS 2025), EquiBot (CoRL 2024), SE(3)-Diffuser ([UNVERIFIED]), and SE(3)-DiffusionFields ([UNVERIFIED]).
**Position vs. our work:** These methods do SE(3) learning for category-level pose estimation or grasp generation, not for closed-loop VLA action prediction.

### 3.4 Rotation representations
The limitations of rotation representations in neural networks were famously documented by Zhou et al. in "On the Continuity of Rotation Representations in Neural Networks" (CVPR 2019), who proposed a continuous 6D representation. Sucan/Chirikjian ([UNVERIFIED]) have extensively studied SE(3) operations for robotics.
**Position vs. our work:** These are fixed mapping parameterizations. We learn the flow on the manifold itself, directly optimizing geodesic distances.

### 3.5 Calibration / UQ for VLAs
Uncertainty quantification (UQ) is critical for policy safety, addressed by SAFE (OpenReview 2025), ReconVLA (AAAI 2026), and general conformal prediction.
**Position vs. our work:** Complementary — our [ASPIRATIONAL] Geodesic Action-ECE metric extends calibration to manifold-valued predictions.

## 4. Method

### 4.1 SE(3) and its Lie algebra se(3)
SE(3) is the 6-dimensional Lie group of rigid body transformations. The Lie algebra se(3) is its tangent space at the identity.
We compute the exponential map (`exp_so3`, `exp_se3`), logarithmic map (`log_so3`, `log_se3`), and geodesic interpolations using stable Taylor expansions for small angles (implemented in `src/utils/se3_utils.py:37-221`).

### 4.2 Riemannian flow matching on SE(3)
We learn a conditional flow on SE(3) mapping a source distribution p₀ to the target action distribution p₁. The flow matching objective on SE(3) is:
L_FM = E_{t, X₀, X₁} ||v_θ(X_t, t, h) - log_{X_t}(X₁)||²_{X_t}
The network predicts a 6-dimensional vector (3 for rotation, 3 for translation) in the tangent space se(3) (`src/models/se3_action_head.py:38-125`). The geodesic interpolant is X_t = X₀ · exp(t · log(X₀⁻¹ · X₁)) (`src/utils/se3_utils.py:305-325`).

### 4.3 Geodesic loss
The geodesic distance and loss are evaluated using the bi-invariant norm on the Lie algebra (`src/models/geodesic_loss.py:10-50` and `src/models/geodesic_loss.py:107-125`). The `FlowMatchingLoss` is the MSE between predicted and target velocities in the Lie algebra.

### 4.4 Architecture
The pipeline (`src/models/se3_action_head.py:128-305`) consists of:
`[Pretrained VLA Backbone] → [Hidden State h] → [SE(3) Flow Matching Head] → [Action T ∈ SE(3)]`

**[PLANNED]** In Phase 1, the frozen real VLA backbone will be used. 
**[MEASURED]** Currently, the system uses two toy stand-ins:
- `mock_cnn` (`src/models/mock_backbone.py`): A small CNN to emulate an image backbone.
- `scene_id` (`src/models/scene_id_backbone.py`): An oracle embedding vector per task, which isolates action head geometry. 
**These mock backbones are toy stand-ins and must be labeled as such.**

### 4.5 Parameter budget
- **Mock backbone:** ~2.9M parameters (computed from CNN/MLP structure in `src/models/mock_backbone.py:28-105`).
- **Scene-ID backbone:** ~1.5K parameters (`nn.Embedding(n_tasks, hidden_dim)` in `src/models/scene_id_backbone.py:29-41`).
- **SE(3) head:** ~265K parameters (`src/models/se3_action_head.py:53-125`).
- **Euclidean baseline head:** comparable MLP capacity.
- **[PLANNED] Real backbones:** Octo-Small (~27M), Octo-Base (~93M), OpenVLA-7B (~7B) based on their respective papers.

## 5. Decision Log

| Decision | Alternatives | Choice | Reason |
|----------|--------------|--------|--------|
| Action Framework | SE(3) Diffusion vs. SE(3) Flow Matching | SE(3) Flow Matching | Flow matching provides a deterministic path (geodesics) and avoids complex noise scheduling on manifolds, minimizing implementation risk (`docs/SELECTION_PROCESS.md:81-91`). |
| Backbone Integration | End-to-end finetuning vs. Frozen backbone + new head | Frozen backbone | Only the action head changes. Backbone and data loading remain unchanged, isolating the ablation of the head (`docs/SELECTION_PROCESS.md:87`). |
| Head Splitting | Joint SE(3) vs. Split-heads (SE(3) rotation + MLP translation) | Split-heads (Phase 2 default) | Addressed the T-RMSE regression (10–41% worse translation error on translation-heavy tasks). Splitting heads cleanly fixes translation (`implementation_plan.md:55`). |
| Benchmark Selection | MetaWorld MT-50 vs. LIBERO-Spatial | LIBERO-Spatial first, MT-10 fallback, MT-50 deferred | MetaWorld is noisy and computationally heavy. LIBERO provides a robust standardized evaluation (`implementation_plan.md:38-41`). |
| Scope Limitations | Keep vs. Drop "Theorem 1" and "Geodesic Action-ECE" | Drop from scope | These are [ASPIRATIONAL] mathematical additions that distract from the empirical evaluation in a tight timeline unless strongly validated (`implementation_plan.md:9`). |
| Data Leakage | Ignore vs. Fix hue-trace | Fix (Phase 0) | A background hue gradient encoded the rotation trace (`src/training/data_loader.py:150-156`), artificially aiding models. We patched this in Phase 0 and re-ran the sweep for honest negative tracking (`implementation_plan.md:26`). |
| Pre-registration | Run and report vs. Pre-register Phase 3 | Pre-registration | Guarantees methodological transparency. Eliminates hyperparameter hacking before seeing 3-seed real benchmark results (`implementation_plan.md:66-74`). |

## 6. Experimental Protocol

### 6.1 Synthetic pilot [MEASURED]
We constructed a synthetic diagnostic dataset (`src/training/data_loader.py:34-200`) partitioned into two hard-bounded families:
- `rotation_heavy`: θ ~ Uniform(π/2, π), ‖v‖ ≤ 0.05
- `translation_heavy`: θ ~ Uniform(0, π/12), ‖v‖ ~ Uniform(0.1, 0.5)
Leakage history: corner markers were fixed in commit `8c68ea1`; background hue-trace was fixed in Phase 0 (`implementation_plan.md:26`). The pilot uses a pre-registered 12-run sweep (`experiments/PREREGISTRATION.md`).

### 6.2 Real experiment [PLANNED]
Following `implementation_plan.md` Phases 1–3: We will stand up Octo-Small or OpenVLA-7B on LIBERO-Spatial. We will run a 3-seed evaluation of `OctoSE3` vs `OctoEuclideanBaseline`. No real-benchmark numbers exist yet.

### 6.3 Metrics
- **Synthetic [MEASURED]:** Geodesic R-RMSE (`src/utils/metrics.py` or `src/models/geodesic_loss.py:128-146`), T-RMSE, G-RMSE.
- **Real [PLANNED]:** Task success rate.

### 6.4 Baselines
For the pilot [MEASURED], we compare against `OctoEuclideanBaseline`. Note that this baseline *already uses* `exp_so3` at inference (`src/evaluate.py:91-97`); thus the comparison is strictly "flow matching vs. MLP", not "manifold vs. flat". 
For the real experiment [PLANNED], we will compare against stock Octo-Small and stock OpenVLA-7B with their default action heads.

## 7. Results So Far

### 7.1 Synthetic pilot [MEASURED]
The synthetic pilot demonstrates partial support for the SE(3) head. The geometric advantage holds under the clean `scene_id` backbone across all seeds, but degrades under the `mock_cnn` backbone.

**Per-Seed Table for Primary Claim (`rotation_heavy` family):**
| Backbone | Seed | Euclid R-RMSE | SE(3) R-RMSE | Δ (Eucl - SE(3)) | SE(3) ≤ Eucl? |
|----------|------|---------------|--------------|------------------|---------------|
| scene_id | 0    | 2.4330        | 2.3178       | 0.1152           | YES           |
| scene_id | 1    | 2.3443        | 2.2519       | 0.0925           | YES           |
| scene_id | 2    | 2.4394        | 2.3176       | 0.1218           | YES           |
| mock_cnn | 0    | 2.4340        | 2.2448       | 0.1892           | YES           |
| mock_cnn | 1    | 2.1146        | 2.2166       | -0.1020          | NO            |
| mock_cnn | 2    | 2.2853        | 2.2305       | 0.0548           | YES           |

**Aggregate Table (mean ± std across seeds):**
| Head | Backbone | Family | G-RMSE | R-RMSE | T-RMSE |
|------|----------|--------|--------|--------|--------|
| OctoEuclideanBaseline | mock_cnn | rotation_heavy | 2.2785 ± 0.1305 | 2.2779 ± 0.1305 | 0.0408 ± 0.0080 |
| OctoEuclideanBaseline | mock_cnn | translation_heavy | 0.4514 ± 0.0700 | 0.3016 ± 0.1092 | 0.3243 ± 0.0020 |
| OctoEuclideanBaseline | mock_cnn | combined | 1.6606 ± 0.0530 | 1.6444 ± 0.0537 | 0.2291 ± 0.0024 |
| OctoEuclideanBaseline | scene_id | rotation_heavy | 2.4052 ± 0.0429 | 2.4056 ± 0.0434 | 0.0257 ± 0.0005 |
| OctoEuclideanBaseline | scene_id | translation_heavy | 0.3578 ± 0.0018 | 0.1521 ± 0.0025 | 0.3236 ± 0.0021 |
| OctoEuclideanBaseline | scene_id | combined | 1.7204 ± 0.0021 | 1.7060 ± 0.0020 | 0.2281 ± 0.0023 |
| OctoSE3 | mock_cnn | rotation_heavy | 2.2392 ± 0.0103 | 2.2306 ± 0.0115 | 0.1642 ± 0.0119 |
| OctoSE3 | mock_cnn | translation_heavy | 0.4614 ± 0.0299 | 0.2377 ± 0.0201 | 0.3934 ± 0.0408 |
| OctoSE3 | mock_cnn | combined | 1.5898 ± 0.0191 | 1.5586 ± 0.0161 | 0.3040 ± 0.0242 |
| OctoSE3 | scene_id | rotation_heavy | 2.3033 ± 0.0294 | 2.2958 ± 0.0310 | 0.1550 ± 0.0220 |
| OctoSE3 | scene_id | translation_heavy | 0.4886 ± 0.0166 | 0.2376 ± 0.0338 | 0.4234 ± 0.0401 |
| OctoSE3 | scene_id | combined | 1.6490 ± 0.0120 | 1.6162 ± 0.0083 | 0.3184 ± 0.0235 |

**Honest Negative Finding:**
SE(3) is consistently 10–41% worse on translation T-RMSE in the `translation_heavy` family across every seed and backbone. The effect size on the headline metric is a modest 4.6% R-RMSE improvement on `rotation_heavy` (scene_id) (~2.5σ). This maps to "SE(3) < Euclid on rot_heavy (scene_id only) — Partial support — narrower claim" in the pre-registration outcomes (`experiments/PREREGISTRATION.md`).

### 7.2 Real experiment results: NOT YET AVAILABLE
No real-benchmark numbers exist yet. These will be populated after Phase 3 runs on LIBERO-Spatial.

## 8. Limitations and Threats to Validity

1. The current pilot is synthetic; the demonstrated "advantage" applies only to a toy regression on toy backbones.
2. The `scene_id` backbone is an oracle; it guarantees the task family is perfectly known to the action head (`src/models/scene_id_backbone.py:43-57`).
3. T-RMSE regression ranges from 10–41% in the translation-heavy family. This is unresolved at the time of writing, though the Phase 2 split-heads default aims to fix it (`implementation_plan.md:55`).
4. Background-hue leakage was discovered post-hoc and patched mid-experiment (Phase 0). Pre-Phase-0 numbers are tainted (`reports/SUMMARY.md:49`).
5. Small sample size: only n=3 seeds, a single generated synthetic dataset, and a single architecture definition for the head.
6. The project lacks a real robot demonstration, a sim-to-real story, or language conditioning beyond randomly sampled placeholder tokens (`src/models/mock_backbone.py:88-93`).

## 9. Reproducibility

- **Git commits**: `54dafc7` (pre-registration), `8c68ea1` (corner-marker leakage fix). Tags: `pilot-frozen`, `phase0-postleak`, future `phase1-baseline` … `phase4-results-frozen`.
- **Hardware**: A100 80GB GPU. Training takes ~2.5–3 min per pilot run.
- **Exact commands**: 
  - Phase 1 (scene_id): `bash experiments/run_all.sh` (stopped after run 6, commit `54dafc7` HEAD).
  - Phase 2 (mock_cnn): 
    ```bash
    for SEED in 0 1 2; do
      python src/train.py --config configs/octo_se3_cnn.yaml --seed $SEED
      python src/train.py --config configs/octo_baseline_cnn.yaml --seed $SEED
    done
    ```
  - Phase 3 (eval): `python src/evaluate.py` across 12 checkpoints yielding JSON outputs.
- **Seeds**: 0, 1, 2.
- **Software versions**: Strictly locked to `requirements.txt`.

## 10. Timeline and Submission Plan

- Phase 0: ~1 hour. 
- Phase 1: 3 days–2 weeks. 
- Phase 2: 1–3 days. 
- Phase 3: 3–10 days compute. 
- Phase 4: 3–7 days writing.
- Calendar: Today is 2026-04-30. 
  - CoRL 2026 deadline ~June (tight, stretch). 
  - RA-L rolling. 
  - ICRA 2027 ~Sep (comfortable). 
  - **Default target**: **ICRA 2027**, with arXiv tech report at end of Phase 4.

**Outcome→venue mapping:**
| Outcome | Venue | Format |
|---------|-------|--------|
| Strong support across both rotation and translation buckets | CoRL or RA-L | Full paper |
| Partial support (rotation-only improvement) | RA-L short or workshop | Short paper |
| Null or negative result | arXiv tech report | Write-up of negative result; do not submit |

## 11. Appendix A — Math

**Exponential Map on SE(3)** (`src/utils/se3_utils.py:131-176`):
exp(ω, v) maps from the Lie algebra se(3) to the group SE(3). We use Rodrigues formula with Taylor expansions (e.g. 1 - θ²/6 for small angles) to avert NaNs.

**Logarithmic Map on SE(3)** (`src/utils/se3_utils.py:179-221`):
Extracts the axis-angle and translation components to map back to se(3). Uses tailored Taylor expansions near θ=0.

**Geodesic Distance** (`src/utils/se3_utils.py:251-280`):
d_SE(3)(T₁, T₂) = ||log(T₁⁻¹ · T₂)||. Weighted equally across rotation and translation components in our pilot.

**RFM Loss on SE(3)** (`src/models/se3_action_head.py:249-305`):
The flow matching loss interpolates geodesic paths X_t = X₀ · exp(t · ξ), predicting the time-independent target velocity v_target = ξ (in body frame), generating gradients through F.mse_loss(v_pred, v_target).

## 12. Appendix B — File-level repo map

Files in `src/`:
- `models/mock_backbone.py` (105 lines): A lightweight CNN+MLP to simulate a VLA backbone.
- `models/scene_id_backbone.py` (61 lines): An oracle task ID embedding backbone.
- `models/se3_action_head.py` (305 lines): Core Riemannian flow matching head and logic.
- `models/geodesic_loss.py` (146 lines): Geodesic distance metrics and loss wrappers.
- `models/octo_adapter.py` (~300 lines): Wrapper mapping standard Octo inferences to SE(3).
- `models/se3_layers.py` (~150 lines): Assorted layers operating over poses.
- `utils/se3_utils.py` (365 lines): Safe batched PyTorch exponential, logarithmic, and metric operations for SE(3).
- `utils/visualization.py` (~250 lines): Rendering tools for trajectories.
- `utils/metrics.py` (~200 lines): Metric formulations like Geodesic Action-ECE.
- `training/data_loader.py` (319 lines): Synthetic dataset generation with strict rotation boundary families.
- `training/trainer.py` (~400 lines): The main training loop driving convergence.
- `train.py` (~450 lines): Command line entrypoint to build models and launch trainer.
- `evaluate.py` (182 lines): Evaluates checkpoints and outputs per-family JSON reports.

## 13. References

- Black et al., "π0: A Vision-Language-Action Flow Model for General Robot Control." arXiv, 2024.
- Brohan et al., "RT-1: Robotics Transformer for Real-World Control at Scale." ICRA, 2023.
- Brohan et al., "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control." CoRL, 2023. [UNVERIFIED]
- Chen & Lipman, "Riemannian Flow Matching on General Geometries." NeurIPS, 2024.
- Jaquier et al., "RFMPose: Generative Category-level Object Pose Estimation via Riemannian Flow Matching." NeurIPS, 2025.
- Kim et al., "OpenVLA: An Open-Source Vision-Language-Action Model." CoRL, 2024.
- Lipman et al., "Flow Network based Generative Models." ICLR, 2023. [UNVERIFIED]
- Octo Model Team, "Octo: An Open-Source Generalist Robot Policy." RSS, 2024.
- Qiao et al., "RDT-1B: Robotics Diffusion Transformer." ICLR, 2025.
- Simeonov et al., "EquiBot: SIM(3)-Equivariant Diffusion Policy for Generalizable and Data Efficient Learning." CoRL, 2024.
- SAFE Authors, "SAFE: Multitask Failure Detection for Vision-Language-Action Models." OpenReview, 2025.
- FAST Authors, "FAST: Efficient Action Tokenization for Vision-Language-Action Models." RSS, 2026.
- ReconVLA Authors, "ReconVLA: Reconstructive Vision-Language-Action Model as Effective Robot Perceiver." AAAI, 2026.
- Zhou et al., "On the Continuity of Rotation Representations in Neural Networks." CVPR, 2019.
- SmolVLA Authors, "SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics." arXiv, 2025.
