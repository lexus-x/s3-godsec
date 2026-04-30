# Graph Report - /home/user/Desktop/vla_projects/q1/s3-godsec  (2026-04-30)

## Corpus Check
- 22 files · ~40,829 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 222 nodes · 344 edges · 20 communities detected
- Extraction: 72% EXTRACTED · 28% INFERRED · 0% AMBIGUOUS · INFERRED: 97 edges (avg confidence: 0.65)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]

## God Nodes (most connected - your core abstractions)
1. `SE3ActionPredictor` - 21 edges
2. `OctoSE3` - 20 edges
3. `OctoEuclideanBaseline` - 19 edges
4. `SceneIDBackbone` - 15 edges
5. `MockOctoBackbone` - 15 edges
6. `main()` - 10 edges
7. `evaluate_loader()` - 10 edges
8. `exp_se3()` - 10 edges
9. `SyntheticSE3Dataset` - 10 edges
10. `validate_single()` - 9 edges

## Surprising Connections (you probably didn't know these)
- `SE3ActionPredictor` --uses--> `Count total and trainable parameters.`  [INFERRED]
  /home/user/Desktop/vla_projects/q1/s3-godsec/src/models/se3_action_head.py → /home/user/Desktop/vla_projects/q1/s3-godsec/src/models/octo_adapter.py
- `train_one_epoch()` --calls--> `log_so3()`  [INFERRED]
  /home/user/Desktop/vla_projects/q1/s3-godsec/src/train.py → /home/user/Desktop/vla_projects/q1/s3-godsec/src/utils/se3_utils.py
- `validate_single()` --calls--> `exp_so3()`  [INFERRED]
  /home/user/Desktop/vla_projects/q1/s3-godsec/src/train.py → /home/user/Desktop/vla_projects/q1/s3-godsec/src/utils/se3_utils.py
- `validate_single()` --calls--> `geodesic_rmse()`  [INFERRED]
  /home/user/Desktop/vla_projects/q1/s3-godsec/src/train.py → /home/user/Desktop/vla_projects/q1/s3-godsec/src/utils/metrics.py
- `validate_single()` --calls--> `rotation_rmse()`  [INFERRED]
  /home/user/Desktop/vla_projects/q1/s3-godsec/src/train.py → /home/user/Desktop/vla_projects/q1/s3-godsec/src/utils/metrics.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.12
Nodes (24): build_model(), make_backbone(), Evaluation script for SE(3)-VLA controlled experiments.  Reports per-family metr, Create backbone based on config., Build model from config., Evaluate on a single dataloader., GeodesicMSELoss, Geodesic MSE loss on SE(3).          L = ||log(X_pred^{-1} * X_target)||² (+16 more)

### Community 1 - "Community 1"
Cohesion: 0.09
Nodes (15): Octo VLA Adapter — plugs SE(3) action head into pretrained Octo.  This module wr, Predict a chunk of future actions on SE(3).                  Args:             o, Compute training loss.                  Args:             observations: dict wit, Octo VLA with SE(3) flow matching action head.          Architecture:         [P, Return only the trainable parameters (SE(3) head)., Count total and trainable parameters., Octo with Euclidean action head (baseline for comparison).          This is the, Args:             octo_model: Pretrained Octo model instance             hidden_ (+7 more)

### Community 2 - "Community 2"
Cohesion: 0.14
Nodes (19): Args:             X_pred: [B, 4, 4] Predicted SE(3) poses             X_target:, Compute the Riemannian flow matching loss on SE(3).                  The loss is, exp_se3(), exp_so3(), geodesic_interpolation(), hat(), inverse_se3(), log_se3() (+11 more)

### Community 3 - "Community 3"
Cohesion: 0.13
Nodes (13): CombinedSE3Dataset, create_dataloaders(), Synthetic SE(3) demonstration dataset for controlled experiments.  Generates syn, Generate synthetic images correlated with target action.          Encodes rotati, Combines rotation_heavy and translation_heavy datasets.      Used for training (, Create train and validation dataloaders.      Training uses a combined dataset (, Sample n unit vectors uniformly on S²., Synthetic dataset with clean bimodal rotation/translation families.      Each sa (+5 more)

### Community 4 - "Community 4"
Cohesion: 0.1
Nodes (14): FlowMatchingLoss, GeodesicDistanceLoss, GeodesicHuberLoss, Geodesic loss functions for SE(3) action prediction., Riemannian flow matching loss on SE(3).          This is the primary training lo, Args:             v_pred: [B, 6] Predicted velocities in se(3)             v_tar, Loss on the rotation angle only (ignoring translation).          Useful for abla, Args:             X_pred: [B, 4, 4] Predicted SE(3) poses             X_target: (+6 more)

### Community 5 - "Community 5"
Cohesion: 0.13
Nodes (18): evaluate_loader(), load_config(), main(), coverage_metric(), geodesic_action_ece(), geodesic_rmse(), Metrics for evaluating SE(3) action prediction.  Includes: - Geodesic Action-ECE, Coverage metric for conformal prediction on SE(3).          Measures whether the (+10 more)

### Community 6 - "Community 6"
Cohesion: 0.13
Nodes (10): GeodesicAttention, SE(3)-specific neural network layers.  Provides layers that operate directly on, Linear layer that operates on SE(3) tangent space.          Maps se(3) vectors t, Args:             xi: [B, ..., 6] se(3) vectors         Returns:             out, Layer normalization for se(3) vectors.          Normalizes rotation and translat, Args:             xi: [B, 6] se(3) vectors (first 3: rotation, last 3: translati, Attention mechanism that uses geodesic distance as bias.          For attending, Args:             h: [B, N, hidden_dim] hidden states             poses: [B, N, (+2 more)

### Community 7 - "Community 7"
Cohesion: 0.16
Nodes (9): SE(3) Flow Matching Action Head for Vision-Language-Action Models.  This module, Predict velocity in se(3) conditioned on VLA state and current pose., Args:             hidden_dim: VLA hidden state dimension             head_hidden, Sinusoidal positional encoding for the time parameter t ∈ [0, 1]., Args:             t: [B, 1] time values in [0, 1]         Returns:             e, Riemannian flow matching head on SE(3).          Predicts a velocity field in th, Args:             hidden_dim: Dimension of the VLA's hidden state             he, SE3FlowHead (+1 more)

### Community 8 - "Community 8"
Cohesion: 0.18
Nodes (9): compute_rotation_magnitude_from_demos(), generate_scatter_plot_data(), Ablation: Rotation magnitude analysis.  This generates the KEY FIGURE for the pa, Compute average rotation magnitude from demonstration data.          Args:, Generate data for the key scatter plot.          X-axis: average rotation magnit, Compute success rate binned by rotation magnitude.          This is the key anal, success_rate_per_rotation_bin(), Extract the rotation angle from an SE(3) matrix.          Args:         T: [B, 4 (+1 more)

### Community 9 - "Community 9"
Cohesion: 0.31
Nodes (6): Count total and trainable parameters., load_config(), main(), train_one_epoch(), validate_all_families(), validate_single()

### Community 10 - "Community 10"
Cohesion: 0.2
Nodes (9): plot_ablation_flow_steps(), plot_geodesic_ece_comparison(), plot_rotation_vs_delta_sr(), plot_success_rate_comparison(), Visualization utilities for SE(3)-VLA experiments.  Generates figures for the pa, Line plot of flow integration steps vs success rate.          Args:         n_st, Calibration plot comparing Euclidean and SE(3) Action-ECE.          Args:, KEY FIGURE: Scatter plot of rotation magnitude vs Δ(success rate).          Vali (+1 more)

### Community 11 - "Community 11"
Cohesion: 0.4
Nodes (3): Ablation: Euclidean vs SE(3) action head on rotation-heavy tasks.  This is the K, Run experiment for a given model type and task set.          Args:         model, run_experiment()

### Community 12 - "Community 12"
Cohesion: 1.0
Nodes (0): 

### Community 13 - "Community 13"
Cohesion: 1.0
Nodes (0): 

### Community 14 - "Community 14"
Cohesion: 1.0
Nodes (0): 

### Community 15 - "Community 15"
Cohesion: 1.0
Nodes (0): 

### Community 16 - "Community 16"
Cohesion: 1.0
Nodes (0): 

### Community 17 - "Community 17"
Cohesion: 1.0
Nodes (0): 

### Community 18 - "Community 18"
Cohesion: 1.0
Nodes (0): 

### Community 19 - "Community 19"
Cohesion: 1.0
Nodes (0): 

## Knowledge Gaps
- **75 isolated node(s):** `Ablation: Rotation magnitude analysis.  This generates the KEY FIGURE for the pa`, `Compute average rotation magnitude from demonstration data.          Args:`, `Generate data for the key scatter plot.          X-axis: average rotation magnit`, `Ablation: Euclidean vs SE(3) action head on rotation-heavy tasks.  This is the K`, `Run experiment for a given model type and task set.          Args:         model` (+70 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 12`** (1 nodes): `parse.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 13`** (1 nodes): `auto_eval.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 14`** (1 nodes): `make_phase_a_configs.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 15`** (1 nodes): `test_octo.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 16`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 17`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 18`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 19`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `SE3ActionPredictor` connect `Community 1` to `Community 0`, `Community 9`, `Community 2`, `Community 7`?**
  _High betweenness centrality (0.157) - this node is a cross-community bridge._
- **Why does `evaluate_loader()` connect `Community 5` to `Community 0`, `Community 1`, `Community 2`?**
  _High betweenness centrality (0.094) - this node is a cross-community bridge._
- **Why does `exp_se3()` connect `Community 2` to `Community 1`, `Community 3`?**
  _High betweenness centrality (0.092) - this node is a cross-community bridge._
- **Are the 14 inferred relationships involving `SE3ActionPredictor` (e.g. with `OctoSE3` and `OctoEuclideanBaseline`) actually correct?**
  _`SE3ActionPredictor` has 14 INFERRED edges - model-reasoned connections that need verification._
- **Are the 11 inferred relationships involving `OctoSE3` (e.g. with `Training script for SE(3)-VLA controlled experiments.  Supports: - SE(3) flow ma` and `Create backbone based on config['model']['backbone_kind'].`) actually correct?**
  _`OctoSE3` has 11 INFERRED edges - model-reasoned connections that need verification._
- **Are the 11 inferred relationships involving `OctoEuclideanBaseline` (e.g. with `Training script for SE(3)-VLA controlled experiments.  Supports: - SE(3) flow ma` and `Create backbone based on config['model']['backbone_kind'].`) actually correct?**
  _`OctoEuclideanBaseline` has 11 INFERRED edges - model-reasoned connections that need verification._
- **Are the 10 inferred relationships involving `SceneIDBackbone` (e.g. with `Training script for SE(3)-VLA controlled experiments.  Supports: - SE(3) flow ma` and `Create backbone based on config['model']['backbone_kind'].`) actually correct?**
  _`SceneIDBackbone` has 10 INFERRED edges - model-reasoned connections that need verification._