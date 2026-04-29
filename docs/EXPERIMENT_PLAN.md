# EXPERIMENT_PLAN.md — Detailed Experiment Protocol

## Benchmarks

### Primary Benchmarks

| Benchmark | Description | Tasks | Metric |
|-----------|-------------|-------|--------|
| **MetaWorld MT-50** | 50 diverse manipulation tasks on Sawyer arm | 50 | Success rate per task |
| **LIBERO** | 4 suites (Spatial, Object, Goal, Long) | 40 | Success rate per suite |

### Secondary Benchmarks

| Benchmark | Description | Tasks | Metric |
|-----------|-------------|-------|--------|
| **LIBERO-Plus** | LIBERO with visual perturbations | 40 | Success rate under perturbation |
| **SimplerEnv** | Real-to-sim evaluation | varies | Success rate |

## Task Categorization for MetaWorld MT-50

### Rotation-Heavy Tasks (~15 tasks)
These tasks require large rotations (>90°) and are where SE(3) head should help most:

| Task | Expected Rotation | Current Octo SR (est.) | Target SR |
|------|-------------------|----------------------|-----------|
| dial-turn | ~180° | ~35% | ~55% |
| door-unlock | ~90° | ~40% | ~60% |
| door-open | ~90° | ~45% | ~60% |
| door-close | ~90° | ~50% | ~65% |
| faucet-open | ~180° | ~30% | ~50% |
| faucet-close | ~180° | ~35% | ~55% |
| nut-assemble | ~180° | ~25% | ~45% |
| nut-disassemble | ~180° | ~30% | ~50% |
| peg-insert-side | ~90° | ~40% | ~55% |
| peg-unplug-side | ~90° | ~45% | ~60% |
| wrench-pickup | ~90° | ~50% | ~65% |
| hammer-pickup | ~45° | ~55% | ~65% |
| hand-insert | ~90° | ~40% | ~55% |
| window-open | ~90° | ~45% | ~60% |
| window-close | ~90° | ~50% | ~65% |

### Translation-Heavy Tasks (~20 tasks)
These tasks are primarily translational and should show minimal change:

| Task | Expected Rotation | Current Octo SR (est.) | Target SR |
|------|-------------------|----------------------|-----------|
| push-left | < 10° | ~65% | ~65% |
| push-right | < 10° | ~65% | ~65% |
| push-front | < 10° | ~60% | ~60% |
| push-back | < 10° | ~60% | ~60% |
| pick-place | < 30° | ~55% | ~55% |
| reach-left | < 10° | ~70% | ~70% |
| reach-right | < 10° | ~70% | ~70% |
| ... | ... | ... | ... |

## Experiment 1: Main Comparison (Euclidean vs. SE(3))

**Hypothesis**: SE(3) flow matching head improves success rate on rotation-heavy tasks while maintaining performance on translation-heavy tasks.

**Protocol**:
1. Load pretrained Octo-Base checkpoint
2. **Baseline**: Fine-tune with original Euclidean action head on MetaWorld MT-50 training set
3. **SE(3)**: Fine-tune with SE(3) flow matching head on same data
4. Both use same: learning rate schedule, batch size, number of epochs, data augmentation
5. Evaluate on all 50 MetaWorld tasks (100 episodes per task, 3 seeds)

**Metrics**:
- Success rate per task (mean ± std over 3 seeds)
- Success rate grouped by rotation-heavy vs. translation-heavy
- Overall MT-50 average

**Expected Result**:
- Translation-heavy: ~0% difference (SE(3) ≈ Euclidean)
- Rotation-heavy: +10-15% improvement (SE(3) > Euclidean)
- Overall: +3-5% improvement

## Experiment 2: Rotation Magnitude Analysis

**Hypothesis**: The improvement of SE(3) over Euclidean is monotonically correlated with the rotation magnitude in the task.

**Protocol**:
1. For each MT-50 task, compute the average rotation magnitude from demonstrations
2. Plot: x-axis = average rotation magnitude, y-axis = Δ(success rate) = SR_SE3 - SR_Euclidean
3. Fit a linear regression

**Expected Result**: Positive correlation (larger rotation → larger improvement)

**This figure is the paper's key result** — it validates Theorem 1 empirically.

## Experiment 3: Ablation — Architecture Choices

**Hypothesis**: Design choices in the SE(3) head matter.

| Ablation | Variable | Options |
|----------|----------|---------|
| A3.1 | Flow steps at inference | 1, 5, 10, 20, 50 |
| A3.2 | Head hidden dim | 128, 256, 512 |
| A3.3 | Number of layers | 2, 4, 6, 8 |
| A3.4 | Training loss | Geodesic MSE vs. Euclidean MSE on se(3) |
| A3.5 | Source distribution | Gaussian on se(3) vs. Uniform on SE(3) |

**Protocol**: Change one variable at a time, evaluate on rotation-heavy MT-50 subset.

## Experiment 4: Ablation — Parameterization Comparison

**Hypothesis**: SE(3) flow matching outperforms other rotation parameterizations.

| Method | Action Space | Loss |
|--------|-------------|------|
| Axis-angle (baseline) | R⁶ | Euclidean MSE |
| Quaternion | R⁷ (unit norm) | Euclidean MSE |
| Rotation matrix | R⁹ (orthogonal) | Frobenius norm |
| 6D rotation | R⁶ (Gram-Schmidt) | Euclidean MSE |
| **SE(3) flow matching** | se(3) | **Geodesic MSE** |

**Protocol**: Same backbone, same data, same training budget. Compare on rotation-heavy tasks.

**Expected Result**: SE(3) ≥ 6D > quaternion > rotation matrix > axis-angle

## Experiment 5: LIBERO Benchmark

**Hypothesis**: SE(3) head maintains performance on LIBERO (which is saturated).

**Protocol**: Fine-tune on LIBERO, evaluate on all 4 suites.

**Expected Result**: Within 1-2% of Euclidean baseline on all suites (LIBERO doesn't need rotation correction).

**Purpose**: This is a sanity check, not a headline result.

## Experiment 6: Inference Speed Comparison

**Hypothesis**: SE(3) head with single-step inference is comparable in speed to Euclidean head.

**Protocol**: Measure inference time (ms) for:
- Euclidean head (single forward pass)
- SE(3) head, 1 step
- SE(3) head, 5 steps
- SE(3) head, 10 steps
- SE(3) head, 20 steps

**Expected Result**: 1-step SE(3) ~ same speed as Euclidean; 10-step SE(3) ~2× slower.

## Experiment 7: Geodesic Action-ECE (Novel Metric)

**Hypothesis**: Euclidean confidence calibration is miscalibrated on rotations; geodesic Action-ECE is better.

**Protocol**:
1. Train both Euclidean and SE(3) heads
2. For each, compute ensemble disagreement (3 seeds) as uncertainty
3. Compute Action-ECE: calibration error between predicted confidence and actual success
4. Compare Euclidean Action-ECE vs. Geodesic Action-ECE

**Expected Result**: Geodesic Action-ECE < Euclidean Action-ECE (better calibration)

## Experiment 8: Real Robot (If Resources Allow)

**Hypothesis**: SE(3) improvements transfer from simulation to real world.

**Tasks**: 5 representative tasks (2 rotation-heavy, 2 translation-heavy, 1 mixed)
**Protocol**: 20 demonstrations per task, fine-tune, evaluate 20 episodes per task

**This experiment is optional for RA-L, required for T-RO.**

## Evaluation Protocol

### For Each Experiment
- **Seeds**: 3 random seeds for training, report mean ± std
- **Episodes**: 100 episodes per task for evaluation
- **Metrics**: Success rate (primary), geodesic distance error (secondary)
- **Statistical test**: Paired t-test between Euclidean and SE(3) results

### Significance Criteria
- p < 0.05 for individual task comparisons
- p < 0.01 for aggregated comparisons (rotation-heavy group)

## Compute Requirements

| Experiment | Compute | Time |
|-----------|---------|------|
| Exp 1 (Main) | 2 A100-days | 2 days |
| Exp 2 (Rotation analysis) | 0 (uses Exp 1 results) | 0 |
| Exp 3 (Ablation architecture) | 3 A100-days | 3 days |
| Exp 4 (Ablation parameterization) | 2 A100-days | 2 days |
| Exp 5 (LIBERO) | 1 A100-day | 1 day |
| Exp 6 (Speed) | 0.1 A100-day | 1 hour |
| Exp 7 (Action-ECE) | 1 A100-day | 1 day |
| Exp 8 (Real robot) | N/A | 2-3 days |
| **Total** | **~9 A100-days** | **~9 days** |
