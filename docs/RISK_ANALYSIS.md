# RISK_ANALYSIS.md — What Could Go Wrong + Mitigations

## Risk Matrix

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|------------|
| R1 | SE(3) head doesn't improve on rotation-heavy tasks | Low-Medium | High | Fall back to ablation-only paper; position as "first SE(3) VLA" without performance claims |
| R2 | Theorem 1 proof has a gap | Medium | Medium | Prove on SO(3) only (simpler); still novel for VLA |
| R3 | Numerical instability in exp/log maps | Medium | Low | Use `geoopt`/`theseus` (battle-tested); add unit tests |
| R4 | Training instability (flow matching doesn't converge) | Low-Medium | High | Use smaller learning rate; increase flow steps; try both Euler and midpoint integration |
| R5 | Someone publishes "Riemannian Flow Matching for VLA" first | Low | High | Accelerate timeline; lead with empirical ablation (which they may not have); submit to arXiv immediately |
| R6 | Octo backbone is too weak for meaningful results | Low | Medium | Replicate with OpenVLA or SmolVLA; the method is model-agnostic |
| R7 | MetaWorld MT-50 results are noisy | Medium | Medium | Use 100 episodes per task, 3 seeds; report confidence intervals |
| R8 | Compute budget exceeded | Low | Medium | Reduce ablation scope; prioritize Experiment 1 (main comparison) |
| R9 | Reviewer says "incremental over RFMPose" | Medium | Medium | Clear positioning: RFMPose is for pose estimation, not VLA actions; different loss, different setting |
| R10 | Real robot experiments fail to transfer | Medium | Low | Not required for RA-L; include as future work |

## Detailed Risk Analysis

### R1: No Performance Improvement (Biggest Risk)

**Scenario**: SE(3) head trains successfully but doesn't outperform Euclidean on rotation-heavy tasks.

**Why this might happen**:
- Octo's backbone features may not provide enough rotation-relevant information
- MetaWorld tasks may have enough demonstrations that Euclidean parameterization is "good enough"
- The rotation magnitudes in MetaWorld may not be large enough for the O(θ²) error to matter

**Mitigation**:
1. Compute the actual rotation magnitudes from demonstrations BEFORE training. If average θ < 60°, the theoretical gap is small.
2. If the gap is small, reposition the paper as "SE(3)-VLA: A Principled Geometric Framework for VLA Action Prediction" — focus on the framework contribution, not the performance number.
3. Add LIBERO-Plus (with perturbations) where the geometric robustness may matter more.

**Decision trigger**: If rotation-heavy tasks show < 5% improvement after full hyperparameter sweep, pivot to framework-only paper.

### R2: Theorem 1 Proof Gap

**Scenario**: The O(θ²) bound has additional assumptions that don't hold in practice.

**Why this might happen**:
- The bound assumes optimal flow matching, but in practice we have finite capacity
- The constant C depends on the data distribution and may be large

**Mitigation**:
1. Prove the theorem for SO(3) first (simpler, 3D rotation group). The full SE(3) theorem can be stated as a corollary.
2. If the proof is too complex, present it as a conjecture with empirical validation. The empirical correlation (Experiment 2) is the more important result anyway.
3. Include the proof in the appendix; keep the main paper focused on empirical results.

### R3: Numerical Instability

**Scenario**: exp/log maps produce NaN or Inf at θ ≈ 0 or θ ≈ π.

**Why this might happen**:
- Division by θ in the Rodrigues formula when θ ≈ 0
- Loss of precision in sin(θ) when θ ≈ π

**Mitigation**:
1. Use Taylor expansions for small θ (standard in robotics libraries)
2. Use `geoopt` or `theseus` which handle these edge cases
3. Add explicit clamping: θ = clamp(θ, ε, π-ε) where ε = 1e-6
4. Write unit tests for known SE(3) identities: exp(log(T)) = T, log(exp(ξ)) = ξ

### R4: Training Instability

**Scenario**: Flow matching loss doesn't converge or oscillates.

**Why this might happen**:
- Learning rate too high for the manifold structure
- Source distribution mismatch (Gaussian on se(3) may not be a good source)
- Gradient explosion due to exp/log map Jacobians

**Mitigation**:
1. Start with very small learning rate (1e-5) and cosine annealing
2. Use gradient clipping (max_norm=1.0)
3. Try both Euler and midpoint integration for the flow
4. Monitor the geodesic distance between predicted and target actions during training
5. If flow matching is unstable, fall back to direct regression on se(3): L = ||log(X_pred⁻¹ · X_target)||²

### R5: Scooped

**Scenario**: A paper on "Riemannian Flow Matching for VLA" appears on arXiv.

**Why this might happen**:
- The RFMPose group (Jaquier et al.) could extend to VLA
- The Chen et al. (NeurIPS 2024) group could apply their framework to robotics

**Mitigation**:
1. Submit to arXiv as soon as Experiments 1-2 are done (even before full paper is written)
2. If scooped, differentiate by:
   - Theorem 1 (they may not have the error bound)
   - Geodesic Action-ECE (they may not have the calibration metric)
   - Comprehensive ablation (they may not compare parameterizations)
3. The "works with any pretrained VLA" angle is always novel if they build a custom architecture

## Escape Hatches

### Escape 1: If SE(3) head doesn't improve performance
→ Pivot to "Geodesic Action-ECE" paper: introduce the calibration metric, show all existing VLAs are miscalibrated on rotations, propose SE(3) as the fix for calibration (not for success rate).

### Escape 2: If Theorem 1 proof is too hard
→ Drop the theory, keep the empirical contribution. Position as "SE(3) Flow Matching for VLA: An Empirical Study." Still publishable at CoRL/ICRA.

### Escape 3: If compute is insufficient
→ Reduce to Experiments 1, 2, 5 only (main comparison, rotation analysis, LIBERO sanity check). Still a complete paper.

### Escape 4: If scooped on arXiv
→ Lead with the ablation study comparing parameterizations (Experiment 4). This is the most comprehensive comparison and unlikely to be in the scooping paper.
