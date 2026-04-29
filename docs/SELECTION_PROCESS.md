# SELECTION_PROCESS.md — How We Selected This Research Direction

## Overview

This document records the systematic process by which we evaluated multiple VLA research directions and selected **SE(3) Geodesic Action Head** as the committed direction. This is preserved for transparency and for the paper's "related work" and "positioning" sections.

## Candidates Evaluated

We identified and evaluated 8 research directions across 5 stages of analysis.

### Stage 1: Unsolved Problems in VLA

We surveyed the VLA literature (April 2026) and identified the following unsolved problems:

| Problem | Status | Gap Size |
|---------|--------|----------|
| Rotation representation in action space | **Unsolved** | Large |
| Calibrated uncertainty for VLA actions | Partially addressed (SAFE, ReconVLA) | Medium |
| Visual redundancy in VLA inference | Partially addressed (VLA-Cache, ADP) | Small |
| Adaptive control frequency | **Unsolved** | Medium |
| Multi-task generalization | Active research | Crowded |
| Action speed (single-step inference) | Active research | Crowded |

### Stage 2: Candidate Architectures

| ID | Candidate | Core Idea |
|----|-----------|-----------|
| P1 | PRISM-VLA (DVE+PACE+EAS) | Differential visual encoding + phase-aware attention + eigenspace actions |
| P2 | GAUSS-VLA | Riemannian flow matching on SE(3) + calibrated ensemble uncertainty |
| P3 | ET-VLA | Event-triggered adaptive control frequency |
| P4 | DCFH | Consistency distillation for single-step VLA action decoding |
| P5 | Conformalized SE(3) Action Sets | Post-hoc conformal prediction with geodesic nonconformity scores |

### Stage 3: Scoop Check (April 29, 2026)

| Candidate | Scooped By | Status |
|-----------|-----------|--------|
| PRISM-VLA: DVE | VLA-Cache, ADP, VLA-Pruner, Sliding-Cache VLA | **KILLED** |
| PRISM-VLA: PACE | Underpowered (30K params steering 256M backbone) | **KILLED** |
| PRISM-VLA: EAS | PCA on axis-angle is mathematically broken (double-cover) | **KILLED** |
| GAUSS-VLA (full) | RFMP exists for pose estimation, not VLA; UQ for VLA is crowded | **ALIVE but complex** |
| ET-VLA | StreamingVLA does adaptive observation, not adaptive rate | **ALIVE** |
| DCFH | No direct competitor found | **ALIVE** |
| Conformalized SE(3) | ReconVLA does Euclidean CP; Baheri does S²; nobody does SE(3) | **ALIVE** |

### Stage 4: Adversarial Review

Each surviving candidate was subjected to "Reviewer 2" attacks:

**DCFH:**
- Mode collapse risk on multi-modal tasks
- 10× speedup claim is misleading (honest: 3-5×)
- Consistency distillation is empirical, not provable

**Conformalized SE(3):**
- Does NOT improve task success rate (provides guarantees, not performance)
- Continuous action set representation is non-trivial
- May be perceived as "just CP applied to VLA"

**ET-VLA:**
- Less theoretical depth
- Harder to evaluate (needs interruptible VLA wrapper)
- Less connection to existing VLA safety literature

**SE(3) Geodesic Action Head:**
- Requires training (not zero-shot)
- SE(3) flow matching implementation is non-trivial
- Need to prove Theorem 1 empirically

### Stage 5: Final Decision Matrix

| Criterion (weight) | DCFH | CP-SE(3) | ET-VLA | **SE(3) Head** |
|---------------------|------|----------|--------|----------------|
| Novelty (25%) | 7/10 | 7/10 | 9/10 | **8/10** |
| Performance gain (25%) | 8/10 | 2/10 | 5/10 | **8/10** |
| Implementation risk (20%) | 6/10 | 9/10 | 6/10 | **7/10** |
| Scoop risk (15%) | 8/10 | 7/10 | 9/10 | **8/10** |
| Q1 venue fit (15%) | 7/10 | 7/10 | 6/10 | **8/10** |
| **Weighted Score** | **7.05** | **6.10** | **6.85** | **7.85** |

## Why SE(3) Action Head Won

1. **Only option with BOTH novelty AND performance gain**: Conformal SE(3) is novel but doesn't improve success rate. DCFH improves speed but doesn't change the action representation. SE(3) head is novel AND directly fixes a measurable failure mode.

2. **Cleanest theory→experiment pipeline**: Theorem 1 (bounded Euclidean error) predicts empirical gains on rotation-heavy tasks. T-RO reviewers reward this pattern.

3. **Minimal modification to existing codebase**: Only the action head changes. Backbone, training pipeline, data loading — all unchanged. This is the lowest-risk architectural change.

4. **Composable with other ideas**: The SE(3) head can later be wrapped with conformal prediction (for safety) or combined with consistency distillation (for speed). It's a foundational change, not a one-off trick.

5. **Clear ablation story**: Swap one component (Euclidean head → SE(3) head), compare on rotation-heavy vs. translation-heavy tasks. The ablation is surgical.

## What Was Rejected and Why

| Rejected | Reason |
|----------|--------|
| PRISM-VLA (DVE+PACE+EAS) | DVE scooped, PACE underpowered, EAS mathematically broken |
| GAUSS-VLA (full 3-head ensemble) | Too complex for first paper; 13 A100-days; calibration angle now crowded |
| Conformalized SE(3) alone | No performance improvement; ReconVLA partially scoops the CP-for-VLA angle |
| ET-VLA alone | Niche evaluation; less connection to mainstream VLA benchmarks |
| DCFH alone | Mode collapse risk; consistency distillation is empirical |

## Decision Triggers (When to Re-evaluate)

1. **If a "Riemannian Flow Matching for VLA" paper appears on arXiv** → accelerate submission, lead with the empirical ablation (which they may not have)
2. **If Theorem 1 has a known counterexample** → fall back to empirical comparison only, drop the theoretical headline
3. **If rotation-heavy MetaWorld tasks are already >70% for Octo** → the gap is smaller than estimated; reduce scope

## Decision Record

- **Date**: April 29, 2026
- **Decision**: COMMIT to SE(3) Geodesic Action Head
- **Target venue**: RA-L + ICRA 2027 (primary), T-RO (secondary)
- **Compute budget**: 3-5 A100-days
- **Timeline**: 4-6 weeks to submission
