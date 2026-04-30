# PAPER_OUTLINE.md — Paper Structure and Writing Plan

## Target Venues (Priority Order)

| Venue | Deadline | Fit | Notes |
|-------|----------|-----|-------|
| **RA-L + ICRA 2027** | ~Sep 2026 | Best fit | Real robot optional for RA-L; ICRA presentation |
| **CoRL 2026** | ~Jun 2026 | Strong | Top robotics learning conference |
| **T-RO** | Rolling | Strong | Needs real robot + stronger theory |
| **NeurIPS 2026** | ~May 2026 | Good | Needs more ML contribution |

**Recommended**: RA-L + ICRA 2027 (most achievable timeline)

## Title Options

1. **SE(3)-VLA: Riemannian Flow Matching on Lie Groups for Vision-Language-Action Models** ← preferred
2. **Geodesic Action Prediction: Replacing Euclidean Approximation with SE(3) Flow Matching in VLAs**
3. **Why Your VLA Fails on Rotations: A Geometric Fix via Riemannian Flow Matching**

## Abstract (Draft)

> TBD — pending Phase 4 results. Abstract will be written after `reports/MAIN_RESULTS.md` is complete.
> No number may appear here that is not in a table in `reports/MAIN_RESULTS.md` or `reports/ABLATIONS.md`.

## Paper Structure (8 pages + references for RA-L)

### Section 1: Introduction (~1 page)

**Paragraph 1 — Hook**:
- VLA models are the dominant paradigm for generalist robot policies
- 164 VLA papers at ICLR 2026 alone
- All predict actions in flat Euclidean space

**Paragraph 2 — Problem**:
- Rotations live on SO(3), a non-Euclidean manifold
- Euclidean approximation fails at large rotations (Theorem 1)
- Empirical evidence: TBD — pending Phase 4 results

**Paragraph 3 — Gap**:
- No published VLA respects the SE(3) geometry of actions
- Riemannian flow matching exists (Chen et al., NeurIPS 2024) but hasn't been applied to VLA actions
- RFMPose does SE(3) flow for pose estimation, not VLA action generation

**Paragraph 4 — Our Approach**:
- SE(3)-VLA: replace Euclidean action head with Riemannian flow matching on SE(3)
- Geodesic interpolation, tangent space prediction, bi-invariant metric

**Paragraph 5 — Contributions**:
1. First VLA with Riemannian flow matching on SE(3) for action generation
2. Theorem 1: bounded error of Euclidean approximation (predicts empirical gains)
3. Geodesic Action-ECE: first calibration metric for manifold-valued predictions
4. TBD — empirical improvement number pending Phase 4 results

### Section 2: Related Work (~1 page)

**VLA Models**: RT-2, Octo, OpenVLA, π0, SmolVLA, RDT-1B, FAST
- Position: "All use Euclidean action heads. We show this is suboptimal for rotations."

**Flow Matching**: Lipman et al., Chen et al. (NeurIPS 2024), π0
- Position: "Flow matching on manifolds is proven but unapplied to VLA actions."

**SE(3) Methods**: RFMPose, EquiBot, SE(3)-Diffuser
- Position: "These do SE(3) for pose estimation or grasp generation, not for VLA action prediction."

**Rotation Representations**: 6D rotation (Zhou et al.), quaternion, axis-angle
- Position: "These are fixed parameterizations. We learn the flow on the manifold itself."

**Safety/UQ for VLAs**: SAFE, ReconVLA, calibrated ensembles
- Position: "Complementary — our Geodesic Action-ECE extends calibration to manifold-valued predictions."

### Section 3: Background (~0.5 page)

- SE(3) definition, Lie algebra, exp/log maps
- Flow matching framework (brief, cite Chen et al.)
- The Euclidean action head in current VLAs

### Section 4: Method (~2 pages)

**4.1 Problem Formulation**
- VLA as conditional flow on SE(3)
- Input: visual observation + language instruction → hidden state h
- Output: action T ∈ SE(3) + gripper ∈ [0,1]

**4.2 Riemannian Flow Matching on SE(3)**
- Flow matching loss on SE(3) (Eq. from TECHNICAL_DETAILS.md)
- Geodesic interpolation
- Tangent space velocity prediction

**4.3 Architecture**
- Figure: full architecture diagram
- Flow head: MLP with SE(3) log coordinates + time as input
- Integration with frozen VLA backbone

**4.4 Theorem 1: Bounded Euclidean Error**
- Statement
- Proof sketch (full proof in appendix)
- Interpretation: predicts improvement on rotation-heavy tasks

### Section 5: Experiments (~2.5 pages)

**5.1 Setup**
- MetaWorld MT-50 (primary), LIBERO (secondary)
- Baseline: Octo with Euclidean action head
- Implementation details

**5.2 Main Results (Experiment 1)**
- Table: Success rate on MT-50 (rotation-heavy, translation-heavy, overall)
- TBD — pending Phase 4 results

**5.3 Rotation Magnitude Analysis (Experiment 2)**
- **KEY FIGURE**: Scatter plot of rotation magnitude vs. Δ(success rate)
- TBD — pending Phase 4 results; may or may not validate Theorem 1

**5.4 Ablation Studies (Experiments 3-4)**
- Architecture ablation (flow steps, hidden dim, layers)
- Parameterization ablation (axis-angle vs. quaternion vs. 6D vs. SE(3))

**5.5 LIBERO Results (Experiment 5)**
- Table: TBD — pending Phase 4 results

**5.6 Inference Speed (Experiment 6)**
- Table: Speed comparison across flow steps

**5.7 Geodesic Action-ECE (Experiment 7)**
- Table: Calibration comparison

### Section 6: Discussion (~0.3 page)

- Why geometry matters more than scale for rotations
- Limitations: translation-heavy tasks don't benefit; additional compute for multi-step inference
- Future: combine with conformal prediction for safety guarantees

### Section 7: Conclusion (~0.2 page)

- Summary of contributions
- SE(3) as a principled replacement for Euclidean action heads

### Appendix

- Full proof of Theorem 1
- Additional ablations
- Real robot experiments (if available)
- SE(3) implementation details

## Key Figures

| # | Content | Section |
|---|---------|---------|
| 1 | Architecture diagram (backbone + SE(3) head) | 4.3 |
| 2 | Geodesic vs. Euclidean interpolation on SO(3) | 4.2 |
| 3 | **Rotation magnitude vs. Δ(success rate)** — KEY FIGURE | 5.3 |
| 4 | Ablation: flow steps vs. success rate | 5.4 |
| 5 | Geodesic Action-ECE comparison | 5.7 |

## Key Tables

| # | Content | Section |
|---|---------|---------|
| 1 | Main results: MT-50 success rate (rotation-heavy, translation-heavy, overall) | 5.2 |
| 2 | Ablation: parameterization comparison | 5.4 |
| 3 | LIBERO results | 5.5 |
| 4 | Inference speed comparison | 5.6 |
| 5 | Geodesic Action-ECE | 5.7 |

## Writing Schedule

| Week | Task |
|------|------|
| Week 1 | Sections 1-3 (Intro, Related Work, Background) |
| Week 2 | Section 4 (Method) + figures |
| Week 3 | Section 5 (Experiments) — after results are in |
| Week 4 | Sections 6-7 + appendix + revision |
