# Novelty Analysis & Competitive Landscape

## Prior Work Comparison

### Braun et al. (RFMP, IROS 2024) — Closest Competitor
- **What**: Riemannian flow matching on SE(3) for single-pose robot policies
- **Overlap**: Same SE(3) math (exp/log maps, geodesic interpolation)
- **Gap**: Single pose only — no action chunking, no uncertainty, not integrated into a VLA
- **Our advantage**: Geodesic chunking (K→H) + conformal prediction on SE(3) + VLA integration

### SmolVLA (HuggingFace, 2025) — Backbone
- **What**: 450M PyTorch VLA with Euclidean flow matching action head
- **Why we use it**: Under 500M budget, pretrained, PyTorch-native, already uses flow matching
- **Our modification**: Replace Euclidean flow expert with SE(3) flow matching + chunking + uncertainty

### Other VLAs
| Model | Params | Action Space | Flow Matching | Uncertainty |
|-------|--------|-------------|---------------|-------------|
| Octo | 93M | Euclidean R⁷ | Diffusion | ❌ |
| OpenVLA | 7B | Euclidean R⁷ | ❌ (autoregressive) | ❌ |
| π0 | ~3B | Euclidean R⁷ | Flow matching | ❌ |
| SmolVLA | 450M | Euclidean R⁷ | Flow matching | ❌ |
| **Ours** | **455M** | **SE(3)** | **Riemannian flow** | **✅ Conformal** |

## Three Novel Contributions

### 1. Geodesic Action Chunking
- **What**: Predict K anchor poses on SE(3), interpolate H actions via geodesics
- **Why novel**: No one has done chunk-wise prediction on SE(3). Existing VLAs predict H independent Euclidean vectors. RFMP predicts single poses.
- **Benefit**: Temporal consistency for free — no antipodal boundary crossings

### 2. Uncertainty-Aware Flow Matching on SE(3)
- **What**: N-sample flow matching with Fréchet mean + geodesic variance
- **Why novel**: First VLA with principled uncertainty on SE(3). Existing VLAs are point estimators.
- **Benefit**: Robot knows when it's uncertain → safe fallback

### 3. Conformal Prediction on SE(3)
- **What**: Calibrated prediction sets {T : d(T, T̄) ≤ q_α} with distribution-free coverage
- **Why novel**: No prior work on conformal prediction for VLA action spaces on manifolds
- **Benefit**: Mathematical guarantee: P(T* ∈ C_α) ≥ 1 - α

## Differentiation from RFMP

| Aspect | RFMP (Braun 2024) | Ours |
|--------|-------------------|------|
| Manifold | SE(3) | SE(3) |
| Flow matching | ✅ | ✅ |
| Action chunking | ❌ Single pose | ✅ K anchors → H geodesic |
| Uncertainty | ❌ Point estimate | ✅ N-sample + conformal |
| VLA integration | ❌ Standalone policy | ✅ SmolVLA backbone |
| Benchmark | Custom tasks | LIBERO, Meta-World |

## What We're NOT Claiming

- We do NOT claim to beat SmolVLA's end-to-end success rate
- We do NOT claim SOTA on LIBERO
- We DO claim: better rotation accuracy, temporal consistency, calibrated uncertainty
