# Research Strategy & Execution Plan

## Goal
Publish a paper on SE(3) action heads for VLAs with geodesic chunking and conformal uncertainty. Target: CoRL workshop, RA-L short, or similar.

## Why This Works as a Paper

1. **Clean A/B test**: Same backbone (SmolVLA-450M), same data, same budget — only the action head geometry differs
2. **Three contributions**: SE(3) flow + geodesic chunking + conformal uncertainty
3. **Honest framing**: "Geometry matters for rotation" — not "we beat SOTA"
4. **Reproducible**: Open-source, open-data, under 500M budget

## Execution Phases

### Phase 1: SmolVLA Baseline (1-2 days)
- [ ] Install lerobot + SmolVLA
- [ ] Load pretrained SmolVLA-450M
- [ ] Run inference on LIBERO-Spatial baseline
- [ ] Record baseline success rate + rotation/translation RMSE

### Phase 2: SE(3) Head Integration (2-3 days)
- [ ] Replace SmolVLA's action expert with our SE(3) flow head
- [ ] Freeze backbone, train SE(3) head on LIBERO
- [ ] A/B test: Euclidean vs SE(3) on rotation RMSE
- [ ] Record results

### Phase 3: Geodesic Chunking (1-2 days)
- [ ] Integrate geodesic chunking head
- [ ] Train and evaluate temporal smoothness
- [ ] Compare: independent vs chunked prediction
- [ ] Record results

### Phase 4: Uncertainty + Conformal (1-2 days)
- [ ] Integrate uncertainty-aware flow head
- [ ] Calibrate conformal prediction on held-out set
- [ ] Evaluate coverage and variance-error correlation
- [ ] Record results

### Phase 5: Full Experiment + Paper (3-5 days)
- [ ] Run 3 seeds × 4 configurations on LIBERO
- [ ] Generate results tables and figures
- [ ] Write paper (8 pages)
- [ ] Submit to workshop / arXiv

## Total Timeline: 2-3 weeks

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| SmolVLA API changes | Pin to specific commit/tag |
| Translation regression too large | Use split heads (SE(3) for rotation, MLP for translation) |
| LIBERO too slow | Use LIBERO-Spatial only (10 tasks, fastest) |
| Null result | Frame as negative result — still publishable |
| Compute budget | 500M fits on single A100, ~3h per run |

## What NOT to Do

- Don't try to beat SmolVLA's end-to-end success rate (unfair comparison)
- Don't train from scratch (use pretrained SmolVLA)
- Don't use more than 500M params (breaks the budget constraint)
- Don't compare to OpenVLA/π0 (different scale entirely)
- Don't overclaim — honest modest improvements get published more easily
