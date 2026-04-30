# SE(3)-VLA Validation Gates

> **Each phase must pass its gate before proceeding to the next. No exceptions.**

---

## Phase 1 Gate: Synthetic Diagnostics

**Requirement:** SE(3) head must outperform Euclidean head on the same compact backbone.

### Criteria (ALL must pass)

| Criterion | Metric | Threshold | Status |
|-----------|--------|-----------|--------|
| Rotation improvement | Δ R-RMSE (Euclid − SE3) on `rotation_heavy` | > 0 (positive = SE3 wins) | ☐ |
| Minimum improvement | % improvement on `rotation_heavy` R-RMSE | ≥ 3% | ☐ |
| No regression on translation | T-RMSE on `translation_heavy` | SE3 ≤ 1.1 × Euclid | ☐ |
| Reproducibility | All 3 seeds show SE3 ≤ Euclid on `rotation_heavy` | 3/3 seeds | ☐ |
| Parameter budget | Total model parameters | < 400M | ☐ |

### How to verify

```bash
# Run Phase 1
bash scripts/run_pipeline.sh --phase 1

# Check results
python src/evaluate_smolvla.py --config configs/compact_se3.yaml \
    --checkpoint checkpoints/compact_se3/se3vla_flow_seed0_best.pt --compare
```

### Decision

- **PASS:** All criteria met → Proceed to Phase 2
- **PARTIAL:** SE(3) wins on rotation but regresses on translation → Try `chunk` head
- **FAIL:** SE(3) loses → Review architecture, do NOT proceed

---

## Phase 2 Gate: Sim Benchmarks (LIBERO / MetaWorld)

**Requirement:** Compact SE(3)-VLA (~252M) must outperform 1B+ models on rotation-heavy tasks.

### Criteria (ALL must pass)

| Criterion | Metric | Threshold | Status |
|-----------|--------|-----------|--------|
| Beat OpenVLA-7B | Success rate, rotation-heavy tasks | ≥ OpenVLA-7B | ☐ |
| Beat Octo-Base | Success rate, rotation-heavy tasks | ≥ Octo-Base (93M) | ☐ |
| Overall parity | Average success rate across all tasks | Within 5pp of best 1B+ model | ☐ |
| Rotation correlation | Correlation between rotation magnitude and Δ success rate | r > 0.3 | ☐ |
| Reproducibility | 3 seeds, consistent ranking | 3/3 seeds | ☐ |

### How to verify

```bash
# Run Phase 2 (requires LIBERO or MetaWorld installed)
bash scripts/run_pipeline.sh --phase 2

# Compare against baselines
python src/evaluate_smolvla.py --config configs/compact_se3.yaml \
    --checkpoint checkpoints/compact_se3/se3vla_flow_seed0_best.pt --compare
```

### Baseline Numbers to Beat

| Model | Params | Rotation-Heavy SR (est.) | Source |
|-------|--------|--------------------------|--------|
| OpenVLA-7B | 7B | ~45% | Kim et al. 2024 |
| Octo-Base | 93M | ~40% | Octo Team 2024 |
| Octo-Small | 27M | ~35% | Octo Team 2024 |
| SmolVLA-Euclidean | 450M | ~42% | HuggingFace 2025 |

### Decision

- **PASS:** Beats 1B+ on rotation-heavy, overall parity → Proceed to Phase 3
- **PARTIAL:** Beats some baselines → Write workshop paper, do NOT proceed to real-world
- **FAIL:** Does not beat baselines → Write negative result, do NOT proceed

---

## Phase 3 Gate: Real-World (only if Phase 2 passes)

**Requirement:** Sim results must transfer to physical robot.

### Criteria

| Criterion | Metric | Threshold | Status |
|-----------|--------|-----------|--------|
| Sim-to-real gap | Success rate drop from sim to real | < 15pp | ☐ |
| Rotation advantage preserved | SE(3) > Euclidean on rotation-heavy real tasks | Positive Δ | ☐ |
| Safety | No unsafe actions during evaluation | 0 incidents | ☐ |
| Reproducibility | 3 seeds on real robot | 3/3 consistent | ☐ |

### Decision

- **PASS:** All criteria met → Submit to CoRL/RA-L
- **PARTIAL:** Partial transfer → Submit to workshop
- **FAIL:** No transfer → Write sim-only paper

---

## Venue Decision Matrix

| Outcome | Phase 1 | Phase 2 | Phase 3 | Venue |
|---------|---------|---------|---------|-------|
| Strong support | ✓ | ✓ | ✓ | CoRL or RA-L (full paper) |
| Sim-only strong | ✓ | ✓ | ✗ | RA-L short or workshop |
| Rotation-only | ✓ | Partial | ✗ | Workshop paper |
| Negative result | ✗ or Partial | ✗ | ✗ | arXiv tech report (do NOT submit) |

---

## Running the Gates

```bash
# Phase 1
bash scripts/run_pipeline.sh --phase 1
# → Check: reports/PHASE1_REPORT_*.md

# Phase 2 (only if Phase 1 passes)
bash scripts/run_pipeline.sh --phase 2
# → Check: reports/PHASE2_REPORT_*.md

# Phase 3 (only if Phase 2 passes)
# Manual: requires real robot setup
```

---

## Emergency Stop

If any gate fails:
1. **STOP** — Do not proceed to the next phase
2. **DOCUMENT** — Write failure analysis in `reports/BLOCKER.md`
3. **REVIEW** — Adjust architecture or scope before retrying
4. **RE-RUN** — Only retry after fixing the root cause
