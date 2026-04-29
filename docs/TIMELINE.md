# TIMELINE.md — Week-by-Week Execution Plan

## Overview

- **Total duration**: 6 weeks
- **Compute budget**: ~9 A100-days
- **Target submission**: RA-L + ICRA 2027 (~Sep 2026 deadline)
- **ArXiv preprint**: Week 4 (to establish priority)

## Week 1: Foundation (May 1-7)

### Goals
- [ ] Set up development environment
- [ ] Download pretrained Octo checkpoint
- [ ] Implement SE(3) utility functions (exp, log, geodesic distance)
- [ ] Write unit tests for SE(3) operations
- [ ] Implement SE(3) flow matching head
- [ ] Verify Theorem 1 numerically on synthetic data

### Deliverables
- `src/utils/se3_utils.py` — complete with tests
- `src/models/se3_action_head.py` — complete
- `tests/test_se3.py` — passing
- Numerical verification of Theorem 1 (notebook)

### Compute: 0 (code only)

## Week 2: Integration + Training (May 8-14)

### Goals
- [ ] Implement Octo adapter (plug SE(3) head into Octo)
- [ ] Implement training loop with geodesic flow matching loss
- [ ] Implement data loader for MetaWorld MT-50
- [ ] Run initial training on rotation-heavy subset (5 tasks)
- [ ] Debug any numerical issues
- [ ] Hyperparameter sweep (learning rate, flow steps, hidden dim)

### Deliverables
- `src/models/octo_adapter.py` — complete
- `src/training/trainer.py` — complete
- `src/training/data_loader.py` — complete
- Initial training curves and sanity checks

### Compute: 1-2 A100-days

## Week 3: Main Experiments (May 15-21)

### Goals
- [ ] **Experiment 1**: Main comparison (Euclidean vs. SE(3)) on full MT-50
- [ ] **Experiment 2**: Rotation magnitude analysis (compute from Exp 1 results)
- [ ] **Experiment 5**: LIBERO sanity check
- [ ] **Experiment 6**: Inference speed comparison
- [ ] Generate all result tables and figures

### Deliverables
- Results tables for Experiments 1, 2, 5, 6
- **KEY FIGURE**: Rotation magnitude vs. Δ(success rate) scatter plot
- Raw result data (JSON/pickle)

### Compute: 3-4 A100-days

## Week 4: Ablations + ArXiv (May 22-28)

### Goals
- [ ] **Experiment 3**: Architecture ablation
- [ ] **Experiment 4**: Parameterization ablation
- [ ] **Experiment 7**: Geodesic Action-ECE
- [ ] Write paper Sections 1-4 (Intro, Related Work, Background, Method)
- [ ] Generate all figures
- [ ] **Submit arXiv preprint** (to establish priority)

### Deliverables
- Results tables for Experiments 3, 4, 7
- Paper draft (Sections 1-4)
- ArXiv submission

### Compute: 3 A100-days

## Week 5: Paper Completion (May 29 - Jun 4)

### Goals
- [ ] Write paper Section 5 (Experiments)
- [ ] Write paper Sections 6-7 (Discussion, Conclusion)
- [ ] Write appendix (Theorem 1 proof, additional ablations)
- [ ] Internal review and revision
- [ ] Prepare supplementary material (code, videos)

### Deliverables
- Complete paper draft
- Supplementary material
- Code repository cleaned up

### Compute: 0 (writing only)

## Week 6: Submission (Jun 5-11)

### Goals
- [ ] Final revision based on internal feedback
- [ ] Format paper for target venue (RA-L template)
- [ ] Submit to RA-L + ICRA 2027
- [ ] Push code to GitHub
- [ ] Announce on social media (optional)

### Deliverables
- Submitted paper
- Public GitHub repository
- Twitter/thread announcement (optional)

### Compute: 0

## Gantt Chart

```
Week 1  ██████████  Foundation (code + tests)
Week 2  ██████████  Integration + Training
Week 3  ██████████  Main Experiments
Week 4  ██████████  Ablations + ArXiv
Week 5  ██████████  Paper Writing
Week 6  ██████████  Submission
```

## Milestones

| Date | Milestone | Status |
|------|-----------|--------|
| May 7 | SE(3) head implemented + tested | ⬜ |
| May 14 | First training run complete | ⬜ |
| May 21 | Main experiment results in | ⬜ |
| May 28 | ArXiv preprint submitted | ⬜ |
| Jun 4 | Paper draft complete | ⬜ |
| Jun 11 | Paper submitted to RA-L | ⬜ |

## Dependencies

| Dependency | Status | Notes |
|-----------|--------|-------|
| Octo pretrained checkpoint | ✅ Public | HuggingFace |
| MetaWorld MT-50 | ✅ Public | pip install metaworld |
| LIBERO | ✅ Public | pip install libero |
| geoopt | ✅ Public | pip install geoopt |
| theseus | ✅ Public | pip install theseus |
| A100 GPU access | ⬜ Need | 9 A100-days total |
