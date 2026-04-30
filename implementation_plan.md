# SE(3)-VLA Real Experiment Implementation Plan

## Goal
Convert the SE(3)-VLA pilot into a real, publishable result by moving from synthetic diagnostics to a real benchmark and backbone.

## Non-goals
- Do not add features to the synthetic dataset; treat it as a frozen diagnostic.
- Do not rewrite `src/models/se3_action_head.py` or `geodesic_loss.py` unless explicitly required.
- Do not invent a "Theorem 1" or "Geodesic Action-ECE" section without validation.
- Do not weaken the pre-registration.
- Do not silently expand scope, swap benchmarks, or change the head architecture mid-run.

## Working Agreement
- Use task.md to track phases and tick off as each completes.
- Commit after every phase with a tag. Tags: `pilot-frozen`, `phase0-postleak`, `phase1-baseline`, `phase2-se3-integrated`, `prereg-real`, `phase4-results-frozen`. Never amend.
- Every phase ends with a written artifact under `s3-godsec/reports/` and a verification gate that a fresh reader can re-run from the command line.
- **Blocker rule:** If a phase's gate fails, stop immediately. Do not start the next phase. Write `reports/BLOCKER.md` with the evidence and proposed scope change; wait for user review before proceeding.

---

## Phase 0 — Freeze the pilot and fix what is shippable in-place
**Goal:** The synthetic study becomes a clearly-labeled diagnostic appendix.

1. **Tag Repository:** Tag current `master` as `pilot-frozen`.
2. **Add Banners:** Add a banner to `s3-godsec/README.md` and `s3-godsec/reports/SUMMARY.md`: *"This is a synthetic diagnostic. Main results are in `reports/MAIN_RESULTS.md` (real benchmark)."*
3. **Fix Leakage:** Fix the hue-trace leakage in `src/training/data_loader.py:150-156` by replacing the rotation-trace-encoded hue with a fixed neutral background.
4. **Re-run Sweep:** Re-run the 12-run sweep and write `reports/PILOT_POSTLEAK.md` with the new tables. If the SE(3) advantage disappears entirely, flag this to adjust paper scope.
5. **Clean Aspirational Data:** Delete the aspirational success-rate tables from `README.md` and the abstract from `docs/PAPER_OUTLINE.md`. Replace with `TBD — pending Phase 4 results`.
6. **Tag:** Commit and tag `phase0-postleak`.

**Gate:** `bash experiments/run_all.sh` reproduces the post-leak numbers exactly; `reports/PILOT_POSTLEAK.md` exists and is consistent with the JSON in `results/`.

---

## Phase 1 — Stand up a real backbone and a real benchmark
**Goal:** Working train+eval loop on one real VLA backbone and one real benchmark (baseline only).

1. **Pick Backbone/Benchmark:** Try to install in this order:
   - (A) Octo-Small + LIBERO-Spatial
   - (B) OpenVLA-7B + LIBERO-Spatial
   - (C) Octo-Small + MetaWorld MT-10
2. **Implement Backbone:** Add `src/backbones/real_backbone.py` (load checkpoint, freeze it, expose `forward(obs) -> hidden`).
3. **Implement Dataset:** Add `src/data/real_dataset.py` yielding `(obs, language, target_action, family_label)` from demonstrations.
4. **Train Baseline:** Train the Euclidean baseline only for 1 seed on the smallest split. "Smallest split" means: for LIBERO-Spatial, 10 demonstrations × 10 tasks; for MT-10, 10 tasks × 1 seed. Output checkpoint and success-rate to `reports/PHASE1_BASELINE.md`.
5. **Tag:** Commit and tag `phase1-baseline`.

**Gate:** Baseline success rate is within ±5pp of the published number. If it is not, invoke the blocker rule — do not proceed.

---

## Phase 2 — Drop the SE(3) head into the real stack
**Goal:** SE(3) head trains and evals on the real benchmark.

1. **Plumb SE(3) Head:** Integrate `OctoSE3` through `real_backbone.py`.
2. **Resolve T-RMSE Regression:** Decide between:
   - Split heads (SE(3) flow-matching for rotation, MLP for translation). Default choice.
   - Joint SE(3) with re-tuned translation scale.
   Document in `reports/PHASE2_DESIGN.md`.
3. **Train SE(3) Head:** 1 seed, smallest split. Confirm success rate ≥ baseline minus 5pp. If it fails this sanity check, do not proceed to Phase 3 — debug or invoke the blocker rule.
4. **Tag:** Commit and tag `phase2-se3-integrated`.

**Gate:** Both heads train without NaNs, eval without errors, and SE(3) doesn't catastrophically regress on translation.

---

## Phase 3 — Pre-register the real experiment, then run it
**Goal:** A 3-seed comparison on the real benchmark with a frozen pre-registration.

1. **Pre-registration:** Write `experiments/PREREGISTRATION_REAL.md` (benchmark, backbone, splits, metric, seeds, decision rule). Commit and tag `prereg-real`.
2. **No hyperparameter changes** between Phase 2's 1-seed run and the 3-seed pre-registered run. The pre-registration is meaningless otherwise.
3. **Run Experiment:** Run 3 seeds × 2 heads on the full eval suite. Record to `results/real/`.
4. **Blind Execution:** Do not look at SE(3) results until all 6 runs are complete.

**Gate:** 6 JSON files in `results/real/`; commit hash matches `prereg-real`'s parent.

---

## Phase 4 — Write the result, not the story
**Goal:** An honest, rigorous report for CoRL/RA-L.

1. **Main Results:** Write `reports/MAIN_RESULTS.md` (3-seed mean ± std, aggregate by rotation magnitude bucket, map to decision rule).
2. **Ablations:** Write `reports/ABLATIONS.md` (head-only ablation, flow-step count).
3. **Update Outline:** Rewrite `docs/PAPER_OUTLINE.md` to match actual measurements. No number appears in the abstract that is not in a table.
4. **Venue Decision:** Decide venue based on outcome:

| Outcome | Venue | Format |
|---------|-------|--------|
| Strong support across both rotation and translation buckets | CoRL or RA-L | Full paper |
| Partial support (rotation-only improvement) | RA-L short or workshop | Short paper |
| Null or negative result | arXiv tech report | Write-up of negative result; **do not submit** |

5. **Tag:** Commit and tag `phase4-results-frozen`.

**Gate:** Every claim in `PAPER_OUTLINE.md` traces to a row in `MAIN_RESULTS.md` or `ABLATIONS.md`. No number appears in the abstract that is not in a table.

---

## Reporting Cadence
At the end of each phase, append one paragraph to `reports/PROGRESS.md`: what ran, what passed the gate, what the artifact is, and what is next.
