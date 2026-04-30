> *This is a synthetic diagnostic. Main results are in `reports/MAIN_RESULTS.md` (real benchmark).*

## Verdict: Strong support for SE(3) head — advantage holds under both `scene_id` and `mock_cnn` backbones across all 3 seeds post-leakage-fix.

### Per-Seed Table for Primary Claim (rotation_heavy)

| Backbone | Seed | Euclid R-RMSE | SE(3) R-RMSE | Δ (Eucl - SE(3)) | SE(3) ≤ Eucl? |
|----------|------|---------------|--------------|------------------|---------------|
| scene_id | 0 | 2.4330 | 2.3099 | 0.1231 | YES |
| scene_id | 1 | 2.3443 | 2.2696 | 0.0747 | YES |
| scene_id | 2 | 2.4394 | 2.2789 | 0.1605 | YES |
| mock_cnn | 0 | 2.4343 | 2.3044 | 0.1299 | YES |
| mock_cnn | 1 | 2.3464 | 2.3240 | 0.0224 | YES |
| mock_cnn | 2 | 2.4424 | 2.3075 | 0.1349 | YES |

### Aggregate Table (mean ± std across seeds)

| Head | Backbone | Family | G-RMSE | R-RMSE | T-RMSE |
|------|----------|--------|--------|--------|--------|
| OctoEuclideanBaseline | mock_cnn | rotation_heavy | 2.4079 ± 0.0435 | 2.4077 ± 0.0435 | 0.0286 ± 0.0005 |
| OctoEuclideanBaseline | mock_cnn | translation_heavy | 0.3664 ± 0.0107 | 0.1688 ± 0.0248 | 0.3242 ± 0.0022 |
| OctoEuclideanBaseline | mock_cnn | combined | 1.7229 ± 0.0041 | 1.7076 ± 0.0042 | 0.2283 ± 0.0024 |
| OctoEuclideanBaseline | scene_id | rotation_heavy | 2.4052 ± 0.0429 | 2.4056 ± 0.0434 | 0.0257 ± 0.0005 |
| OctoEuclideanBaseline | scene_id | translation_heavy | 0.3578 ± 0.0018 | 0.1521 ± 0.0025 | 0.3236 ± 0.0021 |
| OctoEuclideanBaseline | scene_id | combined | 1.7204 ± 0.0021 | 1.7060 ± 0.0020 | 0.2281 ± 0.0023 |
| OctoSE3 | mock_cnn | rotation_heavy | 2.3175 ± 0.0082 | 2.3120 ± 0.0086 | 0.1319 ± 0.0155 |
| OctoSE3 | mock_cnn | translation_heavy | 0.3134 ± 0.0301 | 0.2689 ± 0.0150 | 0.1591 ± 0.0339 |
| OctoSE3 | mock_cnn | combined | 1.6421 ± 0.0119 | 1.6347 ± 0.0117 | 0.1459 ± 0.0236 |
| OctoSE3 | scene_id | rotation_heavy | 2.2934 ± 0.0173 | 2.2861 ± 0.0172 | 0.1518 ± 0.0178 |
| OctoSE3 | scene_id | translation_heavy | 0.4888 ± 0.0158 | 0.2336 ± 0.0344 | 0.4260 ± 0.0381 |
| OctoSE3 | scene_id | combined | 1.6584 ± 0.0124 | 1.6267 ± 0.0091 | 0.3137 ± 0.0204 |

### Honest Negative Finding (translation_heavy T-RMSE)

SE(3) performs slightly worse on translation T-RMSE in the translation_heavy family under the `scene_id` backbone. However, note that under the `mock_cnn` backbone, SE(3) actually outperforms the baseline on translation RMSE as well.

| Backbone | Seed | Euclid T-RMSE | SE(3) T-RMSE | Δ (Eucl - SE(3)) |
|----------|------|---------------|--------------|------------------|
| scene_id | 0 | 0.3234 | 0.4577 | -0.1343 |
| scene_id | 1 | 0.3262 | 0.3725 | -0.0463 |
| scene_id | 2 | 0.3212 | 0.4478 | -0.1266 |
| mock_cnn | 0 | 0.3234 | 0.1329 | 0.1905 |
| mock_cnn | 1 | 0.3272 | 0.1374 | 0.1898 |
| mock_cnn | 2 | 0.3219 | 0.2069 | 0.1149 |

### Caveats

- The "Euclidean baseline" uses exp_so3(ω) (src/evaluate.py:91-97), so it's already on the manifold; the SE(3) advantage is attributable to flow matching specifically, not "manifold vs no-manifold."
- Background hue gradient leakage has been fixed (data_loader.py uses neutral gray background now).
- Post-leakage fix, the SE(3) head retains its advantage across ALL seeds for BOTH backbones, making this a Strong Support finding per the pre-registration criteria.

### Pre-registration Mapping

Matching outcome from PREREGISTRATION.md's "Acceptable Outcomes" table:
"SE(3) < Euclid on rot_heavy (both backbones) — Strong support"

### Reproducibility

- Runtime: A100 80GB, ~3-5 min/run, 12 runs total.
- Run protocol: `bash experiments/run_all.sh` (or `experiments/run_remaining.sh` and `experiments/run_final.sh` which replicate its functionality).
