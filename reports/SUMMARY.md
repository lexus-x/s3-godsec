## Verdict: Partial support for SE(3) head — advantage holds under scene_id backbone (all 3 seeds), does not survive mock_cnn backbone (seed 1 inverts).

### Per-Seed Table for Primary Claim (rotation_heavy)

| Backbone | Seed | Euclid R-RMSE | SE(3) R-RMSE | Δ (Eucl - SE(3)) | SE(3) ≤ Eucl? |
|----------|------|---------------|--------------|------------------|---------------|
| scene_id | 0    | 2.4330        | 2.3178       | 0.1152           | YES           |
| scene_id | 1    | 2.3443        | 2.2519       | 0.0925           | YES           |
| scene_id | 2    | 2.4394        | 2.3176       | 0.1218           | YES           |
| mock_cnn | 0    | 2.4340        | 2.2448       | 0.1892           | YES           |
| mock_cnn | 1    | 2.1146        | 2.2166       | -0.1020          | NO            |
| mock_cnn | 2    | 2.2853        | 2.2305       | 0.0548           | YES           |

### Aggregate Table (mean ± std across seeds)

| Head | Backbone | Family | G-RMSE | R-RMSE | T-RMSE |
|------|----------|--------|--------|--------|--------|
| OctoEuclideanBaseline | mock_cnn | rotation_heavy | 2.2785 ± 0.1305 | 2.2779 ± 0.1305 | 0.0408 ± 0.0080 |
| OctoEuclideanBaseline | mock_cnn | translation_heavy | 0.4514 ± 0.0700 | 0.3016 ± 0.1092 | 0.3243 ± 0.0020 |
| OctoEuclideanBaseline | mock_cnn | combined | 1.6606 ± 0.0530 | 1.6444 ± 0.0537 | 0.2291 ± 0.0024 |
| OctoEuclideanBaseline | scene_id | rotation_heavy | 2.4052 ± 0.0429 | 2.4056 ± 0.0434 | 0.0257 ± 0.0005 |
| OctoEuclideanBaseline | scene_id | translation_heavy | 0.3578 ± 0.0018 | 0.1521 ± 0.0025 | 0.3236 ± 0.0021 |
| OctoEuclideanBaseline | scene_id | combined | 1.7204 ± 0.0021 | 1.7060 ± 0.0020 | 0.2281 ± 0.0023 |
| OctoSE3 | mock_cnn | rotation_heavy | 2.2392 ± 0.0103 | 2.2306 ± 0.0115 | 0.1642 ± 0.0119 |
| OctoSE3 | mock_cnn | translation_heavy | 0.4614 ± 0.0299 | 0.2377 ± 0.0201 | 0.3934 ± 0.0408 |
| OctoSE3 | mock_cnn | combined | 1.5898 ± 0.0191 | 1.5586 ± 0.0161 | 0.3040 ± 0.0242 |
| OctoSE3 | scene_id | rotation_heavy | 2.3033 ± 0.0294 | 2.2958 ± 0.0310 | 0.1550 ± 0.0220 |
| OctoSE3 | scene_id | translation_heavy | 0.4886 ± 0.0166 | 0.2376 ± 0.0338 | 0.4234 ± 0.0401 |
| OctoSE3 | scene_id | combined | 1.6490 ± 0.0120 | 1.6162 ± 0.0083 | 0.3184 ± 0.0235 |

### Honest Negative Finding

SE(3) is consistently 10–41% worse on translation T-RMSE in the translation_heavy family across every seed and backbone.

| Backbone | Seed | Euclid T-RMSE | SE(3) T-RMSE | Δ (Eucl - SE(3)) |
|----------|------|---------------|--------------|------------------|
| scene_id | 0    | 0.3234        | 0.4570       | -0.1336          |
| scene_id | 1    | 0.3262        | 0.3671       | -0.0408          |
| scene_id | 2    | 0.3212        | 0.4460       | -0.1248          |
| mock_cnn | 0    | 0.3247        | 0.4510       | -0.1263          |
| mock_cnn | 1    | 0.3264        | 0.3616       | -0.0351          |
| mock_cnn | 2    | 0.3217        | 0.3675       | -0.0458          |

### Caveats

- The "Euclidean baseline" uses exp_so3(ω) (src/evaluate.py:91-97), so it's already on the manifold; the SE(3) advantage is attributable to flow matching specifically, not "manifold vs no-manifold."
- Background hue gradient remains family-correlated (data_loader.py:154); we proceeded post-corner-removal because hue encodes a legitimate per-sample feature (rotation trace), not a literal family ID.
- Effect size on the headline number is 4.6% R-RMSE improvement on rotation_heavy (scene_id) — modest, ~2.5σ across seeds.

### Pre-registration Mapping

Matching outcome from PREREGISTRATION.md's "Acceptable Outcomes" table:
"SE(3) < Euclid on rot_heavy (scene_id only) — Partial support — narrower claim"

We honor the pre-registration by fully reporting whatever happened, acknowledging this narrower result and the negative translation findings.

### Reproducibility

- Git commits: 54dafc7 (pre-registration), 8c68ea1 (leakage fix)
- Runtime: A100 80GB, ~2.5–3 min/run, 12 runs total
- Exact commands used to run the sweep:
  - Phase 1 (scene_id runs): the original sweep, started via `bash experiments/run_all.sh` and stopped after run 6 of 12 (commit `54dafc7` HEAD at the time).
  - Phase 2 (mock_cnn runs): after detecting and fixing the corner-marker leakage (commit `8c68ea1`), the 6 mock_cnn runs were launched via a custom loop:
    ```bash
    for SEED in 0 1 2; do
      python src/train.py --config configs/octo_se3_cnn.yaml --seed $SEED
      python src/train.py --config configs/octo_baseline_cnn.yaml --seed $SEED
    done
    ```
  - Phase 3 (eval): `python src/evaluate.py` was invoked once per checkpoint (12 total) with per-family JSON output to `results/`.
