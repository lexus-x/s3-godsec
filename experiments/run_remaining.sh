#!/bin/bash
# Continue sweep from run 5/12 (Baseline scene_id seed1 onward)
set -e

echo "=== Continuing 12-run sweep from run 5/12 ==="

# Run 5: Baseline + scene_id, seed 1
echo "[Run 5/12] Baseline + scene_id, Seed 1"
python src/train.py --config configs/octo_baseline.yaml --seed 1

# Run 6: Baseline + scene_id, seed 2
echo "[Run 6/12] Baseline + scene_id, Seed 2"
python src/train.py --config configs/octo_baseline.yaml --seed 2

# Run 7: SE3 + mock_cnn, seed 0
echo "[Run 7/12] SE3 + mock_cnn, Seed 0"
python src/train.py --config configs/octo_se3_cnn.yaml --seed 0

# Run 8: SE3 + mock_cnn, seed 1
echo "[Run 8/12] SE3 + mock_cnn, Seed 1"
python src/train.py --config configs/octo_se3_cnn.yaml --seed 1

# Run 9: SE3 + mock_cnn, seed 2
echo "[Run 9/12] SE3 + mock_cnn, Seed 2"
python src/train.py --config configs/octo_se3_cnn.yaml --seed 2

# Run 10: Baseline + mock_cnn, seed 0
echo "[Run 10/12] Baseline + mock_cnn, Seed 0"
python src/train.py --config configs/octo_baseline_cnn.yaml --seed 0

# Run 11: Baseline + mock_cnn, seed 1
echo "[Run 11/12] Baseline + mock_cnn, Seed 1"
python src/train.py --config configs/octo_baseline_cnn.yaml --seed 1

# Run 12: Baseline + mock_cnn, seed 2
echo "[Run 12/12] Baseline + mock_cnn, Seed 2"
python src/train.py --config configs/octo_baseline_cnn.yaml --seed 2

echo ""
echo "=== All remaining training runs complete ==="
echo "=== Now running evaluation for ALL 12 configs ==="

# Evaluate ALL 12 checkpoints (including the 4 we already re-trained)
CONFIGS=(
    "configs/octo_se3.yaml"
    "configs/octo_baseline.yaml"
    "configs/octo_se3_cnn.yaml"
    "configs/octo_baseline_cnn.yaml"
)
SEEDS=(0 1 2)

for CONFIG in "${CONFIGS[@]}"; do
    MODEL=$(python -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c['model']['name'])")
    BACKBONE=$(python -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c['model'].get('backbone_kind','mock_cnn'))")
    for SEED in "${SEEDS[@]}"; do
        TAG="${MODEL}_${BACKBONE}_seed${SEED}"
        CKPT="checkpoints/${TAG}_best.pt"
        if [ -f "${CKPT}" ]; then
            echo "[Eval] ${TAG}"
            python src/evaluate.py \
                --config "${CONFIG}" \
                --checkpoint "${CKPT}" \
                --seed "${SEED}" \
                --output "results/${TAG}_eval.json"
        else
            echo "[WARN] Missing: ${CKPT}"
        fi
    done
done

echo ""
echo "=== Sweep complete! Results in results/ ==="
