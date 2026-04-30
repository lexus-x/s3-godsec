#!/bin/bash
set -e

echo "=== Finishing 12-run sweep (Run 12 + Evaluations) ==="

# Run 12: Baseline + mock_cnn, seed 2
echo "[Run 12/12] Baseline + mock_cnn, Seed 2"
python src/train.py --config configs/octo_baseline_cnn.yaml --seed 2

echo ""
echo "=== All training runs complete ==="
echo "=== Now running evaluation for ALL 12 configs ==="

CONFIGS=(
    "configs/octo_se3.yaml"
    "configs/octo_baseline.yaml"
    "configs/octo_se3_cnn.yaml"
    "configs/octo_baseline_cnn.yaml"
)
SEEDS=(0 1 2)

mkdir -p results

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
