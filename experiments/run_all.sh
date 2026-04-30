#!/bin/bash
# SE(3)-VLA Controlled Experiment — 12-run sweep
#
# 2 heads × 2 backbones × 3 seeds = 12 runs
# Pre-registered in experiments/PREREGISTRATION.md

set -e

echo "======================================================================"
echo "  SE(3)-VLA: 12-Run Sweep (Pre-Registered)"
echo "======================================================================"
echo ""
echo "  Configs:"
echo "    SE(3) + scene_id:    configs/octo_se3.yaml"
echo "    Euclid + scene_id:   configs/octo_baseline.yaml"
echo "    SE(3) + mock_cnn:    configs/octo_se3_cnn.yaml"
echo "    Euclid + mock_cnn:   configs/octo_baseline_cnn.yaml"
echo ""
echo "  Seeds: 0, 1, 2"
echo "======================================================================"

CONFIGS=(
    "configs/octo_se3.yaml"
    "configs/octo_baseline.yaml"
    "configs/octo_se3_cnn.yaml"
    "configs/octo_baseline_cnn.yaml"
)

SEEDS=(0 1 2)

RUN=0
TOTAL=$((${#CONFIGS[@]} * ${#SEEDS[@]}))

for CONFIG in "${CONFIGS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        RUN=$((RUN + 1))
        echo ""
        echo "[Run ${RUN}/${TOTAL}] Config: ${CONFIG}, Seed: ${SEED}"
        echo "----------------------------------------------------------------------"
        python src/train.py --config "${CONFIG}" --seed "${SEED}"
    done
done

echo ""
echo "======================================================================"
echo "  All ${TOTAL} training runs complete!"
echo "  Now running evaluation..."
echo "======================================================================"

# Evaluate all checkpoints
mkdir -p results

for CONFIG in "${CONFIGS[@]}"; do
    # Extract model name and backbone from config
    MODEL=$(python -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c['model']['name'])")
    BACKBONE=$(python -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c['model'].get('backbone_kind','mock_cnn'))")

    for SEED in "${SEEDS[@]}"; do
        TAG="${MODEL}_${BACKBONE}_seed${SEED}"
        CKPT="checkpoints/${TAG}_best.pt"

        if [ -f "${CKPT}" ]; then
            echo ""
            echo "[Eval] ${TAG}"
            python src/evaluate.py \
                --config "${CONFIG}" \
                --checkpoint "${CKPT}" \
                --seed "${SEED}" \
                --output "results/${TAG}_eval.json"
        else
            echo "[WARN] Missing checkpoint: ${CKPT}"
        fi
    done
done

echo ""
echo "======================================================================"
echo "  Sweep complete! Results in results/"
echo "======================================================================"

# Summary table
echo ""
echo "Generating summary..."
python -c "
import json, glob, os

files = sorted(glob.glob('results/*_eval.json'))
if not files:
    print('No result files found.')
    exit()

print()
print(f\"{'Run':<45s} {'Family':<22s} {'G-RMSE':>8s} {'R-RMSE':>8s} {'T-RMSE':>8s}\")
print(f\"{'-'*45} {'-'*22} {'-'*8} {'-'*8} {'-'*8}\")

for f in files:
    tag = os.path.basename(f).replace('_eval.json', '')
    with open(f) as fh:
        data = json.load(fh)
    for family, m in data.items():
        print(f'{tag:<45s} {family:<22s} {m[\"geodesic_rmse\"]:8.4f} {m[\"rotation_rmse\"]:8.4f} {m[\"translation_rmse\"]:8.4f}')
"
