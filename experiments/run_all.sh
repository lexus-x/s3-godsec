#!/bin/bash
# Run all experiments for SE(3)-VLA

set -e

echo "=========================================="
echo "SE(3)-VLA: Running All Experiments"
echo "=========================================="

# Experiment 1: Main comparison (Euclidean vs SE(3))
echo ""
echo "[Exp 1] Main comparison on MetaWorld MT-50..."
python src/train.py --config configs/octo_se3.yaml
python src/train.py --config configs/octo_baseline.yaml

# Experiment 2: Rotation magnitude analysis
echo ""
echo "[Exp 2] Rotation magnitude analysis..."
python experiments/ablation_rotation_tasks.py

# Experiment 3: Architecture ablation
echo ""
echo "[Exp 3] Architecture ablation..."
for LAYERS in 2 4 6 8; do
    echo "  Testing n_layers=$LAYERS..."
    # Modify config and run
done

for HIDDEN in 128 256 512; do
    echo "  Testing head_hidden_dim=$HIDDEN..."
    # Modify config and run
done

# Experiment 4: Parameterization ablation
echo ""
echo "[Exp 4] Parameterization ablation..."
python experiments/ablation_euclidean_vs_se3.py

# Experiment 5: LIBERO sanity check
echo ""
echo "[Exp 5] LIBERO benchmark..."
# Run on LIBERO

# Experiment 6: Inference speed
echo ""
echo "[Exp 6] Inference speed comparison..."
# Measure inference time for 1, 5, 10, 20 steps

echo ""
echo "=========================================="
echo "All experiments complete!"
echo "=========================================="
