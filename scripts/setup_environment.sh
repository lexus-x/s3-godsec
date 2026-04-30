#!/bin/bash
# =============================================================================
# SE(3)-VLA: Environment Setup Script
# =============================================================================
#
# Sets up the complete training environment.
#
# Usage:
#     bash scripts/setup_environment.sh
#     bash scripts/setup_environment.sh --smolvla   # include SmolVLA
#     bash scripts/setup_environment.sh --benchmarks # include LIBERO/MetaWorld
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

INSTALL_SMOLVLA=false
INSTALL_BENCHMARKS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --smolvla) INSTALL_SMOLVLA=true; shift ;;
        --benchmarks) INSTALL_BENCHMARKS=true; shift ;;
        --full) INSTALL_SMOLVLA=true; INSTALL_BENCHMARKS=true; shift ;;
        *) shift ;;
    esac
done

echo "==========================================="
echo "  SE(3)-VLA Environment Setup"
echo "==========================================="

# Check Python version
echo ""
echo "[1/5] Checking Python..."
python_version=$(python --version 2>&1)
echo "  $python_version"

# Check/install pip
echo ""
echo "[2/5] Upgrading pip..."
pip install --upgrade pip -q

# Install core dependencies
echo ""
echo "[3/5] Installing core dependencies..."
pip install -q -r requirements.txt

# Install package in development mode
echo ""
echo "[4/5] Installing se3-vla in development mode..."
pip install -e . -q

# Verify PyTorch + CUDA
echo ""
echo "[5/5] Verifying installation..."
python -c "
import torch
print(f'  PyTorch:     {torch.__version__}')
print(f'  CUDA:        {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:         {torch.cuda.get_device_name(0)}')
    print(f'  VRAM:        {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
print(f'  Device:      {\"cuda\" if torch.cuda.is_available() else \"cpu\"}')
"

python -c "
import numpy as np
import yaml
import tqdm
import matplotlib
print(f'  NumPy:       {np.__version__}')
print(f'  PyYAML:      {yaml.__version__}')
print(f'  All core dependencies OK ✓')
"

# Optional: SmolVLA
if [ "$INSTALL_SMOLVLA" = true ]; then
    echo ""
    echo "[Optional] Installing SmolVLA (lerobot)..."
    if [ -d "lerobot" ]; then
        echo "  lerobot directory exists, updating..."
        cd lerobot && git pull && pip install -e ".[smolvla]" -q && cd ..
    else
        git clone https://github.com/huggingface/lerobot.git
        cd lerobot && pip install -e ".[smolvla]" -q && cd ..
    fi
    echo "  ✓ SmolVLA installed"
fi

# Optional: Benchmarks
if [ "$INSTALL_BENCHMARKS" = true ]; then
    echo ""
    echo "[Optional] Installing benchmark environments..."
    pip install -q libero 2>/dev/null || echo "  ⚠ LIBERO install failed (may need mujoco)"
    pip install -q "git+https://github.com/Farama-Foundation/Metaworld.git" 2>/dev/null || echo "  ⚠ MetaWorld install failed"
fi

echo ""
echo "==========================================="
echo "  Setup Complete!"
echo ""
echo "  Quick start:"
echo "    # Smoke test (synthetic data, 5 epochs)"
echo "    bash scripts/run_pipeline.sh --smoke"
echo ""
echo "    # Full training"
echo "    bash scripts/run_pipeline.sh"
echo ""
echo "    # Single head type"
echo "    bash scripts/run_pipeline.sh --head flow --seed 0"
echo "==========================================="
