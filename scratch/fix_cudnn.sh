#!/bin/bash
# Fix CuDNN version mismatch for Octo/JAX
# Problem: JAX compiled against CuDNN 9.8.0, system has 9.1.0
# Solution: Install matching CuDNN runtime via pip (no system-level changes)

set -e

echo "=== Current state ==="
python3 -c "import jax; print('JAX:', jax.__version__)" 2>&1
python3 -c "from jax.lib import xla_bridge; print('JAX backend:', xla_bridge.get_backend().platform)" 2>&1 || true

echo ""
echo "=== Installing CuDNN 9.8.0+ via pip ==="
pip install "nvidia-cudnn-cu12>=9.8.0"

echo ""
echo "=== Verifying fix ==="
python3 -c "
import jax
import jax.numpy as jnp
print('JAX version:', jax.__version__)
print('Devices:', jax.devices())
# Quick smoke test
x = jnp.ones((2, 2))
print('JAX matmul test:', jnp.dot(x, x).shape)
print('OK — JAX/CUDA working')
"

echo ""
echo "=== Testing Octo import ==="
python3 -c "
from octo.model.octo_model import OctoModel
print('Octo imported successfully')
# Uncomment to test full load (downloads ~350MB checkpoint):
# model = OctoModel.load_pretrained('octo-small')
# print('Octo loaded successfully')
"

echo ""
echo "=== Done ==="
echo "If Octo import works, proceed with Phase 1:"
echo "  python scratch/test_libero.py"
echo "  python src/train.py --config configs/octo_se3.yaml --seed 0"
