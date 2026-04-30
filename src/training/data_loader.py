"""
Synthetic SE(3) demonstration dataset for controlled experiments.

Generates synthetic observations paired with SE(3) target actions with
clean bimodal rotation distributions for A/B testing SE(3) vs Euclidean heads.

Family definitions (hard-bounded, no leakage):
  rotation_heavy:
    ω direction ~ Uniform(S²), θ ~ Uniform(π/2, π), ‖v‖ capped at 0.05
  translation_heavy:
    ω direction ~ Uniform(S²), θ ~ Uniform(0, π/12), ‖v‖ ~ Uniform(0.1, 0.5)
"""

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
import math
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.se3_utils import exp_se3


def _sample_uniform_sphere(rng, n):
    """Sample n unit vectors uniformly on S²."""
    v = rng.randn(n, 3).astype(np.float32)
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return v / norms


class SyntheticSE3Dataset(Dataset):
    """
    Synthetic dataset with clean bimodal rotation/translation families.

    Each sample contains:
    - observation image (encodes action visually)
    - task_id (integer identifying the family, for scene-ID backbone)
    - target SE(3) action
    - family label ('rotation_heavy' or 'translation_heavy')

    Family sampling (hard bounds, no leakage):
    - rotation_heavy:  θ ~ Uniform(π/2, π),   ‖v‖ ≤ 0.05
    - translation_heavy: θ ~ Uniform(0, π/12), ‖v‖ ~ Uniform(0.1, 0.5)
    """

    # Family constants — integer IDs for embedding lookup
    FAMILY_IDS = {'rotation_heavy': 0, 'translation_heavy': 1}

    def __init__(
        self,
        n_samples: int = 5000,
        task_type: str = 'rotation_heavy',
        image_size: int = 224,
        seed: int = 42,
    ):
        """
        Args:
            n_samples: Number of samples to generate
            task_type: 'rotation_heavy' or 'translation_heavy'
            image_size: Size of generated images
            seed: Random seed for reproducibility
        """
        super().__init__()
        assert task_type in ('rotation_heavy', 'translation_heavy'), \
            f"task_type must be 'rotation_heavy' or 'translation_heavy', got '{task_type}'"

        self.n_samples = n_samples
        self.task_type = task_type
        self.family_id = self.FAMILY_IDS[task_type]
        self.image_size = image_size

        rng = np.random.RandomState(seed)

        # Generate SE(3) target actions with hard-bounded distributions
        self.target_actions = self._generate_targets(rng)

        # Generate corresponding images
        self.images = self._generate_images(rng)

        # Task IDs for scene-ID backbone (constant per dataset instance)
        self.task_ids = torch.full((n_samples,), self.family_id, dtype=torch.long)

        # Language tokens (placeholder)
        self.language_tokens = torch.randint(0, 1000, (n_samples, 10))

        # Gripper targets (binary)
        self.gripper_targets = torch.bernoulli(
            torch.full((n_samples, 1), 0.5)
        )

        # Family label string for each sample
        self.family_labels = [task_type] * n_samples

    def _generate_targets(self, rng):
        """Generate SE(3) targets with hard-bounded rotation/translation."""
        xi_all = np.zeros((self.n_samples, 6), dtype=np.float32)

        # Sample rotation axes uniformly on S²
        axes = _sample_uniform_sphere(rng, self.n_samples)

        if self.task_type == 'rotation_heavy':
            # θ ~ Uniform(π/2, π) — guaranteed large rotations
            thetas = rng.uniform(math.pi / 2, math.pi, size=self.n_samples).astype(np.float32)

            # ω = axis * θ
            xi_all[:, :3] = axes * thetas[:, None]

            # ‖v‖ capped at 0.05 — sample direction, then scale uniformly
            v_dirs = _sample_uniform_sphere(rng, self.n_samples)
            v_mags = rng.uniform(0.0, 0.05, size=self.n_samples).astype(np.float32)
            xi_all[:, 3:] = v_dirs * v_mags[:, None]

        elif self.task_type == 'translation_heavy':
            # θ ~ Uniform(0, π/12) — guaranteed small rotations (≤15°)
            thetas = rng.uniform(0, math.pi / 12, size=self.n_samples).astype(np.float32)

            # ω = axis * θ
            xi_all[:, :3] = axes * thetas[:, None]

            # ‖v‖ ~ Uniform(0.1, 0.5) — guaranteed substantial translations
            v_dirs = _sample_uniform_sphere(rng, self.n_samples)
            v_mags = rng.uniform(0.1, 0.5, size=self.n_samples).astype(np.float32)
            xi_all[:, 3:] = v_dirs * v_mags[:, None]

        # Convert to SE(3) matrices via exp map
        xi_tensor = torch.from_numpy(xi_all)
        T = exp_se3(xi_tensor)  # [N, 4, 4]

        return T

    def _generate_images(self, rng):
        """
        Generate synthetic images correlated with target action.

        Encodes rotation (background hue + orientation bar) and
        translation (circle position) into the image.

        Fully vectorized — no Python per-sample loop for background,
        bars, or noise. Circle masks still loop per-sample (cheap).
        """
        S = self.image_size
        N = self.n_samples
        images = np.full((N, 3, S, S), 0.5, dtype=np.float32)

        # Extract all rotations and translations at once
        actions_np = self.target_actions.numpy()  # [N, 4, 4]
        R_all = actions_np[:, :3, :3]   # [N, 3, 3]
        t_all = actions_np[:, :3, 3]    # [N, 3]

        # --- Translation → circle position ---
        cx = np.clip((t_all[:, 0] + 0.5) * S, 5, S - 5).astype(np.int32)
        cy = np.clip((t_all[:, 1] + 0.5) * S, 5, S - 5).astype(np.int32)
        radii = np.maximum(3, (10 + t_all[:, 2] * 20).astype(np.int32))

        yy, xx = np.mgrid[:S, :S]  # [S, S] each

        for i in range(N):
            dist_sq = (xx - cx[i]) ** 2 + (yy - cy[i]) ** 2
            mask = dist_sq < radii[i] ** 2
            images[i, 0][mask] = 0.8
            images[i, 1][mask] = 0.3
            images[i, 2][mask] = 0.1

        # --- Rotation-encoding bar (vectorized) ---
        angles = np.arctan2(R_all[:, 1, 0], R_all[:, 0, 0])  # [N]
        bar_len = 20
        steps = np.arange(bar_len, dtype=np.float32)  # [bar_len]
        center = S / 2.0

        # All bar pixel coords: [N, bar_len]
        px_all = (center + steps[None, :] * np.cos(angles[:, None])).astype(np.int32)
        py_all = (center + steps[None, :] * np.sin(angles[:, None])).astype(np.int32)

        # Flatten and filter valid coords
        sample_idx = np.repeat(np.arange(N), bar_len)
        px_flat = px_all.ravel()
        py_flat = py_all.ravel()
        valid = (px_flat >= 0) & (px_flat < S) & (py_flat >= 0) & (py_flat < S)
        images[sample_idx[valid], 0, py_flat[valid], px_flat[valid]] = 0.9
        images[sample_idx[valid], 1, py_flat[valid], px_flat[valid]] = 0.9
        images[sample_idx[valid], 2, py_flat[valid], px_flat[valid]] = 0.9

        # --- Add noise and clip (vectorized) ---
        images += rng.randn(*images.shape).astype(np.float32) * 0.02
        np.clip(images, 0, 1, out=images)

        return torch.from_numpy(images)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return {
            'observations': {
                'image': self.images[idx],
                'task_id': self.task_ids[idx],
            },
            'language': self.language_tokens[idx],
            'target_actions': self.target_actions[idx],
            'target_gripper': self.gripper_targets[idx],
            'family': self.family_labels[idx],
        }


class CombinedSE3Dataset(Dataset):
    """
    Combines rotation_heavy and translation_heavy datasets.

    Used for training (model sees both families) and for per-family eval
    (each sample carries its family label).
    """

    def __init__(
        self,
        n_per_family: int = 2500,
        image_size: int = 224,
        seed: int = 42,
    ):
        self.rot_dataset = SyntheticSE3Dataset(
            n_samples=n_per_family,
            task_type='rotation_heavy',
            image_size=image_size,
            seed=seed,
        )
        self.trans_dataset = SyntheticSE3Dataset(
            n_samples=n_per_family,
            task_type='translation_heavy',
            image_size=image_size,
            seed=seed + 500,
        )
        self.total = n_per_family * 2

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        n = len(self.rot_dataset)
        if idx < n:
            return self.rot_dataset[idx]
        else:
            return self.trans_dataset[idx - n]


def create_dataloaders(config, seed=42):
    """
    Create train and validation dataloaders.

    Training uses a combined dataset (both families, shuffled).
    Validation returns separate loaders per family for clean per-family metrics.

    Args:
        config: dict with 'data' and 'training' keys
        seed: random seed

    Returns:
        train_loader, val_loaders dict
            val_loaders = {
                'rotation_heavy': DataLoader,
                'translation_heavy': DataLoader,
                'combined': DataLoader,
            }
    """
    data_cfg = config.get('data', {})
    batch_size = config.get('training', {}).get('batch_size', 32)
    image_size = data_cfg.get('image_size', [224, 224])
    if isinstance(image_size, list):
        image_size = image_size[0]

    n_train_per_family = data_cfg.get('n_train_per_family', 2500)
    n_val_per_family = data_cfg.get('n_val_per_family', 250)

    # Training: combined dataset (both families, shuffled)
    train_dataset = CombinedSE3Dataset(
        n_per_family=n_train_per_family,
        image_size=image_size,
        seed=seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    # Validation: separate loaders per family
    val_rot = SyntheticSE3Dataset(
        n_samples=n_val_per_family,
        task_type='rotation_heavy',
        image_size=image_size,
        seed=seed + 1000,
    )
    val_trans = SyntheticSE3Dataset(
        n_samples=n_val_per_family,
        task_type='translation_heavy',
        image_size=image_size,
        seed=seed + 2000,
    )
    val_combined = CombinedSE3Dataset(
        n_per_family=n_val_per_family,
        image_size=image_size,
        seed=seed + 3000,
    )

    val_loaders = {
        'rotation_heavy': DataLoader(
            val_rot, batch_size=batch_size, shuffle=False,
            num_workers=2, pin_memory=True,
        ),
        'translation_heavy': DataLoader(
            val_trans, batch_size=batch_size, shuffle=False,
            num_workers=2, pin_memory=True,
        ),
        'combined': DataLoader(
            val_combined, batch_size=batch_size, shuffle=False,
            num_workers=2, pin_memory=True,
        ),
    }

    return train_loader, val_loaders
