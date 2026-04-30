"""
Synthetic SE(3) demonstration dataset for training validation.

Generates synthetic image observations paired with SE(3) target actions,
simulating both rotation-heavy and translation-heavy manipulation tasks.

This validates that:
1. The flow matching loss converges
2. The SE(3) head learns to predict correct poses
3. Geodesic metrics decrease during training
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.se3_utils import exp_se3


class SyntheticSE3Dataset(Dataset):
    """
    Synthetic dataset of (observation, language, SE(3) target action) tuples.

    Generates:
    - Random RGB images with simple geometric patterns (circles, rectangles)
      whose spatial properties encode the target action
    - Language tokens (placeholder)
    - SE(3) target actions with configurable rotation/translation distributions

    Task types:
    - 'rotation_heavy': large rotations (θ ~ π/2 to π), small translations
    - 'translation_heavy': small rotations (θ ~ 0 to π/6), large translations
    - 'mixed': balanced rotation and translation
    """

    def __init__(
        self,
        n_samples: int = 5000,
        task_type: str = 'mixed',
        image_size: int = 224,
        seed: int = 42,
    ):
        """
        Args:
            n_samples: Number of samples to generate
            task_type: 'rotation_heavy', 'translation_heavy', or 'mixed'
            image_size: Size of generated images
            seed: Random seed for reproducibility
        """
        super().__init__()
        self.n_samples = n_samples
        self.task_type = task_type
        self.image_size = image_size

        rng = np.random.RandomState(seed)

        # Generate SE(3) target actions
        self.target_actions = self._generate_targets(rng)

        # Generate corresponding images that encode the target
        self.images = self._generate_images(rng)

        # Language tokens (placeholder — mock backbone handles this)
        self.language_tokens = torch.randint(0, 1000, (n_samples, 10))

        # Gripper targets (binary)
        self.gripper_targets = torch.bernoulli(
            torch.full((n_samples, 1), 0.5)
        )

    def _generate_targets(self, rng):
        """Generate SE(3) target actions based on task type."""
        xi_all = np.zeros((self.n_samples, 6), dtype=np.float32)

        if self.task_type == 'rotation_heavy':
            # Large rotations: θ ∈ [π/4, π]
            for i in range(self.n_samples):
                theta = rng.uniform(math.pi / 4, math.pi)
                axis = rng.randn(3)
                axis = axis / (np.linalg.norm(axis) + 1e-8)
                xi_all[i, :3] = axis * theta
                # Small translations
                xi_all[i, 3:] = rng.randn(3) * 0.05

        elif self.task_type == 'translation_heavy':
            # Small rotations: θ ∈ [0, π/6]
            for i in range(self.n_samples):
                theta = rng.uniform(0, math.pi / 6)
                axis = rng.randn(3)
                axis = axis / (np.linalg.norm(axis) + 1e-8)
                xi_all[i, :3] = axis * theta
                # Large translations
                xi_all[i, 3:] = rng.randn(3) * 0.3

        else:  # mixed
            for i in range(self.n_samples):
                theta = rng.uniform(0, math.pi * 0.8)
                axis = rng.randn(3)
                axis = axis / (np.linalg.norm(axis) + 1e-8)
                xi_all[i, :3] = axis * theta
                xi_all[i, 3:] = rng.randn(3) * 0.15

        # Convert to SE(3) matrices via exp map
        xi_tensor = torch.from_numpy(xi_all)
        T = exp_se3(xi_tensor)  # [N, 4, 4]

        return T

    def _generate_images(self, rng):
        """
        Generate synthetic images that are correlated with the target action.

        The images contain colored circles and rectangles whose positions
        and colors encode the rotation and translation components of the
        target action. This gives the model learnable visual-to-action
        correspondences.
        """
        images = np.zeros((self.n_samples, 3, self.image_size, self.image_size),
                          dtype=np.float32)

        for i in range(self.n_samples):
            img = np.zeros((3, self.image_size, self.image_size), dtype=np.float32)

            # Background: encode rotation component as hue
            R = self.target_actions[i, :3, :3].numpy()
            trace = np.clip(R[0, 0] + R[1, 1] + R[2, 2], -1, 3)
            hue = (trace + 1) / 4.0  # normalize to [0, 1]

            # Background gradient
            img[0] = hue * 0.3 + 0.1  # R channel
            img[1] = (1 - hue) * 0.3 + 0.1  # G channel
            img[2] = 0.2  # B channel

            # Translation → circle position
            t = self.target_actions[i, :3, 3].numpy()
            cx = int(np.clip((t[0] + 0.5) * self.image_size, 5, self.image_size - 5))
            cy = int(np.clip((t[1] + 0.5) * self.image_size, 5, self.image_size - 5))
            r = max(3, int(10 + t[2] * 20))

            # Draw circle (simple rasterization)
            Y, X = np.ogrid[:self.image_size, :self.image_size]
            mask = (X - cx) ** 2 + (Y - cy) ** 2 < r ** 2
            img[0][mask] = 0.8
            img[1][mask] = 0.3
            img[2][mask] = 0.1

            # Add rotation-encoding bar
            angle = np.arctan2(R[1, 0], R[0, 0])
            bar_len = 20
            bx = int(self.image_size / 2 + bar_len * np.cos(angle))
            by = int(self.image_size / 2 + bar_len * np.sin(angle))
            # Simple line (approximate with thick pixels)
            for step in range(bar_len):
                px = int(self.image_size / 2 + step * np.cos(angle))
                py = int(self.image_size / 2 + step * np.sin(angle))
                if 0 <= px < self.image_size and 0 <= py < self.image_size:
                    img[0, py, px] = 0.9
                    img[1, py, px] = 0.9
                    img[2, py, px] = 0.9

            # Add some noise
            img += rng.randn(*img.shape).astype(np.float32) * 0.02
            img = np.clip(img, 0, 1)

            images[i] = img

        return torch.from_numpy(images)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return {
            'observations': {'image': self.images[idx]},
            'language': self.language_tokens[idx],
            'target_actions': self.target_actions[idx],
            'target_gripper': self.gripper_targets[idx],
        }


def create_dataloaders(config, seed=42):
    """
    Create train and validation dataloaders from config.

    Args:
        config: dict with 'data' key containing dataset parameters
        seed: random seed

    Returns:
        train_loader, val_loader
    """
    data_cfg = config.get('data', {})
    batch_size = config.get('training', {}).get('batch_size', 32)
    image_size = data_cfg.get('image_size', [224, 224])
    if isinstance(image_size, list):
        image_size = image_size[0]

    n_train = data_cfg.get('n_train_samples', 5000)
    n_val = data_cfg.get('n_val_samples', 500)
    task_type = data_cfg.get('task_type', 'mixed')

    train_dataset = SyntheticSE3Dataset(
        n_samples=n_train,
        task_type=task_type,
        image_size=image_size,
        seed=seed,
    )

    val_dataset = SyntheticSE3Dataset(
        n_samples=n_val,
        task_type=task_type,
        image_size=image_size,
        seed=seed + 1000,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader
