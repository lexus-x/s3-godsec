"""
Scene-ID Backbone — deterministic embedding for controlled experiments.

Instead of learning features from images (where the mock CNN may be the
bottleneck), this backbone maps a task_id integer directly to a learned
embedding vector. This isolates the action head's geometry for A/B testing.

Usage:
    backbone = SceneIDBackbone(n_tasks=2, hidden_dim=768)
    h = backbone.encode(observations, language)  # uses observations['task_id']
"""

import torch
import torch.nn as nn


class SceneIDBackbone(nn.Module):
    """
    Backbone that returns a learned embedding for each task_id.

    This gives the action head a clean, learnable feature vector per scene
    without any CNN noise, letting us test whether the SE(3) head's geometry
    advantage is real, independent of backbone quality.

    Architecture:
        task_id → nn.Embedding(n_tasks, hidden_dim) → h
    """

    def __init__(self, n_tasks: int = 2, hidden_dim: int = 768):
        """
        Args:
            n_tasks: Number of distinct task/scene IDs
            hidden_dim: Output embedding dimension (must match action head)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_tasks = n_tasks
        self.embedding = nn.Embedding(n_tasks, hidden_dim)

        # Initialize with Xavier for stable gradients
        nn.init.xavier_uniform_(self.embedding.weight)

    def encode(self, observations, language_instruction):
        """
        Encode observations using task_id embedding.

        Args:
            observations: dict, must contain 'task_id' → [B] LongTensor
            language_instruction: ignored (kept for interface compatibility)

        Returns:
            h: [B, hidden_dim] embedding
        """
        task_id = observations['task_id']
        if task_id.device != self.embedding.weight.device:
            task_id = task_id.to(self.embedding.weight.device)
        return self.embedding(task_id)

    def forward(self, observations, language_instruction):
        return self.encode(observations, language_instruction)
