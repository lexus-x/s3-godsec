"""
Mock Octo Backbone for standalone training validation.

Replaces the real pretrained Octo model with a lightweight CNN + MLP
that produces the same hidden state interface (h ∈ ℝ^hidden_dim).

This allows validating the entire SE(3) flow matching pipeline
without requiring the real Octo checkpoint.
"""

import torch
import torch.nn as nn


class MockOctoBackbone(nn.Module):
    """
    Lightweight mock of the Octo VLA backbone.

    Takes image observations and language tokens, produces a
    hidden state h ∈ ℝ^hidden_dim suitable for the action head.

    Architecture:
        Image → CNN → Flatten → FC
        Language → Embedding → Mean Pool
        [Image features, Language features] → MLP → h
    """

    def __init__(self, hidden_dim: int = 768, image_size: int = 224):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Simple CNN for image features
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=4, padding=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
        )

        # Simple language embedding (token IDs → features)
        self.language_encoder = nn.Sequential(
            nn.Embedding(1000, 64),  # small vocab for mock
        )
        self.language_pool = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
        )

        # Fusion MLP
        self.fusion = nn.Sequential(
            nn.Linear(512 + 256, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def encode(self, observations, language_instruction):
        """
        Encode observations and language into hidden state.

        Args:
            observations: dict with 'image' key → [B, 3, H, W] tensor
            language_instruction: [B, L] token IDs or ignored

        Returns:
            h: [B, hidden_dim] hidden state
        """
        # Image encoding
        if isinstance(observations, dict) and 'image' in observations:
            img = observations['image']
        elif isinstance(observations, dict):
            # take first available key
            img = next(iter(observations.values()))
        else:
            img = observations

        img_feat = self.image_encoder(img)  # [B, 512]

        # Language encoding (use random tokens if strings provided)
        B = img_feat.shape[0]
        device = img_feat.device

        if isinstance(language_instruction, (list, tuple)):
            lang_tokens = torch.randint(0, 1000, (B, 10), device=device)
        elif isinstance(language_instruction, torch.Tensor):
            lang_tokens = language_instruction
        else:
            lang_tokens = torch.randint(0, 1000, (B, 10), device=device)

        lang_emb = self.language_encoder(lang_tokens)  # [B, L, 64]
        lang_feat = self.language_pool(lang_emb.mean(dim=1))  # [B, 256]

        # Fuse
        h = self.fusion(torch.cat([img_feat, lang_feat], dim=-1))  # [B, hidden_dim]

        return h

    def forward(self, observations, language_instruction):
        return self.encode(observations, language_instruction)
