"""
Compact VLA Backbone for SE(3)-VLA.

Builds a Vision-Language-Action model from smaller pretrained components
to stay under 400M total parameters while maintaining strong performance.

Architecture:
    SigLIP-Base (frozen, 87M) → visual features
    SmolLM-135M (trainable) → language understanding
    Cross-Attention Adapter (~10M) → vision-language fusion
    SE(3) Flow Head (~20M) → action prediction

Total: ~252M parameters (well under 400M budget)

Why this works:
    - SigLIP provides strong visual grounding (better than training from scratch)
    - SmolLM is compact but capable for instruction following
    - SE(3) head's geometric inductive bias compensates for smaller backbone
    - The key advantage is on rotation-heavy tasks where 1B+ models with
      Euclidean heads fail due to antipodal discontinuity

Usage:
    backbone = CompactVLABackbone.from_pretrained()
    h = backbone.encode(observations, language)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class VisionEncoder(nn.Module):
    """
    Frozen vision encoder using SigLIP or DINOv2.

    Extracts spatial visual features from RGB images.
    """

    def __init__(self, model_name: str = "siglip-base", freeze: bool = True):
        super().__init__()
        self.model_name = model_name
        self.freeze = freeze

        if model_name == "siglip-base":
            self._init_siglip_base()
        elif model_name == "siglip-small":
            self._init_siglip_small()
        elif model_name == "dinov2-small":
            self._init_dinov2_small()
        elif model_name == "dinov2-base":
            self._init_dinov2_base()
        else:
            raise ValueError(f"Unknown vision model: {model_name}")

        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def _init_siglip_base(self):
        """SigLIP base patch16-224: ~87M params."""
        try:
            from transformers import SiglipVisionModel
            self.encoder = SiglipVisionModel.from_pretrained(
                "google/siglip-base-patch16-224"
            )
            self.output_dim = 768
            self._use_hf = True
        except ImportError:
            self._init_fallback_vit(768, "siglip-base")

    def _init_siglip_small(self):
        """SigLIP small patch16-224: ~38M params."""
        try:
            from transformers import SiglipVisionModel
            self.encoder = SiglipVisionModel.from_pretrained(
                "google/siglip-so400m-patch14-384"
            )
            self.output_dim = 768
            self._use_hf = True
        except ImportError:
            self._init_fallback_vit(384, "siglip-small")

    def _init_dinov2_small(self):
        """DINOv2 small: ~22M params."""
        try:
            from transformers import Dinov2Model
            self.encoder = Dinov2Model.from_pretrained("facebook/dinov2-small")
            self.output_dim = 384
            self._use_hf = True
        except ImportError:
            self._init_fallback_vit(384, "dinov2-small")

    def _init_dinov2_base(self):
        """DINOv2 base: ~86M params."""
        try:
            from transformers import Dinov2Model
            self.encoder = Dinov2Model.from_pretrained("facebook/dinov2-base")
            self.output_dim = 768
            self._use_hf = True
        except ImportError:
            self._init_fallback_vit(768, "dinov2-base")

    def _init_fallback_vit(self, embed_dim: int, name: str):
        """Fallback ViT when transformers not available."""
        print(f"  [WARN] transformers not available, using fallback ViT for {name}")
        self._use_hf = False
        self.output_dim = embed_dim

        # Lightweight ViT-like CNN fallback
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(8),
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, embed_dim),
            nn.GELU(),
        )

    def forward(self, images):
        """
        Extract visual features.

        Args:
            images: [B, 3, H, W] RGB images

        Returns:
            features: [B, output_dim] visual features
        """
        if self.freeze:
            with torch.no_grad():
                return self._forward(images)
        return self._forward(images)

    def _forward(self, images):
        if self._use_hf:
            outputs = self.encoder(images)
            if hasattr(outputs, 'last_hidden_state'):
                # Pool over spatial tokens
                return outputs.last_hidden_state.mean(dim=1)
            elif hasattr(outputs, 'pooler_output'):
                return outputs.pooler_output
            else:
                return outputs[0].mean(dim=1)
        else:
            return self.encoder(images)


class LanguageEncoder(nn.Module):
    """
    Compact language encoder using SmolLM or GPT-2.

    Tokenizes text instructions and produces language features.
    """

    def __init__(self, model_name: str = "smolLM-135M", freeze: bool = False):
        super().__init__()
        self.model_name = model_name
        self.freeze = freeze

        if model_name == "smolLM-135M":
            self._init_smolLM()
        elif model_name == "gpt2-small":
            self._init_gpt2()
        else:
            self._init_fallback()

        if freeze:
            for param in self.lm.parameters():
                param.requires_grad = False

    def _init_smolLM(self):
        """SmolLM 135M: compact but capable language model."""
        try:
            from transformers import AutoModelForCausalLM
            self.lm = AutoModelForCausalLM.from_pretrained(
                "HuggingFaceTB/SmolLM-135M",
                torch_dtype=torch.float32,
            )
            self.output_dim = 768
            self._use_hf = True
        except ImportError:
            self._init_fallback()

    def _init_gpt2(self):
        """GPT-2 small 124M."""
        try:
            from transformers import GPT2Model
            self.lm = GPT2Model.from_pretrained("gpt2")
            self.output_dim = 768
            self._use_hf = True
        except ImportError:
            self._init_fallback()

    def _init_fallback(self):
        """Fallback: simple embedding + transformer."""
        print(f"  [WARN] Using fallback language encoder")
        self._use_hf = False
        self.output_dim = 256
        self.vocab_size = 32000
        self.embed = nn.Embedding(self.vocab_size, 256)
        self.pos_embed = nn.Parameter(torch.randn(1, 128, 256) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=8, dim_feedforward=512,
            batch_first=True, norm_first=True
        )
        self.lm = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.pool = nn.Linear(256, 256)

    def forward(self, language_tokens):
        """
        Encode language instruction.

        Args:
            language_tokens: [B, L] token IDs or [B, L] already tokenized

        Returns:
            features: [B, output_dim] language features
        """
        if self.freeze:
            with torch.no_grad():
                return self._forward(language_tokens)
        return self._forward(language_tokens)

    def _forward(self, tokens):
        if self._use_hf:
            outputs = self.lm(tokens, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]
            return last_hidden.mean(dim=1)
        else:
            if tokens.dim() == 1:
                tokens = tokens.unsqueeze(0)
            B, L = tokens.shape
            x = self.embed(tokens) + self.pos_embed[:, :L, :]
            x = self.lm(x)
            return self.pool(x.mean(dim=1))


class VisionLanguageFusion(nn.Module):
    """
    Cross-attention fusion between visual and language features.

    Produces a unified hidden state for the action head.
    """

    def __init__(
        self,
        vision_dim: int,
        language_dim: int,
        hidden_dim: int,
        n_heads: int = 8,
        n_layers: int = 2,
    ):
        super().__init__()

        # Project to common dimension
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.language_proj = nn.Linear(language_dim, hidden_dim)

        # Cross-attention layers
        self.cross_attn_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.cross_attn_layers.append(
                nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=n_heads,
                    batch_first=True,
                    norm_first=True,
                )
            )

        # Final MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.output_dim = hidden_dim

    def forward(self, vision_features, language_features):
        """
        Fuse vision and language features.

        Args:
            vision_features: [B, vision_dim]
            language_features: [B, language_dim]

        Returns:
            h: [B, hidden_dim] fused representation
        """
        # Project
        v = self.vision_proj(vision_features).unsqueeze(1)  # [B, 1, H]
        l = self.language_proj(language_features).unsqueeze(1)  # [B, 1, H]

        # Cross-attention: language attends to vision
        x = l
        for attn in self.cross_attn_layers:
            x, _ = attn(query=x, key=v, value=v)

        # Final MLP
        x = x.squeeze(1)  # [B, H]
        h = self.mlp(x)

        return h


class CompactVLABackbone(nn.Module):
    """
    Complete compact VLA backbone under 400M parameters.

    Combines vision encoder, language encoder, and fusion module
    into a unified backbone for the SE(3) action head.

    Target architecture: SigLIP-Base + SmolLM-135M + SE(3) Head
    Total: ~252M parameters

    Key insight: The SE(3) head's geometric inductive bias on rotations
    compensates for the smaller backbone, especially on rotation-heavy
    tasks where 1B+ models with Euclidean heads fail.
    """

    def __init__(
        self,
        vision_model: str = "siglip-base",
        language_model: str = "smolLM-135M",
        hidden_dim: int = 768,
        freeze_vision: bool = True,
        freeze_language: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Vision encoder
        self.vision_encoder = VisionEncoder(
            model_name=vision_model, freeze=freeze_vision
        )

        # Language encoder
        self.language_encoder = LanguageEncoder(
            model_name=language_model, freeze=freeze_language
        )

        # Fusion
        self.fusion = VisionLanguageFusion(
            vision_dim=self.vision_encoder.output_dim,
            language_dim=self.language_encoder.output_dim,
            hidden_dim=hidden_dim,
        )

    @classmethod
    def from_pretrained(
        cls,
        vision_model: str = "siglip-base",
        language_model: str = "smolLM-135M",
        hidden_dim: int = 768,
        device: str = "cpu",
    ):
        """Load pretrained compact VLA backbone."""
        backbone = cls(
            vision_model=vision_model,
            language_model=language_model,
            hidden_dim=hidden_dim,
        )
        return backbone.to(device)

    def encode(self, observations, language_instruction):
        """
        Encode observations and language into hidden state.

        Args:
            observations: dict with 'image' → [B, 3, H, W]
            language_instruction: [B, L] token IDs

        Returns:
            h: [B, hidden_dim]
        """
        # Extract image
        if isinstance(observations, dict):
            images = observations.get('image')
            if images is None:
                images = next(iter(observations.values()))
        else:
            images = observations

        # Encode vision
        vision_features = self.vision_encoder(images)

        # Encode language
        language_features = self.language_encoder(language_instruction)

        # Fuse
        h = self.fusion(vision_features, language_features)

        return h

    def forward(self, observations, language_instruction):
        return self.encode(observations, language_instruction)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable,
        }
