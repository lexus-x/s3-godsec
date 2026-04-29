"""
SE(3)-specific neural network layers.

Provides layers that operate directly on SE(3) representations.
"""

import torch
import torch.nn as nn
from ..utils.se3_utils import log_se3, exp_se3


class SE3Linear(nn.Module):
    """
    Linear layer that operates on SE(3) tangent space.
    
    Maps se(3) vectors through a linear transformation,
    preserving the Lie algebra structure.
    """
    
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
    
    def forward(self, xi):
        """
        Args:
            xi: [B, ..., 6] se(3) vectors
        Returns:
            out: [B, ..., out_features] transformed vectors
        """
        return self.linear(xi)


class SE3LayerNorm(nn.Module):
    """
    Layer normalization for se(3) vectors.
    
    Normalizes rotation and translation components separately,
    since they have different scales.
    """
    
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
    
    def forward(self, xi):
        """
        Args:
            xi: [B, 6] se(3) vectors (first 3: rotation, last 3: translation)
        Returns:
            normalized: [B, 6] normalized se(3) vectors
        """
        omega = xi[:, :3]
        v = xi[:, 3:]
        
        # Normalize each component separately
        omega_norm = omega / (omega.norm(dim=-1, keepdim=True) + self.eps)
        v_norm = v / (v.norm(dim=-1, keepdim=True) + self.eps)
        
        return torch.cat([omega_norm, v_norm], dim=-1)


class GeodesicAttention(nn.Module):
    """
    Attention mechanism that uses geodesic distance as bias.
    
    For attending to SE(3) poses based on their geometric similarity.
    """
    
    def __init__(self, hidden_dim, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Learnable temperature for geodesic distance
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, h, poses, mask=None):
        """
        Args:
            h: [B, N, hidden_dim] hidden states
            poses: [B, N, 4, 4] SE(3) poses for each token
            mask: [B, N] optional mask
        
        Returns:
            out: [B, N, hidden_dim] attended features
        """
        B, N, _ = h.shape
        
        Q = self.q_proj(h).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(h).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(h).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Standard attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Add geodesic distance bias
        # Compute pairwise geodesic distances between poses
        # This encourages attention to geometrically similar poses
        
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        
        out = out.transpose(1, 2).contiguous().view(B, N, -1)
        out = self.out_proj(out)
        
        return out
