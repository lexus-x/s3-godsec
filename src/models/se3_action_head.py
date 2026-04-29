"""
SE(3) Flow Matching Action Head for Vision-Language-Action Models.

This module implements a Riemannian flow matching head that predicts
robot actions on the Special Euclidean group SE(3) instead of flat
Euclidean space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ..utils.se3_utils import exp_se3, log_se3, inverse_se3, geodesic_interpolation, sample_se3_gaussian


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional encoding for the time parameter t ∈ [0, 1]."""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        """
        Args:
            t: [B, 1] time values in [0, 1]
        Returns:
            emb: [B, dim] time embeddings
        """
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=t.dtype) * -emb)
        emb = t * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class SE3FlowHead(nn.Module):
    """
    Riemannian flow matching head on SE(3).
    
    Predicts a velocity field in the Lie algebra se(3) conditioned on
    the VLA's hidden state, the current pose on SE(3), and time.
    
    The velocity is integrated to produce the final action on SE(3).
    
    Architecture:
        Input: [h (VLA hidden state), xi_t (SE(3) log coords), t (time)]
        → MLP with GELU + LayerNorm
        → Output: v ∈ se(3) (6D: 3 rotation + 3 translation)
    """
    
    def __init__(
        self,
        hidden_dim: int,
        head_hidden_dim: int = 256,
        n_layers: int = 4,
        time_embed_dim: int = 64,
        dropout: float = 0.0,
    ):
        """
        Args:
            hidden_dim: Dimension of the VLA's hidden state
            head_hidden_dim: Hidden dimension of the flow head MLP
            n_layers: Number of MLP layers
            time_embed_dim: Dimension of time embedding
            dropout: Dropout rate
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.head_hidden_dim = head_hidden_dim
        
        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        
        # Input: hidden state + SE(3) log coords (6D) + time embedding
        input_dim = hidden_dim + 6 + time_embed_dim
        
        # Build MLP
        layers = []
        for i in range(n_layers):
            in_dim = input_dim if i == 0 else head_hidden_dim
            layers.extend([
                nn.Linear(in_dim, head_hidden_dim),
                nn.GELU(),
                nn.LayerNorm(head_hidden_dim),
            ])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        self.net = nn.Sequential(*layers)
        
        # Output projection: velocity in se(3) (6D)
        self.output_proj = nn.Linear(head_hidden_dim, 6)
        
        # Initialize output to zero for stable training start
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
    
    def forward(self, h, X_t, t):
        """
        Predict velocity in se(3) conditioned on VLA state and current pose.
        
        Args:
            h: [B, hidden_dim] VLA hidden state
            X_t: [B, 4, 4] Current pose on SE(3)
            t: [B, 1] Time in [0, 1]
        
        Returns:
            v: [B, 6] Velocity in se(3) (first 3: rotation, last 3: translation)
        """
        # Convert SE(3) matrix to Lie algebra coordinates
        xi_t = log_se3(X_t)  # [B, 6]
        
        # Time embedding
        t_emb = self.time_embed(t)  # [B, time_embed_dim]
        
        # Concatenate all inputs
        x = torch.cat([h, xi_t, t_emb], dim=-1)  # [B, input_dim]
        
        # Predict velocity
        v = self.output_proj(self.net(x))  # [B, 6]
        
        return v


class SE3ActionPredictor(nn.Module):
    """
    Complete SE(3) action prediction module.
    
    Wraps the flow head with:
    1. Source distribution sampling
    2. Flow integration (Euler or midpoint)
    3. Gripper prediction (Euclidean, separate head)
    
    Usage:
        predictor = SE3ActionPredictor(hidden_dim=768)
        action_se3, gripper = predictor(h, n_steps=10)
    """
    
    def __init__(
        self,
        hidden_dim: int,
        head_hidden_dim: int = 256,
        n_layers: int = 4,
        time_embed_dim: int = 64,
        source_scale: float = 0.1,
    ):
        """
        Args:
            hidden_dim: VLA hidden state dimension
            head_hidden_dim: Flow head hidden dimension
            n_layers: Number of flow head layers
            time_embed_dim: Time embedding dimension
            source_scale: Scale of the source distribution (Gaussian on se(3))
        """
        super().__init__()
        
        self.flow_head = SE3FlowHead(
            hidden_dim=hidden_dim,
            head_hidden_dim=head_hidden_dim,
            n_layers=n_layers,
            time_embed_dim=time_embed_dim,
        )
        
        # Gripper head (Euclidean, separate from SE(3) flow)
        self.gripper_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        
        self.source_scale = source_scale
    
    def sample_source(self, batch_size, device, dtype=torch.float32):
        """Sample from the source distribution (Gaussian on se(3))."""
        xi = torch.randn(batch_size, 6, device=device, dtype=dtype) * self.source_scale
        return exp_se3(xi)
    
    def predict(self, h, n_steps=10, X_0=None):
        """
        Predict action on SE(3) by integrating the flow.
        
        Args:
            h: [B, hidden_dim] VLA hidden state
            n_steps: Number of integration steps (1 = fastest, 20 = most accurate)
            X_0: [B, 4, 4] Optional starting point (if None, sample from source)
        
        Returns:
            X_1: [B, 4, 4] Predicted action on SE(3)
            gripper: [B, 1] Gripper action in [0, 1]
        """
        batch_size = h.shape[0]
        device = h.device
        dtype = h.dtype
        
        # Sample source if not provided
        if X_0 is None:
            X_0 = self.sample_source(batch_size, device, dtype)
        
        # Integrate flow from X_0 to X_1
        dt = 1.0 / n_steps
        X_t = X_0
        
        for step in range(n_steps):
            t = torch.full((batch_size, 1), step * dt, device=device, dtype=dtype)
            v = self.flow_head(h, X_t, t)  # [B, 6]
            
            # Update: X_{t+dt} = X_t * exp(dt * v)
            X_t = torch.bmm(X_t, exp_se3(dt * v))
        
        # Gripper prediction (Euclidean)
        gripper = self.gripper_head(h)  # [B, 1]
        
        return X_t, gripper
    
    def predict_chunk(self, h, chunk_size=8, n_steps=10):
        """
        Predict a chunk of H future actions on SE(3).
        
        Each action in the chunk is predicted conditioned on the same
        hidden state but executed autoregressively.
        
        Args:
            h: [B, hidden_dim] VLA hidden state
            chunk_size: Number of future actions to predict
            n_steps: Flow integration steps per action
        
        Returns:
            actions: [B, H, 4, 4] Predicted action chunk on SE(3)
            gripper: [B, H, 1] Gripper actions
        """
        actions = []
        grippers = []
        
        X_t = None
        for i in range(chunk_size):
            X_t, gripper = self.predict(h, n_steps=n_steps, X_0=X_t)
            actions.append(X_t)
            grippers.append(gripper)
        
        actions = torch.stack(actions, dim=1)  # [B, H, 4, 4]
        grippers = torch.stack(grippers, dim=1)  # [B, H, 1]
        
        return actions, grippers
    
    def training_loss(self, h, X_target, gripper_target=None):
        """
        Compute the Riemannian flow matching loss on SE(3).
        
        The loss is the MSE between predicted and target velocities
        in the Lie algebra se(3), computed along the geodesic.
        
        Args:
            h: [B, hidden_dim] VLA hidden state
            X_target: [B, 4, 4] Target actions from demonstrations
            gripper_target: [B, 1] Optional gripper targets
        
        Returns:
            loss: scalar total loss
            loss_dict: dict with loss components for logging
        """
        batch_size = h.shape[0]
        device = h.device
        dtype = h.dtype
        
        # Sample source
        X_0 = self.sample_source(batch_size, device, dtype)
        
        # Compute geodesic from X_0 to X_target
        # xi = log(X_0^{-1} * X_target) — the Lie algebra vector
        T_rel = torch.bmm(inverse_se3(X_0), X_target)
        xi = log_se3(T_rel)  # [B, 6]
        
        # Sample random time
        t = torch.rand(batch_size, 1, device=device, dtype=dtype)
        
        # Geodesic interpolation: X_t = X_0 * exp(t * xi)
        X_t = torch.bmm(X_0, exp_se3(t * xi))
        
        # Target velocity (constant along geodesic in body frame)
        v_target = xi  # [B, 6]
        
        # Predicted velocity
        v_pred = self.flow_head(h, X_t, t)
        
        # Flow matching loss: MSE in the Lie algebra
        flow_loss = F.mse_loss(v_pred, v_target)
        
        loss = flow_loss
        loss_dict = {'flow_loss': flow_loss.item()}
        
        # Gripper loss (if targets provided)
        if gripper_target is not None:
            gripper_pred = self.gripper_head(h)
            gripper_loss = F.binary_cross_entropy(gripper_pred, gripper_target)
            loss = loss + gripper_loss
            loss_dict['gripper_loss'] = gripper_loss.item()
        
        loss_dict['total_loss'] = loss.item()
        
        return loss, loss_dict
