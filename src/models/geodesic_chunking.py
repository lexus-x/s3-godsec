"""
Geodesic Action Chunking on SE(3).

Novel contribution: predicts temporally-coherent action chunks on the
SE(3) manifold via geodesic interpolation between learned anchor poses.

Unlike existing VLAs (Octo, OpenVLA, π0) that predict H independent
Euclidean action vectors, and unlike RFMP (Braun et al. 2024) that
predicts single SE(3) poses, this module:

1. Predicts K anchor poses on SE(3) (K << H)
2. Interpolates H actions along geodesics between anchors
3. Respects the coupled rotation-translation geometry of SE(3)

Key insight: robot manipulation trajectories are smooth on SE(3),
so H actions can be compressed to K geodesic anchors + interpolation.
This gives temporal consistency for free, unlike chunked Euclidean
prediction which can cross the antipodal boundary.

Architecture:
    h (VLA hidden state) → MLP → K anchor se(3) vectors → exp → SE(3) anchors
    Anchors → geodesic interpolation → H action poses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.se3_utils import exp_se3, log_se3, inverse_se3, geodesic_interpolation


class GeodesicChunkPredictor(nn.Module):
    """
    Predicts H action poses on SE(3) via K geodesic anchors.

    The model learns to predict K anchor poses in se(3) (Lie algebra),
    then interpolates H actions along geodesics between consecutive
    anchors. This enforces temporal smoothness on the manifold.

    For K=2: predicts start and end pose, linear geodesic interpolation
    For K=4: predicts 4 waypoints, piecewise geodesic
    For K=H: degenerates to independent prediction (no temporal prior)

    Args:
        hidden_dim: VLA hidden state dimension
        chunk_size: H — number of output actions
        n_anchors: K — number of geodesic anchor points (must divide H-1)
        head_hidden_dim: MLP hidden dimension
        n_layers: MLP depth
    """

    def __init__(
        self,
        hidden_dim: int,
        chunk_size: int = 8,
        n_anchors: int = 2,
        head_hidden_dim: int = 256,
        n_layers: int = 3,
    ):
        super().__init__()

        assert (chunk_size - 1) % (n_anchors - 1) == 0, \
            f"chunk_size-1 ({chunk_size-1}) must be divisible by n_anchors-1 ({n_anchors-1})"

        self.chunk_size = chunk_size
        self.n_anchors = n_anchors
        self.steps_per_segment = (chunk_size - 1) // (n_anchors - 1)

        # Predict K anchor poses in se(3) from VLA hidden state
        input_dim = hidden_dim
        layers = []
        for i in range(n_layers):
            in_d = input_dim if i == 0 else head_hidden_dim
            layers.extend([
                nn.Linear(in_d, head_hidden_dim),
                nn.GELU(),
                nn.LayerNorm(head_hidden_dim),
            ])
        self.anchor_net = nn.Sequential(*layers)

        # Output: K anchors × 6 (se(3) vectors)
        self.anchor_proj = nn.Linear(head_hidden_dim, n_anchors * 6)

        # Initialize small for stable training
        nn.init.zeros_(self.anchor_proj.weight)
        nn.init.zeros_(self.anchor_proj.bias)

        # Gripper head (separate, Euclidean)
        self.gripper_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, chunk_size),  # one gripper per action
            nn.Sigmoid(),
        )

    def forward(self, h):
        """
        Predict action chunk on SE(3).

        Args:
            h: [B, hidden_dim] VLA hidden state

        Returns:
            actions: [B, H, 4, 4] action chunk on SE(3)
            gripper: [B, H, 1] gripper actions
        """
        B = h.shape[0]
        device = h.device
        dtype = h.dtype

        # Predict K anchor poses in se(3)
        features = self.anchor_net(h)  # [B, head_hidden_dim]
        anchors_se3 = self.anchor_proj(features)  # [B, K*6]
        anchors_se3 = anchors_se3.view(B, self.n_anchors, 6)  # [B, K, 6]

        # Convert anchors to SE(3) matrices
        anchors = exp_se3(anchors_se3.reshape(-1, 6))  # [B*K, 4, 4]
        anchors = anchors.view(B, self.n_anchors, 4, 4)  # [B, K, 4, 4]

        # Geodesic interpolation between consecutive anchors
        actions = []
        for k in range(self.n_anchors - 1):
            T0 = anchors[:, k]      # [B, 4, 4]
            T1 = anchors[:, k + 1]  # [B, 4, 4]

            # Interpolation parameters for this segment
            # Include start, exclude end (except for last segment)
            n_steps = self.steps_per_segment
            if k == self.n_anchors - 2:
                n_steps += 1  # include endpoint for last segment

            t_vals = torch.linspace(0, 1, n_steps, device=device, dtype=dtype)
            t_vals = t_vals.unsqueeze(0).expand(B, -1)  # [B, n_steps]

            for t in t_vals.T:  # iterate over time steps
                T_t = geodesic_interpolation(T0, T1, t.unsqueeze(-1))  # [B, 4, 4]
                actions.append(T_t)

        actions = torch.stack(actions, dim=1)  # [B, H, 4, 4]

        # Gripper
        gripper = self.gripper_head(h).unsqueeze(-1)  # [B, H, 1]

        return actions, gripper

    def training_loss(self, h, target_actions, target_gripper=None):
        """
        Geodesic chunk loss: MSE in se(3) for each predicted action.

        Args:
            h: [B, hidden_dim] VLA hidden state
            target_actions: [B, H, 4, 4] target action chunk
            target_gripper: [B, H, 1] optional gripper targets

        Returns:
            loss: scalar
            loss_dict: dict with loss components
        """
        pred_actions, pred_gripper = self.forward(h)

        B, H = pred_actions.shape[:2]

        # Geodesic loss per action: ||log(pred^{-1} * target)||²
        pred_flat = pred_actions.reshape(-1, 4, 4)
        target_flat = target_actions.reshape(-1, 4, 4)
        T_rel = torch.bmm(inverse_se3(pred_flat), target_flat)
        xi = log_se3(T_rel)  # [B*H, 6]

        rot_loss = xi[:, :3].pow(2).sum(-1).mean()
        trans_loss = xi[:, 3:].pow(2).sum(-1).mean()
        chunk_loss = rot_loss + trans_loss

        loss = chunk_loss
        loss_dict = {
            'chunk_loss': chunk_loss.item(),
            'chunk_rot_loss': rot_loss.item(),
            'chunk_trans_loss': trans_loss.item(),
        }

        if target_gripper is not None:
            gripper_loss = F.binary_cross_entropy(
                pred_gripper.reshape(-1, 1),
                target_gripper.reshape(-1, 1),
            )
            loss = loss + gripper_loss
            loss_dict['gripper_loss'] = gripper_loss.item()

        loss_dict['total_loss'] = loss.item()
        return loss, loss_dict
