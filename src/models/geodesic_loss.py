"""
Geodesic loss functions for SE(3) action prediction.
"""

import torch
import torch.nn as nn
from utils.se3_utils import log_se3, inverse_se3, geodesic_distance, geodesic_distance_rotation_only


class GeodesicMSELoss(nn.Module):
    """
    Geodesic MSE loss on SE(3).
    
    L = ||log(X_pred^{-1} * X_target)||²
    
    This is the natural loss function for SE(3) predictions,
    measuring the squared geodesic distance in the Lie algebra.
    """
    
    def __init__(self, rotation_weight=1.0, translation_weight=1.0):
        """
        Args:
            rotation_weight: Weight for rotation component
            translation_weight: Weight for translation component
        """
        super().__init__()
        self.rotation_weight = rotation_weight
        self.translation_weight = translation_weight
    
    def forward(self, X_pred, X_target):
        """
        Args:
            X_pred: [B, 4, 4] Predicted SE(3) poses
            X_target: [B, 4, 4] Target SE(3) poses
        
        Returns:
            loss: scalar
        """
        T_rel = torch.bmm(inverse_se3(X_pred), X_target)
        xi = log_se3(T_rel)  # [B, 6]
        
        omega = xi[:, :3]  # rotation component
        v = xi[:, 3:]      # translation component
        
        rot_loss = torch.norm(omega, dim=-1).pow(2).mean()
        trans_loss = torch.norm(v, dim=-1).pow(2).mean()
        
        loss = self.rotation_weight * rot_loss + self.translation_weight * trans_loss
        
        return loss


class GeodesicDistanceLoss(nn.Module):
    """
    Geodesic distance loss on SE(3).
    
    L = ||log(X_pred^{-1} * X_target)||
    
    Uses the norm of the Lie algebra vector as the loss.
    """
    
    def forward(self, X_pred, X_target):
        """
        Args:
            X_pred: [B, 4, 4] Predicted SE(3) poses
            X_target: [B, 4, 4] Target SE(3) poses
        
        Returns:
            loss: scalar
        """
        dist = geodesic_distance(X_pred, X_target)
        return dist.mean()


class GeodesicHuberLoss(nn.Module):
    """
    Geodesic Huber loss on SE(3).
    
    Combines L1 and L2 loss for robustness to outliers.
    """
    
    def __init__(self, delta=0.1):
        super().__init__()
        self.delta = delta
    
    def forward(self, X_pred, X_target):
        """
        Args:
            X_pred: [B, 4, 4] Predicted SE(3) poses
            X_target: [B, 4, 4] Target SE(3) poses
        
        Returns:
            loss: scalar
        """
        dist = geodesic_distance(X_pred, X_target)
        
        # Huber loss
        loss = torch.where(
            dist < self.delta,
            0.5 * dist.pow(2),
            self.delta * (dist - 0.5 * self.delta)
        )
        
        return loss.mean()


class FlowMatchingLoss(nn.Module):
    """
    Riemannian flow matching loss on SE(3).
    
    This is the primary training loss for the SE(3) flow head.
    It computes the MSE between predicted and target velocities
    in the Lie algebra, sampled along geodesics.
    """
    
    def forward(self, v_pred, v_target):
        """
        Args:
            v_pred: [B, 6] Predicted velocities in se(3)
            v_target: [B, 6] Target velocities in se(3)
        
        Returns:
            loss: scalar
        """
        return torch.nn.functional.mse_loss(v_pred, v_target)


class RotationAngleLoss(nn.Module):
    """
    Loss on the rotation angle only (ignoring translation).
    
    Useful for ablation studies comparing rotation-specific improvements.
    """
    
    def forward(self, X_pred, X_target):
        """
        Args:
            X_pred: [B, 4, 4] Predicted SE(3) poses
            X_target: [B, 4, 4] Target SE(3) poses
        
        Returns:
            loss: scalar (mean rotation angle error in radians)
        """
        angles = geodesic_distance_rotation_only(X_pred, X_target)
        return angles.mean()
