"""
Octo VLA Adapter — plugs SE(3) action head into pretrained Octo.

This module wraps a pretrained Octo model, replacing its Euclidean
action head with the SE(3) flow matching head while keeping the
backbone frozen.
"""

import torch
import torch.nn as nn
from models.se3_action_head import SE3ActionPredictor


class OctoSE3(nn.Module):
    """
    Octo VLA with SE(3) flow matching action head.
    
    Architecture:
        [Pretrained Octo Backbone (frozen)]
                ↓
           [Hidden State h]
                ↓
        [SE(3) Flow Matching Head (trainable)]
                ↓
           [Action T ∈ SE(3)]
    
    The backbone is completely frozen. Only the SE(3) action head
    (~20M parameters) is trained.
    """
    
    def __init__(
        self,
        octo_model,
        hidden_dim: int = 768,
        head_hidden_dim: int = 256,
        n_layers: int = 4,
        n_flow_steps_train: int = 10,
        n_flow_steps_eval: int = 10,
        source_scale: float = 0.1,
        freeze_backbone: bool = True,
    ):
        """
        Args:
            octo_model: Pretrained Octo model instance
            hidden_dim: Octo's hidden state dimension
            head_hidden_dim: Flow head hidden dimension
            n_layers: Flow head MLP layers
            n_flow_steps_train: Flow integration steps during training
            n_flow_steps_eval: Flow integration steps during evaluation
            source_scale: Source distribution scale
            freeze_backbone: Whether to freeze the backbone
        """
        super().__init__()
        
        self.octo = octo_model
        self.hidden_dim = hidden_dim
        self.n_flow_steps_train = n_flow_steps_train
        self.n_flow_steps_eval = n_flow_steps_eval
        
        # Freeze backbone
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for param in self.octo.parameters():
                param.requires_grad = False
        
        # SE(3) action predictor (trainable)
        self.action_predictor = SE3ActionPredictor(
            hidden_dim=hidden_dim,
            head_hidden_dim=head_hidden_dim,
            n_layers=n_layers,
            source_scale=source_scale,
        )
    
    def encode(self, observations, language_instruction):
        """
        Get hidden state from the frozen Octo backbone.
        
        Args:
            observations: dict with observation data
            language_instruction: str or tokenized instruction
        
        Returns:
            h: [B, hidden_dim] hidden state
        """
        if self.freeze_backbone:
            with torch.no_grad():
                h = self.octo.encode(observations, language_instruction)
        else:
            h = self.octo.encode(observations, language_instruction)
        return h
    
    def forward(self, observations, language_instruction):
        """
        Full forward pass: encode observations → predict SE(3) action.
        
        Args:
            observations: dict with observation data
            language_instruction: str or tokenized instruction
        
        Returns:
            action: [B, 4, 4] predicted action on SE(3)
            gripper: [B, 1] gripper action in [0, 1]
        """
        h = self.encode(observations, language_instruction)
        
        n_steps = self.n_flow_steps_train if self.training else self.n_flow_steps_eval
        action, gripper = self.action_predictor.predict(h, n_steps=n_steps)
        
        return action, gripper
    
    def predict_chunk(self, observations, language_instruction, chunk_size=8):
        """
        Predict a chunk of future actions on SE(3).
        
        Args:
            observations: dict with observation data
            language_instruction: str or tokenized instruction
            chunk_size: number of future actions
        
        Returns:
            actions: [B, H, 4, 4] action chunk on SE(3)
            grippers: [B, H, 1] gripper actions
        """
        h = self.encode(observations, language_instruction)
        
        n_steps = self.n_flow_steps_train if self.training else self.n_flow_steps_eval
        actions, grippers = self.action_predictor.predict_chunk(
            h, chunk_size=chunk_size, n_steps=n_steps
        )
        
        return actions, grippers
    
    def compute_loss(self, observations, language_instruction, target_actions, target_gripper=None):
        """
        Compute training loss.
        
        Args:
            observations: dict with observation data
            language_instruction: str or tokenized instruction
            target_actions: [B, 4, 4] target SE(3) actions from demonstrations
            target_gripper: [B, 1] optional gripper targets
        
        Returns:
            loss: scalar
            loss_dict: dict with loss components
        """
        h = self.encode(observations, language_instruction)
        loss, loss_dict = self.action_predictor.training_loss(
            h, target_actions, target_gripper
        )
        return loss, loss_dict
    
    def trainable_parameters(self):
        """Return only the trainable parameters (SE(3) head)."""
        return [p for p in self.parameters() if p.requires_grad]
    
    def count_parameters(self):
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        return {
            'total': total,
            'trainable': trainable,
            'frozen': frozen,
        }


class OctoEuclideanBaseline(nn.Module):
    """
    Octo with Euclidean action head (baseline for comparison).
    
    This is the standard Octo architecture with a simple MLP
    predicting actions in R^7 (axis-angle + translation + gripper).
    """
    
    def __init__(self, octo_model, hidden_dim=768, freeze_backbone=True):
        super().__init__()
        
        self.octo = octo_model
        self.hidden_dim = hidden_dim
        self.freeze_backbone = freeze_backbone
        
        if freeze_backbone:
            for param in self.octo.parameters():
                param.requires_grad = False
        
        # Euclidean action head (simple MLP)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 7),  # 6D action + 1D gripper
        )
    
    def encode(self, observations, language_instruction):
        if self.freeze_backbone:
            with torch.no_grad():
                h = self.octo.encode(observations, language_instruction)
        else:
            h = self.octo.encode(observations, language_instruction)
        return h
    
    def forward(self, observations, language_instruction):
        h = self.encode(observations, language_instruction)
        action = self.action_head(h)
        
        # Split into action (axis-angle + translation) and gripper
        pose = action[:, :6]  # [B, 6] axis-angle + translation
        gripper = torch.sigmoid(action[:, 6:7])  # [B, 1]
        
        return pose, gripper
    
    def compute_loss(self, observations, language_instruction, target_pose, target_gripper=None):
        h = self.encode(observations, language_instruction)
        action = self.action_head(h)
        
        pose_loss = nn.functional.mse_loss(action[:, :6], target_pose)
        loss = pose_loss
        
        loss_dict = {'pose_loss': pose_loss.item()}
        
        if target_gripper is not None:
            gripper_loss = nn.functional.binary_cross_entropy(
                torch.sigmoid(action[:, 6:7]), target_gripper
            )
            loss = loss + gripper_loss
            loss_dict['gripper_loss'] = gripper_loss.item()
        
        loss_dict['total_loss'] = loss.item()
        return loss, loss_dict
    
    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def count_parameters(self):
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        return {
            'total': total,
            'trainable': trainable,
            'frozen': frozen,
        }
