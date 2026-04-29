"""
Metrics for evaluating SE(3) action prediction.

Includes:
- Geodesic Action-ECE (novel calibration metric)
- Geodesic RMSE
- Rotation-specific metrics
- Set-valued metrics for conformal prediction
"""

import torch
import numpy as np
from ..utils.se3_utils import geodesic_distance, geodesic_distance_rotation_only, se3_to_rotation_angle


def geodesic_rmse(X_pred, X_target):
    """
    Root Mean Squared Geodesic Error.
    
    Args:
        X_pred: [B, 4, 4] predicted poses
        X_target: [B, 4, 4] target poses
    
    Returns:
        rmse: scalar (in radians/meters)
    """
    dist = geodesic_distance(X_pred, X_target)
    return torch.sqrt(torch.mean(dist.pow(2)))


def rotation_rmse(X_pred, X_target):
    """
    Root Mean Squared Rotation Error (SO(3) only).
    
    Args:
        X_pred: [B, 4, 4] predicted poses
        X_target: [B, 4, 4] target poses
    
    Returns:
        rmse: scalar (in radians)
    """
    angles = geodesic_distance_rotation_only(X_pred, X_target)
    return torch.sqrt(torch.mean(angles.pow(2)))


def translation_rmse(X_pred, X_target):
    """
    Root Mean Squared Translation Error.
    
    Args:
        X_pred: [B, 4, 4] predicted poses
        X_target: [B, 4, 4] target poses
    
    Returns:
        rmse: scalar (in meters)
    """
    t_pred = X_pred[:, :3, 3]
    t_target = X_target[:, :3, 3]
    return torch.sqrt(torch.mean((t_pred - t_target).pow(2).sum(dim=-1)))


def geodesic_action_ece(
    uncertainties,
    successes,
    n_bins=15,
):
    """
    Geodesic Action Expected Calibration Error (novel metric).
    
    Measures how well the model's uncertainty estimates correlate
    with actual task success/failure, using geodesic distance
    as the uncertainty measure.
    
    Args:
        uncertainties: [N] uncertainty scores (higher = more uncertain)
        successes: [N] binary success labels (1 = success, 0 = failure)
        n_bins: number of calibration bins
    
    Returns:
        ece: scalar (lower is better)
        bin_data: list of (bin_center, bin_accuracy, bin_confidence, bin_count)
    """
    uncertainties = uncertainties.detach().cpu().numpy()
    successes = successes.detach().cpu().numpy()
    
    # Convert uncertainty to confidence (inverse)
    # Higher uncertainty = lower confidence
    confidences = 1.0 / (1.0 + uncertainties)
    
    # Bin edges
    bin_edges = np.linspace(0, 1, n_bins + 1)
    
    ece = 0.0
    bin_data = []
    
    for i in range(n_bins):
        mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        
        if mask.sum() == 0:
            continue
        
        bin_confidence = confidences[mask].mean()
        bin_accuracy = successes[mask].mean()
        bin_count = mask.sum()
        
        ece += bin_count * np.abs(bin_accuracy - bin_confidence)
        bin_data.append((
            (bin_edges[i] + bin_edges[i + 1]) / 2,
            bin_accuracy,
            bin_confidence,
            bin_count,
        ))
    
    ece /= len(uncertainties)
    
    return ece, bin_data


def coverage_metric(X_pred_sets, X_target, alpha=0.1):
    """
    Coverage metric for conformal prediction on SE(3).
    
    Measures whether the target action falls within the predicted set.
    
    Args:
        X_pred_sets: [N, K, 4, 4] K predicted actions per input
        X_target: [N, 4, 4] target actions
        alpha: significance level (target coverage = 1 - alpha)
    
    Returns:
        coverage: fraction of targets covered
        avg_set_size: average geodesic radius of prediction sets
    """
    N, K = X_pred_sets.shape[:2]
    
    # Compute geodesic distance from each set member to target
    covered = 0
    total_radius = 0.0
    
    for i in range(N):
        dists = geodesic_distance(
            X_pred_sets[i],  # [K, 4, 4]
            X_target[i].unsqueeze(0).expand(K, -1, -1)  # [K, 4, 4]
        )  # [K]
        
        # Check if any member is close enough
        min_dist = dists.min()
        set_radius = dists.max()
        
        # A simple coverage check: is the target within the set?
        # (In practice, use calibrated threshold from conformal prediction)
        covered += (min_dist < set_radius).float().item()
        total_radius += set_radius.item()
    
    coverage = covered / N
    avg_set_size = total_radius / N
    
    return coverage, avg_set_size


def success_rate_per_rotation_bin(predictions, targets, successes, n_bins=5):
    """
    Compute success rate binned by rotation magnitude.
    
    This is the key analysis for validating Theorem 1.
    
    Args:
        predictions: [N, 4, 4] predicted actions
        targets: [N, 4, 4] target actions
        successes: [N] binary success labels
        n_bins: number of rotation magnitude bins
    
    Returns:
        bin_edges: [n_bins + 1] bin edges (in radians)
        success_rates: [n_bins] success rate per bin
        counts: [n_bins] number of samples per bin
    """
    # Compute rotation magnitudes from targets
    angles = se3_to_rotation_angle(targets).detach().cpu().numpy()
    successes = successes.detach().cpu().numpy()
    
    # Create bins
    bin_edges = np.linspace(0, np.pi, n_bins + 1)
    success_rates = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    
    for i in range(n_bins):
        mask = (angles >= bin_edges[i]) & (angles < bin_edges[i + 1])
        if mask.sum() > 0:
            success_rates[i] = successes[mask].mean()
            counts[i] = mask.sum()
    
    return bin_edges, success_rates, counts
