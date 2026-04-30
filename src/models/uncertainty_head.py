"""
Uncertainty-Aware Flow Matching on SE(3).

Novel contribution: calibrated uncertainty quantification for VLA
action prediction using the generative nature of flow matching.

Unlike point-estimate VLA heads (Octo, OpenVLA, π0), this module:
1. Draws N samples from the learned flow → N candidate SE(3) actions
2. Computes geodesic variance on the manifold as uncertainty
3. Provides conformal prediction sets with coverage guarantees

This is the first VLA head with principled uncertainty on SE(3).

The key insight: flow matching is already a generative model —
sampling is nearly free (just run the ODE with different noise seeds).
We exploit this to get uncertainty without ensembles or MC dropout.

Mathematical foundation:
    Given N samples {T_1, ..., T_N} from the flow,
    the geodesic mean is: T̄ = argmin_T Σ d(T, T_i)²
    The geodesic variance is: σ² = (1/N) Σ d(T̄, T_i)²
    Conformal prediction set: {T : d(T, T̄) ≤ q_α}
    where q_α is calibrated on a held-out set for coverage 1-α.
"""

import torch
import torch.nn as nn
import math
from utils.se3_utils import (
    exp_se3, log_se3, inverse_se3,
    geodesic_distance, geodesic_interpolation,
)


class GeodesicStats:
    """
    Computes statistics on SE(3) from multiple samples.

    Uses iterative Fréchet mean computation on the manifold.
    """

    @staticmethod
    def frechet_mean(samples, n_iter=10):
        """
        Approximate Fréchet mean on SE(3) via iterative averaging.

        Args:
            samples: [N, B, 4, 4] or [B, N, 4, 4] — N samples per batch
            n_iter: number of iterations

        Returns:
            mean: [B, 4, 4] Fréchet mean on SE(3)
        """
        if samples.dim() == 3:
            # [N, 4, 4] — single batch element
            samples = samples.unsqueeze(1)
            squeeze = True
        else:
            squeeze = False

        # samples: [N, B, 4, 4]
        N, B = samples.shape[:2]

        # Initialize mean as first sample
        mean = samples[0]  # [B, 4, 4]

        for _ in range(n_iter):
            # Compute log map of each sample relative to current mean
            # mean^{-1} * sample_i
            mean_inv = inverse_se3(mean)  # [B, 4, 4]
            # Broadcast: [B, 4, 4] vs [N, B, 4, 4]
            rel = torch.bmm(
                mean_inv.unsqueeze(0).expand(N, -1, -1, -1).reshape(N * B, 4, 4),
                samples.reshape(N * B, 4, 4),
            )  # [N*B, 4, 4]

            xi = log_se3(rel)  # [N*B, 6]
            xi = xi.view(N, B, 6)  # [N, B, 6]

            # Average in tangent space
            delta = xi.mean(dim=0)  # [B, 6]

            # Update mean via exponential map
            mean = torch.bmm(mean, exp_se3(delta))  # [B, 4, 4]

        if squeeze:
            mean = mean.squeeze(0)
        return mean

    @staticmethod
    def geodesic_variance(samples, mean=None):
        """
        Geodesic variance: σ² = (1/N) Σ d(mean, T_i)²

        Args:
            samples: [N, B, 4, 4]
            mean: [B, 4, 4] (computed via Frechet_mean if None)

        Returns:
            variance: [B] geodesic variance per batch element
            mean: [B, 4, 4] Fréchet mean
        """
        if mean is None:
            mean = GeodesicStats.frechet_mean(samples)

        N, B = samples.shape[:2]

        # d(mean, T_i) for each sample
        mean_expanded = mean.unsqueeze(0).expand(N, -1, -1, -1)  # [N, B, 4, 4]
        dists = geodesic_distance(
            mean_expanded.reshape(N * B, 4, 4),
            samples.reshape(N * B, 4, 4),
        ).view(N, B)  # [N, B]

        variance = dists.pow(2).mean(dim=0)  # [B]
        return variance, mean

    @staticmethod
    def rotation_variance(samples, mean=None):
        """
        Rotation-only variance (SO(3) component).

        Args:
            samples: [N, B, 4, 4]
            mean: [B, 4, 4]

        Returns:
            rot_var: [B] rotation variance
        """
        from utils.se3_utils import geodesic_distance_rotation_only

        if mean is None:
            mean = GeodesicStats.frechet_mean(samples)

        N, B = samples.shape[:2]
        mean_expanded = mean.unsqueeze(0).expand(N, -1, -1, -1)

        dists = geodesic_distance_rotation_only(
            mean_expanded.reshape(N * B, 4, 4),
            samples.reshape(N * B, 4, 4),
        ).view(N, B)

        return dists.pow(2).mean(dim=0)


class UncertaintyAwareFlowHead(nn.Module):
    """
    Flow matching head with built-in uncertainty via multi-sample prediction.

    Wraps SE3ActionPredictor to:
    1. Draw N samples from the flow (different noise seeds)
    2. Compute geodesic mean and variance
    3. Return prediction + uncertainty score

    Args:
        base_predictor: SE3ActionPredictor instance
        n_samples: number of flow samples for uncertainty estimation
    """

    def __init__(self, base_predictor, n_samples: int = 10):
        super().__init__()
        self.predictor = base_predictor
        self.n_samples = n_samples

    def predict_with_uncertainty(self, h, n_steps=10):
        """
        Predict action on SE(3) with uncertainty estimate.

        Args:
            h: [B, hidden_dim] VLA hidden state
            n_steps: flow integration steps per sample

        Returns:
            mean_action: [B, 4, 4] Fréchet mean action
            gripper: [B, 1] gripper prediction
            uncertainty: [B] geodesic variance (higher = more uncertain)
            samples: [N, B, 4, 4] all samples (for conformal prediction)
        """
        B = h.shape[0]
        device = h.device

        # Draw N samples from the flow
        samples = []
        grippers = []
        for _ in range(self.n_samples):
            action, gripper = self.predictor.predict(h, n_steps=n_steps)
            samples.append(action)
            grippers.append(gripper)

        samples = torch.stack(samples, dim=0)  # [N, B, 4, 4]
        gripper = grippers[0]  # use first sample's gripper (deterministic head)

        # Compute Fréchet mean and geodesic variance
        variance, mean = GeodesicStats.geodesic_variance(samples)

        return mean, gripper, variance, samples

    def conformal_set(self, h, q_alpha, n_steps=10):
        """
        Construct a conformal prediction set on SE(3).

        The set is a geodesic ball of radius q_alpha around the mean.
        Any T within this ball is a valid action at coverage level 1-alpha.

        Args:
            h: [B, hidden_dim] VLA hidden state
            q_alpha: calibrated quantile (geodesic radius)
            n_steps: flow integration steps

        Returns:
            mean: [B, 4, 4] center of prediction set
            radius: scalar — q_alpha
            samples_in_set: [N', B, 4, 4] samples within the ball
        """
        mean, gripper, variance, samples = self.predict_with_uncertainty(
            h, n_steps=n_steps
        )

        N, B = samples.shape[:2]

        # Filter samples within the conformal ball
        dists = geodesic_distance(
            mean.unsqueeze(0).expand(N, -1, -1, -1).reshape(N * B, 4, 4),
            samples.reshape(N * B, 4, 4),
        ).view(N, B)  # [N, B]

        mask = dists <= q_alpha  # [N, B]

        return mean, gripper, q_alpha, mask, samples, variance


class ConformalCalibrator:
    """
    Calibrates conformal prediction radius on SE(3).

    Uses a held-out calibration set to find the geodesic radius
    q_α that achieves desired coverage level 1-α.

    This is distribution-free — no assumptions on the action distribution.
    """

    def __init__(self, alpha: float = 0.1):
        """
        Args:
            alpha: miscoverage level (coverage = 1 - alpha)
        """
        self.alpha = alpha
        self.q_alpha = None

    def calibrate(self, calibration_scores):
        """
        Compute the conformal quantile from calibration scores.

        Args:
            calibration_scores: [M] geodesic distances from mean to
                ground truth on the calibration set

        Sets self.q_alpha to the (1-alpha) quantile.
        """
        M = len(calibration_scores)
        level = math.ceil((1 - self.alpha) * (M + 1)) / M
        self.q_alpha = torch.quantile(calibration_scores, level).item()
        return self.q_alpha

    def get_coverage(self, test_scores):
        """
        Evaluate empirical coverage on a test set.

        Args:
            test_scores: [M] geodesic distances

        Returns:
            coverage: fraction of test points within q_alpha
        """
        if self.q_alpha is None:
            raise RuntimeError("Must calibrate before evaluating coverage")
        return (test_scores <= self.q_alpha).float().mean().item()
