"""
SE(3) utility operations for Riemannian flow matching.

Provides numerically stable implementations of:
- Exponential map: se(3) → SE(3)
- Logarithmic map: SE(3) → se(3)
- Geodesic distance on SE(3)
- Geodesic interpolation on SE(3)
- Inverse of SE(3) matrices
"""

import torch
import torch.nn.functional as F


def hat(omega):
    """
    Hat map: R³ → so(3) (skew-symmetric matrix).
    
    Args:
        omega: [B, 3] rotation vectors
    
    Returns:
        K: [B, 3, 3] skew-symmetric matrices
    """
    B = omega.shape[0]
    K = torch.zeros(B, 3, 3, device=omega.device, dtype=omega.dtype)
    K[:, 0, 1] = -omega[:, 2]
    K[:, 0, 2] = omega[:, 1]
    K[:, 1, 0] = omega[:, 2]
    K[:, 1, 2] = -omega[:, 0]
    K[:, 2, 0] = -omega[:, 1]
    K[:, 2, 1] = omega[:, 0]
    return K


def exp_so3(omega):
    """
    Numerically stable exponential map for SO(3).
    
    Rodrigues formula with Taylor expansion for small theta.
    
    Args:
        omega: [B, 3] rotation vectors (axis * angle)
    
    Returns:
        R: [B, 3, 3] rotation matrices
    """
    theta = torch.norm(omega, dim=-1, keepdim=True)  # [B, 1]
    theta_sq = theta ** 2
    
    # Skew-symmetric matrices
    K = hat(omega)  # [B, 3, 3]
    K_sq = torch.bmm(K, K)  # [B, 3, 3]
    
    # Coefficients with Taylor expansion for small theta
    # sin(theta) / theta
    a = torch.where(
        theta < 1e-6,
        1.0 - theta_sq / 6.0,  # Taylor: 1 - θ²/6
        torch.sin(theta) / theta
    )
    
    # (1 - cos(theta)) / theta²
    b = torch.where(
        theta < 1e-6,
        0.5 - theta_sq / 24.0,  # Taylor: 1/2 - θ²/24
        (1.0 - torch.cos(theta)) / theta_sq
    )
    
    # Rodrigues formula: I + a*K + b*K²
    I = torch.eye(3, device=omega.device, dtype=omega.dtype).unsqueeze(0)
    R = I + a.unsqueeze(-1) * K + b.unsqueeze(-1) * K_sq
    
    return R


def log_so3(R):
    """
    Logarithmic map for SO(3).
    
    Args:
        R: [B, 3, 3] rotation matrices
    
    Returns:
        omega: [B, 3] rotation vectors
    """
    # Rotation angle
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    # Clamp to avoid NaN in arccos
    cos_theta = ((trace - 1) / 2).clamp(-1 + 1e-7, 1 - 1e-7)
    theta = torch.acos(cos_theta)  # [B]
    
    # Axis from skew-symmetric part
    # (R - R^T) / 2 = sin(theta) * [n_hat]_x
    skew = (R - R.transpose(-1, -2)) / 2  # [B, 3, 3]
    omega = torch.stack([
        skew[:, 2, 1],  # wx
        skew[:, 0, 2],  # wy
        skew[:, 1, 0],  # wz
    ], dim=-1)  # [B, 3]
    
    # omega = theta / sin(theta) * omega_extracted
    sin_theta = torch.sin(theta)
    
    # Handle small theta (Taylor expansion)
    # theta / sin(theta) ≈ 1 + theta²/6
    coeff = torch.where(
        theta.abs() < 1e-6,
        1.0 + theta.pow(2) / 6.0,
        theta / sin_theta
    )
    
    omega = coeff.unsqueeze(-1) * omega
    
    # Handle theta ≈ pi (special case)
    # When theta ≈ pi, sin(theta) ≈ 0, need different extraction
    near_pi = theta.abs() > (torch.pi - 1e-3)
    if near_pi.any():
        # For theta ≈ pi, use the column of (R + I) with largest norm
        R_plus_I = R[near_pi] + torch.eye(3, device=R.device, dtype=R.dtype)
        norms = torch.norm(R_plus_I, dim=-1)  # [B, 3]
        best_col = norms.argmax(dim=-1)  # [B]
        axis = R_plus_I[torch.arange(best_col.shape[0]), :, best_col]
        axis = F.normalize(axis, dim=-1)
        omega[near_pi] = theta[near_pi].unsqueeze(-1) * axis
    
    return omega


def exp_se3(xi):
    """
    Exponential map for SE(3).
    
    Args:
        xi: [B, 6] vectors in se(3) (first 3: rotation, last 3: translation)
    
    Returns:
        T: [B, 4, 4] SE(3) matrices
    """
    omega = xi[:, :3]  # [B, 3] rotation
    v = xi[:, 3:]      # [B, 3] translation
    
    R = exp_so3(omega)  # [B, 3, 3]
    
    theta = torch.norm(omega, dim=-1, keepdim=True)  # [B, 1]
    theta_sq = theta ** 2
    
    K = hat(omega)  # [B, 3, 3]
    K_sq = torch.bmm(K, K)  # [B, 3, 3]
    
    # V matrix: I + ((1-cos)/theta²)*K + ((theta-sin)/theta³)*K²
    # Taylor for small theta: V ≈ I + K/2 + K²/6
    a = torch.where(
        theta < 1e-6,
        0.5 - theta_sq / 24.0,  # 1/2 - θ²/24
        (1.0 - torch.cos(theta)) / theta_sq
    )
    b = torch.where(
        theta < 1e-6,
        1.0 / 6.0 - theta_sq / 120.0,  # 1/6 - θ²/120
        (theta - torch.sin(theta)) / (theta_sq * theta)
    )
    
    I = torch.eye(3, device=xi.device, dtype=xi.dtype).unsqueeze(0)
    V = I + a.unsqueeze(-1) * K + b.unsqueeze(-1) * K_sq
    
    t = torch.bmm(V, v.unsqueeze(-1)).squeeze(-1)  # [B, 3]
    
    # Assemble SE(3) matrix
    T = torch.zeros(xi.shape[0], 4, 4, device=xi.device, dtype=xi.dtype)
    T[:, :3, :3] = R
    T[:, :3, 3] = t
    T[:, 3, 3] = 1.0
    
    return T


def log_se3(T):
    """
    Logarithmic map for SE(3).
    
    Args:
        T: [B, 4, 4] SE(3) matrices
    
    Returns:
        xi: [B, 6] vectors in se(3) (first 3: rotation, last 3: translation)
    """
    R = T[:, :3, :3]  # [B, 3, 3]
    t = T[:, :3, 3]   # [B, 3]
    
    omega = log_so3(R)  # [B, 3]
    
    theta = torch.norm(omega, dim=-1, keepdim=True)  # [B, 1]
    theta_sq = theta ** 2
    
    K = hat(omega)  # [B, 3, 3]
    K_sq = torch.bmm(K, K)  # [B, 3, 3]
    
    # V^{-1} = I - K/2 + (1/theta² - (1+cos)/(2*theta*sin)) * K²
    # Taylor for small theta: V^{-1} ≈ I - K/2 + K²/12
    half_theta = theta / 2
    a = torch.where(
        theta < 1e-6,
        -0.5 + theta_sq / 12.0,  # -1/2 + θ²/12
        -0.5
    )
    b = torch.where(
        theta < 1e-6,
        1.0 / 12.0,  # 1/12
        (1.0 / theta_sq - (1.0 + torch.cos(theta)) / (2.0 * theta * torch.sin(theta)))
    )
    
    I = torch.eye(3, device=T.device, dtype=T.dtype).unsqueeze(0)
    V_inv = I + a.unsqueeze(-1) * K + b.unsqueeze(-1) * K_sq
    
    v = torch.bmm(V_inv, t.unsqueeze(-1)).squeeze(-1)  # [B, 3]
    
    xi = torch.cat([omega, v], dim=-1)  # [B, 6]
    
    return xi


def inverse_se3(T):
    """
    Inverse of SE(3) matrix.
    
    T^{-1} = [R^T | -R^T * t]
              [  0 |         1]
    
    Args:
        T: [B, 4, 4] SE(3) matrices
    
    Returns:
        T_inv: [B, 4, 4] inverse SE(3) matrices
    """
    R = T[:, :3, :3]
    t = T[:, :3, 3]
    
    R_inv = R.transpose(-1, -2)
    t_inv = -torch.bmm(R_inv, t.unsqueeze(-1)).squeeze(-1)
    
    T_inv = torch.zeros_like(T)
    T_inv[:, :3, :3] = R_inv
    T_inv[:, :3, 3] = t_inv
    T_inv[:, 3, 3] = 1.0
    
    return T_inv


def geodesic_distance(T1, T2):
    """
    Geodesic distance between two SE(3) poses.
    
    d(T1, T2) = ||log(T1^{-1} * T2)||
    
    Uses left-invariant metric with equal rotation/translation weighting.
    
    Args:
        T1: [B, 4, 4] SE(3) matrices
        T2: [B, 4, 4] SE(3) matrices
    
    Returns:
        dist: [B] geodesic distances
    """
    T_rel = torch.bmm(inverse_se3(T1), T2)
    xi = log_se3(T_rel)  # [B, 6]
    
    # Rotation component: angle theta
    omega = xi[:, :3]
    theta = torch.norm(omega, dim=-1)
    
    # Translation component
    v = xi[:, 3:]
    t_norm = torch.norm(v, dim=-1)
    
    # Combined distance (equal weighting)
    dist = torch.sqrt(theta ** 2 + t_norm ** 2)
    
    return dist


def geodesic_distance_rotation_only(T1, T2):
    """
    Geodesic distance on SO(3) only (ignoring translation).
    
    Args:
        T1: [B, 4, 4] SE(3) matrices
        T2: [B, 4, 4] SE(3) matrices
    
    Returns:
        dist: [B] rotation geodesic distances (in radians)
    """
    R1 = T1[:, :3, :3]
    R2 = T2[:, :3, :3]
    
    R_rel = torch.bmm(R1.transpose(-1, -2), R2)
    trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
    cos_theta = ((trace - 1) / 2).clamp(-1 + 1e-7, 1 - 1e-7)
    theta = torch.acos(cos_theta)
    
    return theta


def geodesic_interpolation(T0, T1, t):
    """
    Geodesic interpolation on SE(3).
    
    T(t) = T0 * exp(t * log(T0^{-1} * T1))
    
    Args:
        T0: [B, 4, 4] start poses
        T1: [B, 4, 4] end poses
        t: [B, 1] interpolation parameter in [0, 1]
    
    Returns:
        T_t: [B, 4, 4] interpolated poses
    """
    T_rel = torch.bmm(inverse_se3(T0), T1)
    xi = log_se3(T_rel)  # [B, 6]
    
    T_t = torch.bmm(T0, exp_se3(t * xi))
    
    return T_t


def sample_se3_gaussian(batch_size, device='cuda', dtype=torch.float32):
    """
    Sample from a Gaussian distribution on se(3), then map to SE(3).
    
    This is used as the source distribution for flow matching.
    
    Args:
        batch_size: number of samples
        device: torch device
        dtype: torch dtype
    
    Returns:
        T: [B, 4, 4] SE(3) matrices sampled from the source distribution
    """
    # Sample in se(3) (tangent space at identity)
    xi = torch.randn(batch_size, 6, device=device, dtype=dtype) * 0.1
    
    # Map to SE(3)
    T = exp_se3(xi)
    
    return T


def se3_to_rotation_angle(T):
    """
    Extract the rotation angle from an SE(3) matrix.
    
    Args:
        T: [B, 4, 4] SE(3) matrices
    
    Returns:
        theta: [B] rotation angles in radians
    """
    R = T[:, :3, :3]
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    cos_theta = ((trace - 1) / 2).clamp(-1 + 1e-7, 1 - 1e-7)
    theta = torch.acos(cos_theta)
    return theta
