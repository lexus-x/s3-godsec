# TECHNICAL_DETAILS.md — Mathematical Formulation

## 1. Preliminaries

### 1.1 The Special Euclidean Group SE(3)

SE(3) is the 6-dimensional Lie group of rigid body transformations in 3D space:

```
SE(3) = {(R, t) : R ∈ SO(3), t ∈ ℝ³}
```

with group operation:
```
(R₁, t₁) · (R₂, t₂) = (R₁R₂, R₁t₂ + t₁)
```

As a matrix group:
```
T = [R | t] ∈ ℝ⁴ˣ⁴
    [0 | 1]
```

### 1.2 The Lie Algebra se(3)

The Lie algebra se(3) is the tangent space at the identity:

```
se(3) = {(ω, v) : ω ∈ so(3), v ∈ ℝ³} ≅ ℝ⁶
```

where so(3) is the space of 3×3 skew-symmetric matrices.

As a matrix:
```
[ω]ₓ v] ∈ ℝ⁴ˣ⁴
[  0  0]
```

where [ω]ₓ is the skew-symmetric matrix of ω.

### 1.3 Exponential and Logarithmic Maps

**Exponential map** exp: se(3) → SE(3):
```
exp(ω, v) = (exp_so(3)(ω), V·v)
```

where:
```
exp_so(3)(ω) = I + (sinθ/θ)[ω]ₓ + ((1-cosθ)/θ²)[ω]ₓ²
V = I + ((1-cosθ)/θ²)[ω]ₓ + ((θ-sinθ)/θ³)[ω]ₓ²
θ = ||ω||
```

**Logarithmic map** log: SE(3) → se(3):
```
log(R, t) = (log_so(3)(R), V⁻¹·t)
```

where:
```
log_so(3)(R) = (θ/(2sinθ)) · (R - Rᵀ)   for θ ≠ 0
θ = arccos((tr(R) - 1) / 2)
V⁻¹ = I - ½[ω]ₓ + (1/θ² - (1+cosθ)/(2θ·sinθ))·[ω]ₓ²
```

### 1.4 Bi-Invariant Metric on SE(3)

SE(3) does **not** have a bi-invariant metric (unlike compact Lie groups). However, it has a **left-invariant metric**:

```
⟨(ω₁, v₁), (ω₂, v₂)⟩_e = α·ω₁ᵀω₂ + β·v₁ᵀv₂
```

where α, β > 0 are scaling factors. We use α = 1, β = 1 (equal weighting of rotation and translation).

The geodesic distance:
```
d_SE(3)(T₁, T₂) = ||log(T₁⁻¹ · T₂)||_G
```

where ||·||_G is the norm induced by the left-invariant metric.

## 2. Riemannian Flow Matching on SE(3)

### 2.1 Problem Formulation

Given a VLA model that produces a hidden state h ∈ ℝ^d from visual observations and language instructions, we want to learn a conditional flow on SE(3) that maps a source distribution p₀ (e.g., Gaussian on the tangent space) to the target action distribution p₁ (the demonstration distribution on SE(3)).

### 2.2 Flow Matching Loss

The flow matching objective on SE(3):

```
L_FM = E_{t, X₀, X₁} ||v_θ(X_t, t, h) - log_{X_t}(X₁)||²_{X_t}
```

where:
- t ~ Uniform(0, 1)
- X₀ ~ p₀ (source distribution)
- X₁ ~ p₁ (demonstration distribution)
- X_t = X₀ · exp(t · log(X₀⁻¹ · X₁)) is the geodesic interpolation
- v_θ is the learned velocity field
- ||·||_{X_t} is the Riemannian metric at X_t

### 2.3 Practical Implementation

In practice, we parameterize the velocity in the tangent space at the identity (the Lie algebra):

```
v_θ(X_t, t, h) ≈ (ω_θ(h, t), v_θ(h, t)) ∈ se(3)
```

The network predicts a 6-dimensional vector (3 for rotation, 3 for translation) conditioned on the VLA hidden state and time.

### 2.4 Geodesic Interpolation

The geodesic from X₀ to X₁ at time t:

```
X_t = X₀ · exp(t · ξ)
where ξ = log(X₀⁻¹ · X₁) ∈ se(3)
```

This is computed using the exponential map formula from Section 1.3.

## 3. The Euclidean Approximation Error Bound

### Theorem 1 (Bounded Error of Euclidean Approximation)

Let f^E be the optimal Euclidean flow matching policy on R⁶ axis-angle and f^* be the optimal SE(3) flow matching policy. For any rotation R with ||log(R)|| = θ:

```
E||f^E(R) - f^*(R)|| ≤ C · θ² + O(θ⁴)
```

**Proof sketch:**

The key insight is that the Euclidean approximation error comes from two sources:

1. **Metric distortion**: The Euclidean metric on R⁶ differs from the Riemannian metric on SE(3) by a factor that depends on the curvature. For small rotations (θ → 0), the metrics agree to first order, so the error is O(θ²).

2. **Topological obstruction**: At θ = π, the axis-angle parameterization has a singularity (the antipodal point). The Euclidean flow cannot represent the geodesic through this point, leading to a discontinuity.

The bound is tight: there exist data distributions where the Euclidean flow achieves the rate exactly.

**Implication for VLAs**: On tasks where the average rotation magnitude is θ > π/2 (rotation-heavy MetaWorld tasks), the Euclidean approximation error is non-negligible. The SE(3) flow matching head eliminates this error.

## 4. Architecture

### 4.1 Overall Architecture

```
[Pretrained VLA Backbone (frozen)]
         ↓
    [Hidden State h ∈ ℝ^d]
         ↓
[SE(3) Flow Matching Head]
         ↓
    [Action T ∈ SE(3)]
```

### 4.2 Flow Matching Head Architecture

```python
class SE3FlowHead(nn.Module):
    """
    Riemannian flow matching head on SE(3).
    
    Input: VLA hidden state h ∈ ℝ^d, current pose X_t ∈ SE(3), time t ∈ [0,1]
    Output: Velocity v ∈ se(3) (6-dimensional)
    """
    def __init__(self, hidden_dim, hidden_dim_head=256, n_layers=4):
        super().__init__()
        # Input: hidden state + SE(3) log coordinates (6D) + time (1D)
        input_dim = hidden_dim + 6 + 1
        
        layers = []
        for i in range(n_layers):
            layers.extend([
                nn.Linear(input_dim if i == 0 else hidden_dim_head, hidden_dim_head),
                nn.GELU(),
                nn.LayerNorm(hidden_dim_head),
            ])
        self.net = nn.Sequential(*layers)
        
        # Output: velocity in se(3) (6D: 3 rotation + 3 translation)
        self.output_proj = nn.Linear(hidden_dim_head, 6)
        
        # Initialize output to zero for stable training
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
    
    def forward(self, h, X_t, t):
        """
        h: [B, hidden_dim] - VLA hidden state
        X_t: [B, 4, 4] - current pose on SE(3)
        t: [B, 1] - time in [0, 1]
        """
        # Convert SE(3) matrix to Lie algebra coordinates
        xi_t = se3_log(X_t)  # [B, 6]
        
        # Concatenate inputs
        x = torch.cat([h, xi_t, t], dim=-1)  # [B, hidden_dim + 7]
        
        # Predict velocity
        v = self.output_proj(self.net(x))  # [B, 6]
        
        return v  # velocity in se(3)
```

### 4.3 Integration with Octo

```python
class OctoSE3(nn.Module):
    """
    Octo VLA with SE(3) flow matching action head.
    
    Replaces Octo's flat R^7 action head with SE(3) flow matching.
    """
    def __init__(self, octo_checkpoint, n_flow_steps=20):
        super().__init__()
        # Load pretrained Octo (frozen)
        self.backbone = load_octo_backbone(octo_checkpoint)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False
        
        # SE(3) flow matching head (trainable)
        self.flow_head = SE3FlowHead(
            hidden_dim=self.backbone.hidden_dim,
            hidden_dim_head=256,
            n_layers=4
        )
        
        # Gripper head (separate, Euclidean)
        self.gripper_head = nn.Linear(self.backbone.hidden_dim, 1)
        
        self.n_flow_steps = n_flow_steps
    
    def forward(self, observations, language_instruction):
        """
        Returns action in SE(3) + gripper.
        """
        # Get hidden state from frozen backbone
        h = self.backbone.encode(observations, language_instruction)
        
        # Sample initial pose from source distribution
        X_0 = sample_se3_gaussian(h.shape[0])  # [B, 4, 4]
        
        # Flow matching: integrate from X_0 to X_1
        dt = 1.0 / self.n_flow_steps
        X_t = X_0
        for step in range(self.n_flow_steps):
            t = torch.full((h.shape[0], 1), step * dt, device=h.device)
            v = self.flow_head(h, X_t, t)
            
            # Update: X_{t+dt} = X_t · exp(dt * v)
            X_t = X_t @ se3_exp(dt * v)
        
        # Gripper action (Euclidean)
        gripper = torch.sigmoid(self.gripper_head(h))
        
        return X_t, gripper  # SE(3) pose + gripper
```

## 5. Loss Function

### 5.1 Geodesic Flow Matching Loss

```python
def geodesic_flow_loss(flow_head, h, X_0, X_1, n_steps=20):
    """
    Compute the Riemannian flow matching loss on SE(3).
    
    h: [B, hidden_dim] - VLA hidden state
    X_0: [B, 4, 4] - source samples
    X_1: [B, 4, 4] - target actions (from demonstrations)
    """
    # Geodesic interpolation
    xi = se3_log(torch.bmm(inverse_se3(X_0), X_1))  # [B, 6]
    
    # Sample random time
    t = torch.rand(h.shape[0], 1, device=h.device)
    
    # Interpolate on SE(3)
    X_t = X_0 @ se3_exp(t * xi)  # [B, 4, 4]
    
    # Target velocity (tangent vector at X_t)
    # The geodesic velocity is constant in the body frame
    v_target = xi  # [B, 6]
    
    # Predicted velocity
    v_pred = flow_head(h, X_t, t)
    
    # Loss: MSE in the Lie algebra
    loss = F.mse_loss(v_pred, v_target)
    
    return loss
```

### 5.2 Geodesic Distance Metric

```python
def geodesic_distance(T1, T2):
    """
    Compute geodesic distance between two SE(3) poses.
    
    Returns: [B] tensor of distances
    """
    T_rel = torch.bmm(inverse_se3(T1), T2)
    xi = se3_log(T_rel)  # [B, 6]
    
    # Split into rotation and translation
    omega = xi[:, :3]  # rotation component
    v = xi[:, 3:]      # translation component
    
    # Geodesic distance (with equal weighting)
    theta = torch.norm(omega, dim=-1)  # rotation angle
    t_norm = torch.norm(v, dim=-1)     # translation distance
    
    return torch.sqrt(theta**2 + t_norm**2)
```

## 6. Inference

### 6.1 Single-Step Prediction (Fast)

For inference, we can use a single Euler step:

```python
def predict_action_single_step(model, h, X_0):
    """Fast inference: single Euler step."""
    t = torch.zeros(h.shape[0], 1, device=h.device)
    v = model.flow_head(h, X_0, t)
    X_1 = X_0 @ se3_exp(v)
    return X_1
```

### 6.2 Multi-Step Prediction (Accurate)

For higher accuracy, use multiple steps:

```python
def predict_action_multi_step(model, h, X_0, n_steps=10):
    """Accurate inference: multiple Euler steps."""
    dt = 1.0 / n_steps
    X_t = X_0
    for step in range(n_steps):
        t = torch.full((h.shape[0], 1), step * dt, device=h.device)
        v = model.flow_head(h, X_t, t)
        X_t = X_t @ se3_exp(dt * v)
    return X_t
```

## 7. Key Implementation Notes

### 7.1 Numerical Stability

The exponential and logarithmic maps have singularities at θ = 0 and θ = π. Use the following safe implementations:

```python
def safe_exp_so3(omega):
    """Numerically stable exponential map for SO(3)."""
    theta = torch.norm(omega, dim=-1, keepdim=True)  # [B, 1]
    
    # Taylor expansion for small theta
    small = theta < 1e-6
    
    # Compute using Rodrigues formula
    K = skew_symmetric(omega)  # [B, 3, 3]
    
    # Coefficients
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    
    # Safe division
    a = torch.where(small, 1 - theta**2/6, sin_theta / theta)
    b = torch.where(small, 0.5 - theta**2/24, (1 - cos_theta) / theta**2)
    
    # Rodrigues formula: I + a*K + b*K²
    I = torch.eye(3, device=omega.device).unsqueeze(0)
    R = I + a.unsqueeze(-1) * K + b.unsqueeze(-1) * torch.bmm(K, K)
    
    return R
```

### 7.2 Action Chunking

For action chunking (H=8 future steps), predict H poses on SE(3):

```python
def predict_action_chunk(model, h, X_0, chunk_size=8):
    """Predict a chunk of H future actions on SE(3)."""
    actions = []
    X_t = X_0
    
    for i in range(chunk_size):
        X_next = predict_action_single_step(model, h, X_t)
        actions.append(X_next)
        X_t = X_next  # autoregressive
    
    return torch.stack(actions, dim=1)  # [B, H, 4, 4]
```

### 7.3 Gripper Action

The gripper is a binary (open/close) or continuous (0-1) action that is NOT on SE(3). We predict it separately using a standard Euclidean head:

```python
gripper = torch.sigmoid(self.gripper_head(h))  # [B, 1]
```

The final action is the SE(3) pose concatenated with the gripper value.
