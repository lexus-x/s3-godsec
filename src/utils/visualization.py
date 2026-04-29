"""
Visualization utilities for SE(3)-VLA experiments.

Generates figures for the paper.
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def plot_rotation_vs_delta_sr(
    rotation_magnitudes,
    delta_success_rates,
    task_names=None,
    save_path='figures/rotation_vs_delta_sr.png',
):
    """
    KEY FIGURE: Scatter plot of rotation magnitude vs Δ(success rate).
    
    Validates Theorem 1: SE(3) improvement is correlated with rotation magnitude.
    
    Args:
        rotation_magnitudes: [N] rotation angles in radians
        delta_success_rates: [N] Δ(success rate) = SR_SE3 - SR_Euclidean
        task_names: [N] optional task labels
        save_path: path to save figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Convert to degrees
    angles_deg = np.degrees(rotation_magnitudes)
    
    # Scatter plot
    ax.scatter(angles_deg, delta_success_rates * 100, s=80, alpha=0.7, c='steelblue', edgecolors='navy')
    
    # Annotate tasks if provided
    if task_names is not None:
        for i, name in enumerate(task_names):
            ax.annotate(name, (angles_deg[i], delta_success_rates[i] * 100),
                       fontsize=7, alpha=0.7, ha='left', va='bottom')
    
    # Fit linear regression
    z = np.polyfit(angles_deg, delta_success_rates * 100, 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, 180, 100)
    ax.plot(x_line, p(x_line), 'r--', alpha=0.5, label=f'Linear fit (slope={z[0]:.2f}%/°)')
    
    # Correlation coefficient
    corr = np.corrcoef(angles_deg, delta_success_rates * 100)[0, 1]
    ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('Average Rotation Magnitude (degrees)', fontsize=12)
    ax.set_ylabel('Δ Success Rate (SE(3) - Euclidean) [%]', fontsize=12)
    ax.set_title('Theorem 1 Validation: SE(3) Improvement vs Rotation Magnitude', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 190)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Figure saved to {save_path}")


def plot_success_rate_comparison(
    tasks,
    sr_euclidean,
    sr_se3,
    save_path='figures/sr_comparison.png',
):
    """
    Bar chart comparing Euclidean vs SE(3) success rates per task.
    
    Args:
        tasks: list of task names
        sr_euclidean: [N] success rates for Euclidean baseline
        sr_se3: [N] success rates for SE(3) head
        save_path: path to save figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    
    x = np.arange(len(tasks))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, sr_euclidean * 100, width, label='Euclidean', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, sr_se3 * 100, width, label='SE(3)', color='coral', alpha=0.8)
    
    ax.set_xlabel('Task', fontsize=12)
    ax.set_ylabel('Success Rate [%]', fontsize=12)
    ax.set_title('Euclidean vs SE(3) Action Head: Per-Task Success Rate', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Figure saved to {save_path}")


def plot_ablation_flow_steps(
    n_steps_list,
    success_rates,
    save_path='figures/ablation_flow_steps.png',
):
    """
    Line plot of flow integration steps vs success rate.
    
    Args:
        n_steps_list: list of step counts
        success_rates: corresponding success rates
        save_path: path to save figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    
    ax.plot(n_steps_list, [sr * 100 for sr in success_rates], 'o-', color='steelblue', linewidth=2, markersize=8)
    
    ax.set_xlabel('Number of Flow Integration Steps', fontsize=12)
    ax.set_ylabel('Success Rate [%]', fontsize=12)
    ax.set_title('Ablation: Flow Steps vs Performance', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Figure saved to {save_path}")


def plot_geodesic_ece_comparison(
    bin_centers_euclidean,
    bin_accuracies_euclidean,
    bin_centers_se3,
    bin_accuracies_se3,
    save_path='figures/geodesic_ece.png',
):
    """
    Calibration plot comparing Euclidean and SE(3) Action-ECE.
    
    Args:
        bin_centers_euclidean: bin centers for Euclidean
        bin_accuracies_euclidean: actual accuracy per bin for Euclidean
        bin_centers_se3: bin centers for SE(3)
        bin_accuracies_se3: actual accuracy per bin for SE(3)
        save_path: path to save figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect calibration')
    
    ax.plot(bin_centers_euclidean, bin_accuracies_euclidean, 'o-', color='steelblue', label='Euclidean', linewidth=2)
    ax.plot(bin_centers_se3, bin_accuracies_se3, 'o-', color='coral', label='SE(3)', linewidth=2)
    
    ax.set_xlabel('Predicted Confidence', fontsize=12)
    ax.set_ylabel('Actual Accuracy', fontsize=12)
    ax.set_title('Geodesic Action-ECE: Calibration Comparison', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Figure saved to {save_path}")
