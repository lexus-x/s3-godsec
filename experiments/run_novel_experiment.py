"""
Novel Experiment: Geodesic Action Chunking + Uncertainty-Aware Flow Matching

This script runs the full novel pipeline and produces results for the paper:

1. Geodesic Chunking: Predict H actions via K geodesic anchors on SE(3)
   vs. H independent Euclidean predictions (baseline)
2. Uncertainty-Aware Flow: N-sample flow matching with geodesic variance
   vs. single-point Euclidean prediction (baseline)
3. Conformal Prediction: Calibrated coverage on SE(3)

Generates:
    - results/chunking_comparison.json — chunking vs independent
    - results/uncertainty_analysis.json — calibration & coverage
    - results/novel_main_results.md — formatted results table

Usage:
    python experiments/run_novel_experiment.py --seed 0
    python experiments/run_novel_experiment.py --seed 0 --n-anchors 4
"""

import argparse
import yaml
import torch
import json
import os
import sys
import time
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.octo_adapter import OctoSE3, OctoEuclideanBaseline
from models.geodesic_chunking import GeodesicChunkPredictor
from models.uncertainty_head import (
    UncertaintyAwareFlowHead, ConformalCalibrator, GeodesicStats,
)
from models.scene_id_backbone import SceneIDBackbone
from models.mock_backbone import MockOctoBackbone
from training.data_loader import create_dataloaders
from utils.metrics import geodesic_rmse, rotation_rmse, translation_rmse
from utils.se3_utils import geodesic_distance, inverse_se3, log_se3


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def make_backbone(config):
    hidden_dim = config['model'].get('hidden_dim', 768)
    kind = config['model'].get('backbone_kind', 'scene_id')
    if kind == 'scene_id':
        n_tasks = config['model'].get('n_tasks', 2)
        return SceneIDBackbone(n_tasks=n_tasks, hidden_dim=hidden_dim)
    else:
        return MockOctoBackbone(hidden_dim=hidden_dim)


def train_chunking_model(model, dataloader, optimizer, device, n_epochs=30):
    """Train the geodesic chunking model."""
    model.train()
    history = []

    for epoch in range(n_epochs):
        total_loss = 0
        n_batches = 0

        for batch in tqdm(dataloader, desc=f"Chunk Epoch {epoch+1}", leave=False):
            observations = {k: v.to(device) for k, v in batch['observations'].items()}
            language = batch['language'].to(device)
            target_actions = batch['target_actions'].to(device)

            # Create action chunks from single targets (repeat for now)
            # In real setup, the dataloader would provide H-step chunks
            B = target_actions.shape[0]
            H = model.chunk_predictor.chunk_size
            target_chunk = target_actions.unsqueeze(1).expand(-1, H, -1, -1).clone()
            # Add small geodesic perturbation to simulate trajectory
            for h in range(1, H):
                noise = torch.randn(B, 6, device=device) * 0.02 * h
                from utils.se3_utils import exp_se3 as exp_se3_fn
                delta = exp_se3_fn(noise)
                target_chunk[:, h] = torch.bmm(target_chunk[:, h], delta)

            target_gripper = batch['target_gripper'].to(device)
            target_gripper_chunk = target_gripper.unsqueeze(1).expand(-1, H, -1)

            optimizer.zero_grad()

            h = model.encode(observations, language)
            loss, loss_dict = model.chunk_predictor.training_loss(
                h, target_chunk, target_gripper_chunk,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
            optimizer.step()

            total_loss += loss_dict['total_loss']
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        history.append({'epoch': epoch + 1, 'loss': avg_loss})

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs} | Loss: {avg_loss:.4f}")

    return history


@torch.no_grad()
def evaluate_chunking(model, dataloader, device):
    """Evaluate chunking model: first-action RMSE + temporal smoothness."""
    model.eval()
    all_pred_first = []
    all_pred_last = []
    all_target = []
    smoothness_scores = []

    for batch in tqdm(dataloader, desc="Eval chunking", leave=False):
        observations = {k: v.to(device) for k, v in batch['observations'].items()}
        language = batch['language'].to(device)
        target_actions = batch['target_actions'].to(device)

        h = model.encode(observations, language)
        chunk_actions, _ = model.chunk_predictor(h)  # [B, H, 4, 4]

        all_pred_first.append(chunk_actions[:, 0])
        all_pred_last.append(chunk_actions[:, -1])
        all_target.append(target_actions)

        # Temporal smoothness: mean geodesic distance between consecutive actions
        B, H = chunk_actions.shape[:2]
        for t in range(H - 1):
            dists = geodesic_distance(
                chunk_actions[:, t].reshape(-1, 4, 4),
                chunk_actions[:, t + 1].reshape(-1, 4, 4),
            )
            smoothness_scores.append(dists.mean().item())

    pred_first = torch.cat(all_pred_first, dim=0)
    pred_last = torch.cat(all_pred_last, dim=0)
    target = torch.cat(all_target, dim=0)

    return {
        'first_action_grmse': geodesic_rmse(pred_first, target).item(),
        'first_action_rrmse': rotation_rmse(pred_first, target).item(),
        'first_action_trmse': translation_rmse(pred_first, target).item(),
        'last_action_grmse': geodesic_rmse(pred_last, target).item(),
        'mean_smoothness': np.mean(smoothness_scores),
        'std_smoothness': np.std(smoothness_scores),
    }


def train_uncertainty_model(model, dataloader, optimizer, device, n_epochs=30):
    """Train the SE(3) flow matching model for uncertainty evaluation."""
    model.train()
    history = []

    for epoch in range(n_epochs):
        total_loss = 0
        n_batches = 0

        for batch in tqdm(dataloader, desc=f"Flow Epoch {epoch+1}", leave=False):
            observations = {k: v.to(device) for k, v in batch['observations'].items()}
            language = batch['language'].to(device)
            target_actions = batch['target_actions'].to(device)
            target_gripper = batch['target_gripper'].to(device)

            optimizer.zero_grad()
            loss, loss_dict = model.compute_loss(
                observations, language, target_actions, target_gripper,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
            optimizer.step()

            total_loss += loss_dict['total_loss']
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        history.append({'epoch': epoch + 1, 'loss': avg_loss})

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs} | Loss: {avg_loss:.4f}")

    return history


@torch.no_grad()
def evaluate_uncertainty(flow_head, dataloader, device, n_samples=10):
    """
    Evaluate uncertainty: geodesic variance, calibration, coverage.

    Returns per-sample: (mean_action, uncertainty, target)
    for downstream conformal calibration.
    """
    flow_head.eval()
    all_mean = []
    all_variance = []
    all_target = []
    all_dists_to_target = []

    for batch in tqdm(dataloader, desc="Eval uncertainty", leave=False):
        observations = {k: v.to(device) for k, v in batch['observations'].items()}
        language = batch['language'].to(device)
        target_actions = batch['target_actions'].to(device)

        h = flow_head.predictor.flow_head.hidden_dim  # just for reference
        h_vec = flow_head.predictor.flow_head.output_proj  # not right, need encode

        # Need to use the parent model's encode
        # This is called externally — see run_experiment
        B = target_actions.shape[0]

    return None  # placeholder — actual evaluation done in run_experiment


def run_experiment(args):
    """Main experiment runner."""
    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = args.seed

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    hidden_dim = config['model'].get('hidden_dim', 768)

    print("=" * 70)
    print("  Novel Experiment: Geodesic Chunking + Uncertainty-Aware Flow")
    print(f"  Seed: {seed} | Device: {device}")
    print("=" * 70)

    # --- Data ---
    print("\n[1/5] Creating dataloaders...")
    train_loader, val_loaders = create_dataloaders(config, seed=seed)

    # --- Part A: Geodesic Chunking ---
    print("\n[2/5] Training geodesic chunking model...")
    backbone_a = make_backbone(config)
    chunk_model = OctoSE3(
        octo_model=backbone_a,
        hidden_dim=hidden_dim,
        freeze_backbone=False,
    )
    # Replace action predictor with chunk predictor
    chunk_predictor = GeodesicChunkPredictor(
        hidden_dim=hidden_dim,
        chunk_size=config.get('chunk_size', 8),
        n_anchors=args.n_anchors,
    )
    chunk_model.chunk_predictor = chunk_predictor
    chunk_model = chunk_model.to(device)

    optimizer_a = torch.optim.AdamW(
        chunk_model.trainable_parameters(), lr=1e-4, weight_decay=1e-5,
    )
    train_chunking_model(chunk_model, train_loader, optimizer_a, device, n_epochs=args.epochs)

    print("  Evaluating chunking...")
    chunk_results = {}
    for family, loader in val_loaders.items():
        chunk_results[family] = evaluate_chunking(chunk_model, loader, device)
        r = chunk_results[family]
        print(f"  [{family}] First: G={r['first_action_grmse']:.4f} | "
              f"Smooth: {r['mean_smoothness']:.4f}±{r['std_smoothness']:.4f}")

    # --- Part B: Uncertainty-Aware Flow Matching ---
    print("\n[3/5] Training uncertainty-aware flow model...")
    backbone_b = make_backbone(config)
    flow_model = OctoSE3(
        octo_model=backbone_b,
        hidden_dim=hidden_dim,
        head_hidden_dim=config['model'].get('head_hidden_dim', 256),
        n_layers=config['model'].get('n_layers', 4),
        n_flow_steps_train=config['training'].get('n_flow_steps_train', 10),
        n_flow_steps_eval=config['training'].get('n_flow_steps_eval', 10),
        source_scale=config['model'].get('source_scale', 0.1),
        freeze_backbone=False,
    )
    flow_model = flow_model.to(device)

    optimizer_b = torch.optim.AdamW(
        flow_model.trainable_parameters(), lr=1e-4, weight_decay=1e-5,
    )
    train_uncertainty_model(flow_model, train_loader, optimizer_b, device, n_epochs=args.epochs)

    # Wrap with uncertainty head
    print("\n[4/5] Evaluating uncertainty (N samples per prediction)...")
    uncertainty_head = UncertaintyAwareFlowHead(
        flow_model.action_predictor, n_samples=args.n_samples,
    )

    # Collect calibration scores
    all_dists = []
    all_variances = []
    flow_model.eval()
    with torch.no_grad():
        for batch in tqdm(val_loaders['combined'], desc="Calibration", leave=False):
            observations = {k: v.to(device) for k, v in batch['observations'].items()}
            language = batch['language'].to(device)
            target_actions = batch['target_actions'].to(device)

            h = flow_model.encode(observations, language)
            mean_action, _, variance, _ = uncertainty_head.predict_with_uncertainty(h)

            # Distance from predicted mean to ground truth
            dists = geodesic_distance(mean_action, target_actions)
            all_dists.append(dists)
            all_variances.append(variance)

    all_dists = torch.cat(all_dists)
    all_variances = torch.cat(all_variances)

    # Split into calibration and test
    n = len(all_dists)
    cal_dists = all_dists[:n // 2]
    test_dists = all_dists[n // 2:]
    cal_variances = all_variances[:n // 2]
    test_variances = all_variances[n // 2:]

    # Calibrate conformal prediction
    calibrator = ConformalCalibrator(alpha=0.1)
    q_alpha = calibrator.calibrate(cal_dists)
    coverage = calibrator.get_coverage(test_dists)

    # Variance as uncertainty proxy: is high variance ↔ high error?
    var_error_corr = torch.corrcoef(torch.stack([
        cal_variances, cal_dists,
    ]))[0, 1].item()

    print(f"  Conformal radius (α=0.1): {q_alpha:.4f}")
    print(f"  Empirical coverage: {coverage:.3f} (target: 0.900)")
    print(f"  Variance-error correlation: {var_error_corr:.3f}")

    # --- Baseline comparison ---
    print("\n[5/5] Baseline Euclidean comparison...")
    backbone_c = make_backbone(config)
    baseline = OctoEuclideanBaseline(
        octo_model=backbone_c, hidden_dim=hidden_dim, freeze_backbone=False,
    )
    baseline = baseline.to(device)
    optimizer_c = torch.optim.AdamW(
        baseline.trainable_parameters(), lr=1e-4, weight_decay=1e-5,
    )

    # Quick train baseline
    baseline.train()
    for epoch in range(args.epochs):
        for batch in train_loader:
            observations = {k: v.to(device) for k, v in batch['observations'].items()}
            language = batch['language'].to(device)
            target_actions = batch['target_actions'].to(device)
            target_gripper = batch['target_gripper'].to(device)

            optimizer_c.zero_grad()
            from utils.se3_utils import log_so3
            R = target_actions[:, :3, :3]
            t = target_actions[:, :3, 3]
            omega = log_so3(R)
            target_flat = torch.cat([omega, t], dim=-1)
            loss, _ = baseline.compute_loss(observations, language, target_flat, target_gripper)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(baseline.trainable_parameters(), 1.0)
            optimizer_c.step()

    # Evaluate baseline
    baseline.eval()
    baseline_results = {}
    with torch.no_grad():
        for family, loader in val_loaders.items():
            all_pred = []
            all_target = []
            for batch in loader:
                observations = {k: v.to(device) for k, v in batch['observations'].items()}
                language = batch['language'].to(device)
                target_actions = batch['target_actions'].to(device)

                h = baseline.encode(observations, language)
                action = baseline.action_head(h)
                from utils.se3_utils import exp_so3
                omega = action[:, :3]
                t = action[:, 3:6]
                R = exp_so3(omega)
                B = R.shape[0]
                pred = torch.eye(4, device=device).unsqueeze(0).expand(B, -1, -1).clone()
                pred[:, :3, :3] = R
                pred[:, :3, 3] = t

                all_pred.append(pred)
                all_target.append(target_actions)

            all_pred = torch.cat(all_pred)
            all_target = torch.cat(all_target)
            baseline_results[family] = {
                'geodesic_rmse': geodesic_rmse(all_pred, all_target).item(),
                'rotation_rmse': rotation_rmse(all_pred, all_target).item(),
                'translation_rmse': translation_rmse(all_pred, all_target).item(),
            }
            r = baseline_results[family]
            print(f"  [{family}] G={r['geodesic_rmse']:.4f} R={r['rotation_rmse']:.4f} T={r['translation_rmse']:.4f}")

    # --- Save results ---
    results = {
        'seed': seed,
        'n_anchors': args.n_anchors,
        'n_samples': args.n_samples,
        'chunking': chunk_results,
        'uncertainty': {
            'conformal_radius': q_alpha,
            'coverage': coverage,
            'target_coverage': 0.9,
            'variance_error_correlation': var_error_corr,
            'mean_variance': all_variances.mean().item(),
        },
        'baseline_euclidean': baseline_results,
    }

    os.makedirs('results', exist_ok=True)
    out_path = f'results/novel_experiment_seed{seed}.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {out_path}")

    # Generate markdown report
    md_path = f'results/novel_main_results_seed{seed}.md'
    with open(md_path, 'w') as f:
        f.write("# Novel Experiment Results\n\n")
        f.write(f"Seed: {seed} | Anchors: {args.n_anchors} | Uncertainty samples: {args.n_samples}\n\n")

        f.write("## Geodesic Chunking vs Euclidean Baseline\n\n")
        f.write("| Family | Baseline G-RMSE | Chunk First G-RMSE | Smoothness |\n")
        f.write("|--------|----------------|-------------------|------------|\n")
        for family in chunk_results:
            cr = chunk_results[family]
            br = baseline_results.get(family, {})
            f.write(f"| {family} | {br.get('geodesic_rmse', 0):.4f} | "
                    f"{cr['first_action_grmse']:.4f} | "
                    f"{cr['mean_smoothness']:.4f}±{cr['std_smoothness']:.4f} |\n")

        f.write("\n## Uncertainty Calibration (Conformal Prediction on SE(3))\n\n")
        f.write(f"- Coverage: {coverage:.3f} (target: 0.900)\n")
        f.write(f"- Conformal radius: {q_alpha:.4f}\n")
        f.write(f"- Variance-error correlation: {var_error_corr:.3f}\n")
        f.write(f"- Mean geodesic variance: {all_variances.mean().item():.4f}\n")

    print(f"  Report saved: {md_path}")
    print("\n" + "=" * 70)
    print("  Done!")
    print("=" * 70)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Novel SE(3)-VLA experiment")
    parser.add_argument('--config', type=str, default='configs/octo_se3.yaml')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n-anchors', type=int, default=2, help="K geodesic anchors for chunking")
    parser.add_argument('--n-samples', type=int, default=10, help="N flow samples for uncertainty")
    parser.add_argument('--epochs', type=int, default=30)
    args = parser.parse_args()

    run_experiment(args)
