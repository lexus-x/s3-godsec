"""
SmolVLA Evaluation Script for SE(3)-VLA.

Evaluates trained SmolVLA + SE(3) action head checkpoints.
Reports per-family metrics and generates evaluation reports.

Usage:
    # Evaluate flow head
    python src/evaluate_smolvla.py --config configs/smolvla_se3.yaml \
        --checkpoint checkpoints/smolvla_se3/smolvla_flow_seed0_best.pt

    # Evaluate chunk head
    python src/evaluate_smolvla.py --config configs/smolvla_se3.yaml \
        --checkpoint checkpoints/smolvla_se3/smolvla_chunk_seed0_best.pt \
        --head-type chunk

    # Evaluate uncertainty head (includes conformal coverage)
    python src/evaluate_smolvla.py --config configs/smolvla_se3.yaml \
        --checkpoint checkpoints/smolvla_se3/smolvla_uncertainty_seed0_best.pt \
        --head-type uncertainty
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backbones.smolvla_backbone import SmolVLABackbone, SmolVLAAdapter
from models.se3_action_head import SE3ActionPredictor
from models.geodesic_chunking import GeodesicChunkPredictor
from models.uncertainty_head import UncertaintyAwareFlowHead, ConformalCalibrator
from utils.metrics import geodesic_rmse, rotation_rmse, translation_rmse
from utils.se3_utils import geodesic_distance


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def build_model_from_checkpoint(config, checkpoint_path, device, head_type='flow'):
    """Build model and load checkpoint weights."""
    from train_smolvla import build_smolvla_model

    model = build_smolvla_model(config, device, head_type=head_type)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    epoch = ckpt.get('epoch', '?')
    seed = ckpt.get('seed', '?')

    print(f"  Loaded checkpoint: epoch={epoch}, seed={seed}")
    return model, ckpt


@torch.no_grad()
def evaluate_flow(model, dataloader, device, n_flow_steps=10):
    """Evaluate standard flow head."""
    model.eval()
    all_pred = []
    all_target = []
    all_gripper_pred = []
    all_gripper_target = []

    for batch in tqdm(dataloader, desc="Evaluating [flow]", leave=False):
        observations = {k: v.to(device) for k, v in batch['observations'].items()}
        language = batch['language'].to(device)
        target_actions = batch['target_actions'].to(device)
        target_gripper = batch.get('target_gripper')
        if target_gripper is not None:
            target_gripper = target_gripper.to(device)

        h = model.encode(observations, language)
        pred_actions, pred_gripper = model.action_head.predict(h, n_steps=n_flow_steps)

        all_pred.append(pred_actions)
        all_target.append(target_actions)
        all_gripper_pred.append(pred_gripper)
        if target_gripper is not None:
            all_gripper_target.append(target_gripper)

    all_pred = torch.cat(all_pred, dim=0)
    all_target = torch.cat(all_target, dim=0)

    results = {
        'geodesic_rmse': geodesic_rmse(all_pred, all_target).item(),
        'rotation_rmse': rotation_rmse(all_pred, all_target).item(),
        'translation_rmse': translation_rmse(all_pred, all_target).item(),
        'n_samples': all_pred.shape[0],
    }

    # Gripper accuracy
    if all_gripper_target:
        all_gripper_pred = torch.cat(all_gripper_pred, dim=0)
        all_gripper_target = torch.cat(all_gripper_target, dim=0)
        gripper_acc = ((all_gripper_pred > 0.5) == (all_gripper_target > 0.5)).float().mean()
        results['gripper_accuracy'] = gripper_acc.item()

    return results


@torch.no_grad()
def evaluate_chunk(model, dataloader, device):
    """Evaluate geodesic chunking head."""
    model.eval()
    all_pred = []
    all_target = []

    for batch in tqdm(dataloader, desc="Evaluating [chunk]", leave=False):
        observations = {k: v.to(device) for k, v in batch['observations'].items()}
        language = batch['language'].to(device)
        target_actions = batch['target_actions'].to(device)

        h = model.encode(observations, language)
        pred_chunk, pred_gripper = model.action_head(h)

        # Evaluate all actions in the chunk
        all_pred.append(pred_chunk)
        all_target.append(target_actions)

    all_pred = torch.cat(all_pred, dim=0)  # [N, H, 4, 4]
    all_target = torch.cat(all_target, dim=0)

    B, H = all_pred.shape[:2]

    # Per-step metrics
    step_metrics = []
    for h in range(H):
        step_results = {
            'step': h,
            'geodesic_rmse': geodesic_rmse(all_pred[:, h], all_target[:, h]).item(),
            'rotation_rmse': rotation_rmse(all_pred[:, h], all_target[:, h]).item(),
            'translation_rmse': translation_rmse(all_pred[:, h], all_target[:, h]).item(),
        }
        step_metrics.append(step_results)

    # Overall (average across chunk)
    results = {
        'geodesic_rmse': np.mean([m['geodesic_rmse'] for m in step_metrics]),
        'rotation_rmse': np.mean([m['rotation_rmse'] for m in step_metrics]),
        'translation_rmse': np.mean([m['translation_rmse'] for m in step_metrics]),
        'chunk_size': H,
        'n_samples': B,
        'per_step': step_metrics,
    }

    # Temporal smoothness: mean geodesic distance between consecutive actions
    smoothness = []
    for h in range(H - 1):
        dists = geodesic_distance(all_pred[:, h], all_pred[:, h + 1])
        smoothness.append(dists.mean().item())
    results['temporal_smoothness'] = np.mean(smoothness)

    return results


@torch.no_grad()
def evaluate_uncertainty(model, dataloader, device, n_flow_steps=10, alpha=0.1):
    """Evaluate uncertainty-aware head with conformal prediction."""
    model.eval()
    all_pred = []
    all_target = []
    all_uncertainty = []

    for batch in tqdm(dataloader, desc="Evaluating [uncertainty]", leave=False):
        observations = {k: v.to(device) for k, v in batch['observations'].items()}
        language = batch['language'].to(device)
        target_actions = batch['target_actions'].to(device)

        h = model.encode(observations, language)
        mean_action, gripper, variance, samples = \
            model.action_head.predict_with_uncertainty(h, n_steps=n_flow_steps)

        all_pred.append(mean_action)
        all_target.append(target_actions)
        all_uncertainty.append(variance)

    all_pred = torch.cat(all_pred, dim=0)
    all_target = torch.cat(all_target, dim=0)
    all_uncertainty = torch.cat(all_uncertainty, dim=0)

    # Core metrics
    results = {
        'geodesic_rmse': geodesic_rmse(all_pred, all_target).item(),
        'rotation_rmse': rotation_rmse(all_pred, all_target).item(),
        'translation_rmse': translation_rmse(all_pred, all_target).item(),
        'n_samples': all_pred.shape[0],
        'mean_uncertainty': all_uncertainty.mean().item(),
        'std_uncertainty': all_uncertainty.std().item(),
    }

    # Conformal prediction coverage
    # Split into calibration (first half) and test (second half)
    n = all_pred.shape[0]
    n_cal = n // 2

    cal_dists = geodesic_distance(all_pred[:n_cal], all_target[:n_cal])
    test_dists = geodesic_distance(all_pred[n_cal:], all_target[n_cal:])

    calibrator = ConformalCalibrator(alpha=alpha)
    q_alpha = calibrator.calibrate(cal_dists)
    coverage = calibrator.get_coverage(test_dists)

    results['conformal'] = {
        'alpha': alpha,
        'target_coverage': 1 - alpha,
        'empirical_coverage': coverage,
        'q_alpha': q_alpha,
        'n_calibration': n_cal,
        'n_test': n - n_cal,
    }

    return results


def print_results_table(results, head_type):
    """Pretty-print evaluation results."""
    print(f"\n{'='*70}")
    print(f"  EVALUATION RESULTS — SmolVLA + SE(3) [{head_type}]")
    print(f"{'='*70}")

    if 'per_family' in results:
        # Multi-family results
        print(f"  {'Family':<22s} {'G-RMSE':>8s} {'R-RMSE':>8s} {'T-RMSE':>8s} {'N':>6s}")
        print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")
        for family, m in results['per_family'].items():
            print(
                f"  {family:<22s} "
                f"{m.get('geodesic_rmse', 0):8.4f} "
                f"{m.get('rotation_rmse', 0):8.4f} "
                f"{m.get('translation_rmse', 0):8.4f} "
                f"{m.get('n_samples', 0):6d}"
            )
    else:
        m = results
        print(f"  G-RMSE: {m.get('geodesic_rmse', 0):.4f}")
        print(f"  R-RMSE: {m.get('rotation_rmse', 0):.4f}")
        print(f"  T-RMSE: {m.get('translation_rmse', 0):.4f}")
        print(f"  N:      {m.get('n_samples', 0)}")

    if head_type == 'chunk' and 'per_step' in results:
        print(f"\n  Per-step breakdown:")
        for step in results['per_step']:
            print(f"    Step {step['step']}: G={step['geodesic_rmse']:.4f} "
                  f"R={step['rotation_rmse']:.4f} T={step['translation_rmse']:.4f}")
        print(f"  Temporal smoothness: {results.get('temporal_smoothness', 0):.4f}")

    if head_type == 'uncertainty' and 'conformal' in results:
        conf = results['conformal']
        print(f"\n  Conformal Prediction:")
        print(f"    Target coverage:    {conf['target_coverage']:.1%}")
        print(f"    Empirical coverage: {conf['empirical_coverage']:.1%}")
        print(f"    q_alpha:            {conf['q_alpha']:.4f}")
        print(f"    Mean uncertainty:   {results.get('mean_uncertainty', 0):.4f}")

    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate SmolVLA + SE(3)")
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--head-type', type=str, default='flow',
                        choices=['flow', 'chunk', 'uncertainty'],
                        help='Action head type')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for data generation')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu/mps)')
    parser.add_argument('--n-flow-steps', type=int, default=10,
                        help='Flow integration steps at eval')
    args = parser.parse_args()

    config = load_config(args.config)

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print("=" * 70)
    print(f"  SmolVLA + SE(3) Evaluation")
    print(f"  Config:     {args.config}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Head type:  {args.head_type}")
    print(f"  Device:     {device}")
    print("=" * 70)

    # Build model + load weights
    print("\n[1/3] Loading model...")
    model, ckpt = build_model_from_checkpoint(
        config, args.checkpoint, device, head_type=args.head_type
    )

    # Create dataloaders
    print("\n[2/3] Creating dataloaders...")
    from train_smolvla import create_smolvla_dataloaders
    _, val_loaders = create_smolvla_dataloaders(config, seed=args.seed)
    for name, loader in val_loaders.items():
        print(f"  Val [{name}]: {len(loader.dataset)} samples")

    # Evaluate
    print("\n[3/3] Evaluating...")
    eval_fn = {
        'flow': evaluate_flow,
        'chunk': evaluate_chunk,
        'uncertainty': evaluate_uncertainty,
    }[args.head_type]

    all_results = {}
    for family_name, loader in val_loaders.items():
        print(f"\n  Family: {family_name}")
        if args.head_type == 'uncertainty':
            results = eval_fn(model, loader, device, n_flow_steps=args.n_flow_steps)
        elif args.head_type == 'flow':
            results = eval_fn(model, loader, device, n_flow_steps=args.n_flow_steps)
        else:
            results = eval_fn(model, loader, device)
        all_results[family_name] = results

    # Print results
    print_results_table({'per_family': all_results}, args.head_type)

    # Save
    output_path = args.output or f"results/smolvla_{args.head_type}_eval.json"
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    # Make results JSON-serializable
    def make_serializable(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    output_data = {
        'config': args.config,
        'checkpoint': args.checkpoint,
        'head_type': args.head_type,
        'seed': args.seed,
        'n_flow_steps': args.n_flow_steps,
        'checkpoint_epoch': ckpt.get('epoch', '?'),
        'results': make_serializable(all_results),
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\n  Results saved: {output_path}")


if __name__ == '__main__':
    main()
