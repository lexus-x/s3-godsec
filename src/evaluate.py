"""
Evaluation script for SE(3)-VLA.

Loads a trained checkpoint and runs full evaluation with geodesic metrics.

Usage:
    python src/evaluate.py --config configs/octo_se3.yaml --checkpoint checkpoints/OctoSE3_best.pt
    python src/evaluate.py --config configs/octo_baseline.yaml --checkpoint checkpoints/OctoEuclideanBaseline_best.pt
"""

import argparse
import yaml
import torch
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.octo_adapter import OctoSE3, OctoEuclideanBaseline
from models.mock_backbone import MockOctoBackbone
from training.data_loader import create_dataloaders
from utils.metrics import (
    geodesic_rmse,
    rotation_rmse,
    translation_rmse,
)
from utils.se3_utils import geodesic_distance, exp_so3


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def build_model(config, device):
    """Build model from config."""
    model_name = config['model']['name']
    hidden_dim = config['model'].get('hidden_dim', 768)

    backbone = MockOctoBackbone(hidden_dim=hidden_dim)

    if model_name == 'OctoSE3':
        model = OctoSE3(
            octo_model=backbone,
            hidden_dim=hidden_dim,
            head_hidden_dim=config['model'].get('head_hidden_dim', 256),
            n_layers=config['model'].get('n_layers', 4),
            n_flow_steps_train=config['training'].get('n_flow_steps_train', 10),
            n_flow_steps_eval=config['training'].get('n_flow_steps_eval', 10),
            source_scale=config['model'].get('source_scale', 0.1),
            freeze_backbone=config['model'].get('freeze_backbone', False),
        )
    else:
        model = OctoEuclideanBaseline(
            octo_model=backbone,
            hidden_dim=hidden_dim,
            freeze_backbone=config['model'].get('freeze_backbone', False),
        )

    model = model.to(device)
    return model


@torch.no_grad()
def evaluate_model(model, dataloader, device, is_se3=True):
    """Run full evaluation."""
    model.eval()

    all_pred = []
    all_target = []
    all_geodesic_dists = []

    for batch in dataloader:
        observations = {k: v.to(device) for k, v in batch['observations'].items()}
        language = batch['language'].to(device)
        target_actions = batch['target_actions'].to(device)

        if is_se3:
            h = model.encode(observations, language)
            pred_actions, _ = model.action_predictor.predict(h, n_steps=10)
        else:
            h = model.encode(observations, language)
            action = model.action_head(h)
            # Convert Euclidean prediction to SE(3)
            omega = action[:, :3]
            t = action[:, 3:6]
            R = exp_so3(omega)
            B = R.shape[0]
            pred_actions = torch.eye(4, device=device).unsqueeze(0).expand(B, -1, -1).clone()
            pred_actions[:, :3, :3] = R
            pred_actions[:, :3, 3] = t

        all_pred.append(pred_actions)
        all_target.append(target_actions)

        # Per-batch geodesic distances
        dists = geodesic_distance(pred_actions, target_actions)
        all_geodesic_dists.append(dists)

    all_pred = torch.cat(all_pred, dim=0)
    all_target = torch.cat(all_target, dim=0)
    all_dists = torch.cat(all_geodesic_dists, dim=0)

    results = {
        'geodesic_rmse': geodesic_rmse(all_pred, all_target).item(),
        'rotation_rmse': rotation_rmse(all_pred, all_target).item(),
        'translation_rmse': translation_rmse(all_pred, all_target).item(),
        'mean_geodesic_distance': all_dists.mean().item(),
        'median_geodesic_distance': all_dists.median().item(),
        'max_geodesic_distance': all_dists.max().item(),
        'std_geodesic_distance': all_dists.std().item(),
        'n_samples': len(all_dists),
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate SE(3)-VLA")
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = config['model']['name']
    is_se3 = (model_name == 'OctoSE3')

    print("=" * 60)
    print(f"  SE(3)-VLA Evaluation")
    print(f"  Model: {model_name}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Device: {device}")
    print("=" * 60)

    # Build model
    print("\n[1/3] Building model...")
    model = build_model(config, device)

    # Load checkpoint
    print(f"[2/3] Loading checkpoint...")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    ckpt_epoch = ckpt.get('epoch', '?')
    print(f"  Loaded from epoch {ckpt_epoch}")

    if 'val_metrics' in ckpt:
        print(f"  Checkpoint val metrics: {ckpt['val_metrics']}")

    # Create evaluation dataloader
    print(f"[3/3] Running evaluation...")
    _, val_loader = create_dataloaders(config, seed=42)
    print(f"  Eval samples: {len(val_loader.dataset)}")

    results = evaluate_model(model, val_loader, device, is_se3=is_se3)

    # Print results
    print("\n" + "=" * 60)
    print(f"  EVALUATION RESULTS — {model_name}")
    print("=" * 60)
    print(f"  Geodesic RMSE:          {results['geodesic_rmse']:.6f}")
    print(f"  Rotation RMSE (rad):    {results['rotation_rmse']:.6f}")
    print(f"  Translation RMSE:       {results['translation_rmse']:.6f}")
    print(f"  Mean Geodesic Distance: {results['mean_geodesic_distance']:.6f}")
    print(f"  Median Geodesic Dist:   {results['median_geodesic_distance']:.6f}")
    print(f"  Max Geodesic Distance:  {results['max_geodesic_distance']:.6f}")
    print(f"  Std Geodesic Distance:  {results['std_geodesic_distance']:.6f}")
    print(f"  N samples:              {results['n_samples']}")
    print("=" * 60)

    # Save results
    output_path = args.output or f"results/{model_name}_evaluation.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {output_path}")


if __name__ == '__main__':
    main()
