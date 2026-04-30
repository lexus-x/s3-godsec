"""
Evaluation script for SE(3)-VLA controlled experiments.

Reports per-family metrics: rotation_RMSE, translation_RMSE, geodesic_RMSE
for each family (rotation_heavy, translation_heavy, combined).

Usage:
    python src/evaluate.py --config configs/octo_se3.yaml \
        --checkpoint checkpoints/OctoSE3_scene_id_seed0_best.pt
"""

import argparse
import yaml
import torch
import json
import os
import sys
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.octo_adapter import OctoSE3, OctoEuclideanBaseline
from models.mock_backbone import MockOctoBackbone
from models.scene_id_backbone import SceneIDBackbone
from training.data_loader import create_dataloaders
from utils.metrics import geodesic_rmse, rotation_rmse, translation_rmse
from utils.se3_utils import geodesic_distance, exp_so3


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def make_backbone(config):
    """Create backbone based on config."""
    hidden_dim = config['model'].get('hidden_dim', 768)
    kind = config['model'].get('backbone_kind', 'mock_cnn')
    if kind == 'scene_id':
        n_tasks = config['model'].get('n_tasks', 2)
        return SceneIDBackbone(n_tasks=n_tasks, hidden_dim=hidden_dim)
    else:
        return MockOctoBackbone(hidden_dim=hidden_dim)


def build_model(config, device):
    """Build model from config."""
    model_name = config['model']['name']
    hidden_dim = config['model'].get('hidden_dim', 768)
    backbone = make_backbone(config)

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

    return model.to(device)


@torch.no_grad()
def evaluate_loader(model, dataloader, device, is_se3=True):
    """Evaluate on a single dataloader."""
    model.eval()
    all_pred = []
    all_target = []

    for batch in tqdm(dataloader, leave=False):
        observations = {k: v.to(device) for k, v in batch['observations'].items()}
        language = batch['language'].to(device)
        target_actions = batch['target_actions'].to(device)

        if is_se3:
            h = model.encode(observations, language)
            pred_actions, _ = model.action_predictor.predict(h, n_steps=10)
        else:
            h = model.encode(observations, language)
            action = model.action_head(h)
            omega = action[:, :3]
            t = action[:, 3:6]
            R = exp_so3(omega)
            B = R.shape[0]
            pred_actions = torch.eye(4, device=device).unsqueeze(0).expand(B, -1, -1).clone()
            pred_actions[:, :3, :3] = R
            pred_actions[:, :3, 3] = t

        all_pred.append(pred_actions)
        all_target.append(target_actions)

    all_pred = torch.cat(all_pred, dim=0)
    all_target = torch.cat(all_target, dim=0)
    dists = geodesic_distance(all_pred, all_target)

    return {
        'geodesic_rmse': geodesic_rmse(all_pred, all_target).item(),
        'rotation_rmse': rotation_rmse(all_pred, all_target).item(),
        'translation_rmse': translation_rmse(all_pred, all_target).item(),
        'mean_geodesic_dist': dists.mean().item(),
        'median_geodesic_dist': dists.median().item(),
        'max_geodesic_dist': dists.max().item(),
        'std_geodesic_dist': dists.std().item(),
        'n_samples': len(dists),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate SE(3)-VLA")
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = config['model']['name']
    backbone_kind = config['model'].get('backbone_kind', 'mock_cnn')
    is_se3 = (model_name == 'OctoSE3')

    print("=" * 70)
    print(f"  SE(3)-VLA Evaluation")
    print(f"  Model:      {model_name}")
    print(f"  Backbone:   {backbone_kind}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Device:     {device}")
    print("=" * 70)

    # Build model and load checkpoint
    model = build_model(config, device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"  Loaded from epoch {ckpt.get('epoch', '?')}, seed {ckpt.get('seed', '?')}")

    # Create per-family eval loaders
    _, val_loaders = create_dataloaders(config, seed=args.seed)

    # Evaluate per family
    all_results = {}
    for family_name, loader in val_loaders.items():
        print(f"\n  Evaluating [{family_name}] ({len(loader.dataset)} samples)...")
        results = evaluate_loader(model, loader, device, is_se3)
        all_results[family_name] = results

    # Print per-family table
    print("\n" + "=" * 70)
    print(f"  EVALUATION RESULTS — {model_name} / {backbone_kind}")
    print("=" * 70)
    print(f"  {'Family':<22s} {'G-RMSE':>8s} {'R-RMSE':>8s} {'T-RMSE':>8s} {'N':>6s}")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")
    for family, m in all_results.items():
        print(
            f"  {family:<22s} "
            f"{m['geodesic_rmse']:8.4f} "
            f"{m['rotation_rmse']:8.4f} "
            f"{m['translation_rmse']:8.4f} "
            f"{m['n_samples']:6d}"
        )
    print("=" * 70)

    # Save
    output_path = args.output or f"results/{model_name}_{backbone_kind}_eval.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved: {output_path}")


if __name__ == '__main__':
    main()
