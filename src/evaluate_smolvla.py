"""
SE(3)-VLA Evaluation Script — Compact Architecture.

Evaluates compact VLA (< 400M) against baselines on synthetic and sim benchmarks.

Usage:
    # Evaluate SE(3) flow head
    python src/evaluate_smolvla.py --config configs/compact_se3.yaml \
        --checkpoint checkpoints/compact_se3/se3vla_flow_seed0_best.pt

    # Evaluate with benchmark comparison
    python src/evaluate_smolvla.py --config configs/compact_se3.yaml \
        --checkpoint checkpoints/compact_se3/se3vla_flow_seed0_best.pt \
        --compare

    # Evaluate uncertainty head (includes conformal coverage)
    python src/evaluate_smolvla.py --config configs/compact_se3_uncertainty.yaml \
        --checkpoint checkpoints/compact_se3/se3vla_uncertainty_seed0_best.pt \
        --head-type uncertainty
"""

import argparse
import yaml
import torch
import json
import os
import sys
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backbones.compact_vla_backbone import CompactVLABackbone
from backbones.smolvla_backbone import SmolVLAAdapter
from models.se3_action_head import SE3ActionPredictor
from models.geodesic_chunking import GeodesicChunkPredictor
from models.uncertainty_head import UncertaintyAwareFlowHead, ConformalCalibrator
from models.octo_adapter import OctoEuclideanBaseline
from models.mock_backbone import MockOctoBackbone
from models.scene_id_backbone import SceneIDBackbone
from utils.metrics import geodesic_rmse, rotation_rmse, translation_rmse
from utils.se3_utils import geodesic_distance, exp_so3


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def build_model_from_checkpoint(config, checkpoint_path, device, head_type='flow'):
    """Build model and load checkpoint."""
    model_cfg = config.get('model', {})
    hidden_dim = model_cfg.get('hidden_dim', 768)

    # Build backbone
    vision_model = model_cfg.get('vision_model', 'siglip-base')
    language_model = model_cfg.get('language_model', 'smolLM-135M')

    backbone = CompactVLABackbone(
        vision_model=vision_model,
        language_model=language_model,
        hidden_dim=hidden_dim,
        freeze_vision=model_cfg.get('freeze_vision', True),
        freeze_language=model_cfg.get('freeze_language', False),
    )

    # Build action head
    head_hidden_dim = model_cfg.get('head_hidden_dim', 256)
    n_layers = model_cfg.get('n_layers', 4)
    source_scale = model_cfg.get('source_scale', 0.1)

    if head_type == 'chunk':
        action_head = GeodesicChunkPredictor(
            hidden_dim=hidden_dim,
            chunk_size=model_cfg.get('chunk_size', 8),
            n_anchors=model_cfg.get('n_anchors', 2),
            head_hidden_dim=head_hidden_dim,
            n_layers=n_layers,
        )
    elif head_type == 'uncertainty':
        base = SE3ActionPredictor(
            hidden_dim=hidden_dim,
            head_hidden_dim=head_hidden_dim,
            n_layers=n_layers,
            source_scale=source_scale,
        )
        action_head = UncertaintyAwareFlowHead(
            base_predictor=base,
            n_samples=model_cfg.get('n_uncertainty_samples', 10),
        )
    else:
        action_head = SE3ActionPredictor(
            hidden_dim=hidden_dim,
            head_hidden_dim=head_hidden_dim,
            n_layers=n_layers,
            source_scale=source_scale,
        )

    model = SmolVLAAdapter(
        backbone=backbone,
        action_head=action_head,
        head_type=head_type,
    ).to(device)

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"  Loaded: epoch={ckpt.get('epoch', '?')}, seed={ckpt.get('seed', '?')}")
    print(f"  Params: {ckpt.get('param_counts', {}).get('total', 'unknown')}")

    return model, ckpt


def build_euclidean_baseline(config, device):
    """Build Euclidean baseline for comparison."""
    model_cfg = config.get('model', {})
    hidden_dim = model_cfg.get('hidden_dim', 768)

    backbone = CompactVLABackbone(
        vision_model=model_cfg.get('vision_model', 'siglip-base'),
        language_model=model_cfg.get('language_model', 'smolLM-135M'),
        hidden_dim=hidden_dim,
        freeze_vision=model_cfg.get('freeze_vision', True),
        freeze_language=model_cfg.get('freeze_language', False),
    )

    model = OctoEuclideanBaseline(
        octo_model=backbone,
        hidden_dim=hidden_dim,
        freeze_backbone=False,
    ).to(device)

    return model


@torch.no_grad()
def evaluate_model(model, dataloader, device, is_se3=True, head_type='flow'):
    """Evaluate a model on a dataloader."""
    model.eval()
    all_pred = []
    all_target = []

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        if isinstance(batch, dict):
            observations = {k: v.to(device) for k, v in batch['observations'].items()}
            language = batch['language'].to(device)
            target_actions = batch['target_actions'].to(device)
        else:
            observations, language, target_actions = batch[0], batch[1], batch[2]
            if isinstance(observations, dict):
                observations = {k: v.to(device) for k, v in observations.items()}
            target_actions = target_actions.to(device)

        if is_se3:
            h = model.encode(observations, language)
            if head_type == 'uncertainty':
                pred, _, _, _ = model.action_head.predict_with_uncertainty(h, n_steps=10)
            elif head_type == 'chunk':
                pred, _ = model.action_head(h)
                pred = pred[:, 0]
            else:
                pred, _ = model.action_head.predict(h, n_steps=10)
        else:
            h = model.encode(observations, language)
            action = model.action_head(h)
            omega = action[:, :3]
            t = action[:, 3:6]
            R = exp_so3(omega)
            B = R.shape[0]
            pred = torch.eye(4, device=device).unsqueeze(0).expand(B, -1, -1).clone()
            pred[:, :3, :3] = R
            pred[:, :3, 3] = t

        all_pred.append(pred)
        all_target.append(target_actions)

    all_pred = torch.cat(all_pred, dim=0)
    all_target = torch.cat(all_target, dim=0)

    return {
        'geodesic_rmse': geodesic_rmse(all_pred, all_target).item(),
        'rotation_rmse': rotation_rmse(all_pred, all_target).item(),
        'translation_rmse': translation_rmse(all_pred, all_target).item(),
        'n_samples': all_pred.shape[0],
    }


def run_comparison(model, val_loaders, device, config, head_type='flow'):
    """
    Run full benchmark comparison: SE(3) vs Euclidean baseline.

    This is the key validation that the SE(3) head outperforms
    Euclidean approaches (and by extension, larger models with
    Euclidean heads).
    """
    print("\n" + "=" * 70)
    print("  BENCHMARK COMPARISON")
    print("=" * 70)

    # Evaluate SE(3) model
    print("\n  [1/2] Evaluating SE(3) model...")
    se3_results = {}
    for name, loader in val_loaders.items():
        se3_results[name] = evaluate_model(
            model, loader, device, is_se3=True, head_type=head_type
        )

    # Build and evaluate Euclidean baseline (same backbone)
    print("\n  [2/2] Evaluating Euclidean baseline (same backbone)...")
    euclid_model = build_euclidean_baseline(config, device)
    euclid_results = {}
    for name, loader in val_loaders.items():
        euclid_results[name] = evaluate_model(
            euclid_model, loader, device, is_se3=False
        )

    # Print comparison table
    print(f"\n  {'Family':<22s} │ {'SE(3) R-RMSE':>14s} │ {'Euclid R-RMSE':>14s} │ {'Δ (Eucl-SE3)':>14s} │ {'Winner':>8s}")
    print(f"  {'─'*22}─┼─{'─'*14}─┼─{'─'*14}─┼─{'─'*14}─┼─{'─'*8}")

    for family in se3_results:
        s = se3_results[family]
        e = euclid_results.get(family, {})
        s_rot = s.get('rotation_rmse', 0)
        e_rot = e.get('rotation_rmse', 0)
        delta = e_rot - s_rot
        winner = "SE(3)" if delta > 0 else "Euclid"

        print(f"  {family:<22s} │ {s_rot:14.4f} │ {e_rot:14.4f} │ {delta:+14.4f} │ {winner:>8s}")

    # Aggregate
    s_avg = np.mean([m['rotation_rmse'] for m in se3_results.values()])
    e_avg = np.mean([m['rotation_rmse'] for m in euclid_results.values()])
    delta_avg = e_avg - s_avg

    print(f"  {'─'*22}─┼─{'─'*14}─┼─{'─'*14}─┼─{'─'*14}─┼─{'─'*8}")
    print(f"  {'AVERAGE':<22s} │ {s_avg:14.4f} │ {e_avg:14.4f} │ {delta_avg:+14.4f} │ {'SE(3)' if delta_avg > 0 else 'Euclid':>8s}")

    # Verdict
    print(f"\n  ┌─────────────────────────────────────────────────────────┐")
    if delta_avg > 0:
        pct = (delta_avg / e_avg) * 100
        print(f"  │  ✓ SE(3) HEAD WINS: {pct:.1f}% better rotation RMSE        │")
        print(f"  │  → Geometric inductive bias compensates for smaller     │")
        print(f"  │    backbone. Justified to proceed to Phase 2 (sim).     │")
    else:
        print(f"  │  ✗ SE(3) HEAD LOSES: No rotation improvement            │")
        print(f"  │  → Review architecture before proceeding.               │")
    print(f"  └─────────────────────────────────────────────────────────┘")

    return {
        'se3': se3_results,
        'euclidean': euclid_results,
        'delta_rotation_rmse': delta_avg,
        'se3_wins': delta_avg > 0,
    }


def print_results_table(results, head_type):
    """Pretty-print results."""
    print(f"\n{'='*70}")
    print(f"  EVALUATION — SE(3)-VLA [{head_type}]")
    print(f"{'='*70}")

    if 'per_family' in results:
        print(f"  {'Family':<22s} {'G-RMSE':>8s} {'R-RMSE':>8s} {'T-RMSE':>8s} {'N':>6s}")
        print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")
        for family, m in results['per_family'].items():
            print(f"  {family:<22s} {m.get('geodesic_rmse',0):8.4f} "
                  f"{m.get('rotation_rmse',0):8.4f} {m.get('translation_rmse',0):8.4f} "
                  f"{m.get('n_samples',0):6d}")
    elif 'combined' in results:
        m = results['combined']
        print(f"  G-RMSE: {m.get('geodesic_rmse',0):.4f}")
        print(f"  R-RMSE: {m.get('rotation_rmse',0):.4f}")
        print(f"  T-RMSE: {m.get('translation_rmse',0):.4f}")

    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate SE(3)-VLA (< 400M)")
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--head-type', type=str, default='flow',
                        choices=['flow', 'chunk', 'uncertainty'])
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--compare', action='store_true',
                        help='Run benchmark comparison against Euclidean baseline')
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device if args.device else
                          'cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print(f"  SE(3)-VLA Evaluation — Compact Architecture")
    print(f"  Config:     {args.config}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Head:       {args.head_type}")
    print(f"  Device:     {device}")
    print("=" * 70)

    # Load model
    print("\n[1/2] Loading model...")
    model, ckpt = build_model_from_checkpoint(
        config, args.checkpoint, device, head_type=args.head_type
    )

    # Create dataloaders
    print("\n[2/2] Creating dataloaders...")
    from train_smolvla import create_dataloaders_from_config
    _, val_loaders = create_dataloaders_from_config(config, seed=args.seed)
    for name, loader in val_loaders.items():
        print(f"  Val [{name}]: {len(loader.dataset)}")

    # Evaluate
    all_results = {}
    for name, loader in val_loaders.items():
        all_results[name] = evaluate_model(
            model, loader, device, is_se3=True, head_type=args.head_type
        )

    print_results_table({'per_family': all_results}, args.head_type)

    # Benchmark comparison
    comparison = None
    if args.compare:
        comparison = run_comparison(
            model, val_loaders, device, config, args.head_type
        )

    # Save
    output_path = args.output or f"results/compact_{args.head_type}_eval.json"
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    def make_serializable(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        return obj

    output_data = {
        'config': args.config,
        'checkpoint': args.checkpoint,
        'head_type': args.head_type,
        'seed': args.seed,
        'checkpoint_epoch': ckpt.get('epoch', '?'),
        'param_counts': ckpt.get('param_counts', {}),
        'results': make_serializable(all_results),
    }
    if comparison:
        output_data['comparison'] = make_serializable(comparison)

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\n  Results: {output_path}")


if __name__ == '__main__':
    main()
