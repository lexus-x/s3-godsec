"""
Training script for SE(3)-VLA.

Supports both SE(3) flow matching head and Euclidean baseline.
Uses synthetic SE(3) demonstration data for pipeline validation.

Usage:
    # Train SE(3) model
    python src/train.py --config configs/octo_se3.yaml

    # Train Euclidean baseline
    python src/train.py --config configs/octo_baseline.yaml
"""

import argparse
import yaml
import torch
import os
import sys
import time
import json
from tqdm import tqdm

# Ensure src/ is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.octo_adapter import OctoSE3, OctoEuclideanBaseline
from models.mock_backbone import MockOctoBackbone
from models.geodesic_loss import GeodesicMSELoss
from utils.metrics import geodesic_rmse, rotation_rmse, translation_rmse
from training.data_loader import create_dataloaders


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def train_one_epoch(model, dataloader, optimizer, device, config, is_se3=True):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0
    loss_components = {}

    for batch in tqdm(dataloader, desc="Training", leave=False):
        observations = {k: v.to(device) for k, v in batch['observations'].items()}
        language = batch['language'].to(device)
        target_actions = batch['target_actions'].to(device)
        target_gripper = batch.get('target_gripper')
        if target_gripper is not None:
            target_gripper = target_gripper.to(device)

        optimizer.zero_grad()

        if is_se3:
            # SE(3) model expects SE(3) matrices as targets
            loss, loss_dict = model.compute_loss(
                observations, language, target_actions, target_gripper
            )
        else:
            # Euclidean baseline expects R^6 (axis-angle + translation)
            # Extract from SE(3) matrix: rotation as log map, translation directly
            from utils.se3_utils import log_so3
            R = target_actions[:, :3, :3]
            t = target_actions[:, :3, 3]
            omega = log_so3(R)  # [B, 3]
            target_flat = torch.cat([omega, t], dim=-1)  # [B, 6]
            loss, loss_dict = model.compute_loss(
                observations, language, target_flat, target_gripper
            )

        loss.backward()

        # Gradient clipping
        grad_clip = config['training'].get('gradient_clip_norm', 1.0)
        trainable_params = model.trainable_parameters()
        if trainable_params:
            torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)

        optimizer.step()

        total_loss += loss_dict['total_loss']
        n_batches += 1

        # Accumulate loss components
        for k, v in loss_dict.items():
            if k not in loss_components:
                loss_components[k] = 0
            loss_components[k] += v

    avg_loss = total_loss / max(n_batches, 1)
    avg_components = {k: v / max(n_batches, 1) for k, v in loss_components.items()}

    return avg_loss, avg_components


@torch.no_grad()
def validate(model, dataloader, device, is_se3=True):
    """Run validation and compute metrics."""
    model.eval()
    all_pred = []
    all_target = []

    for batch in tqdm(dataloader, desc="Validating", leave=False):
        observations = {k: v.to(device) for k, v in batch['observations'].items()}
        language = batch['language'].to(device)
        target_actions = batch['target_actions'].to(device)

        if is_se3:
            h = model.encode(observations, language)
            pred_actions, _ = model.action_predictor.predict(h, n_steps=10)
            all_pred.append(pred_actions)
            all_target.append(target_actions)
        else:
            h = model.encode(observations, language)
            action = model.action_head(h)
            # Convert Euclidean prediction to SE(3) for metric comparison
            from utils.se3_utils import exp_so3
            omega = action[:, :3]
            t = action[:, 3:6]
            R = exp_so3(omega)
            B = R.shape[0]
            T_pred = torch.eye(4, device=device).unsqueeze(0).expand(B, -1, -1).clone()
            T_pred[:, :3, :3] = R
            T_pred[:, :3, 3] = t
            all_pred.append(T_pred)
            all_target.append(target_actions)

    all_pred = torch.cat(all_pred, dim=0)
    all_target = torch.cat(all_target, dim=0)

    g_rmse = geodesic_rmse(all_pred, all_target).item()
    r_rmse = rotation_rmse(all_pred, all_target).item()
    t_rmse = translation_rmse(all_pred, all_target).item()

    return {
        'geodesic_rmse': g_rmse,
        'rotation_rmse': r_rmse,
        'translation_rmse': t_rmse,
    }


def main():
    parser = argparse.ArgumentParser(description="Train SE(3)-VLA")
    parser.add_argument('--config', type=str, required=True, help="Path to config YAML")
    parser.add_argument('--resume', type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument('--epochs', type=int, default=None, help="Override number of epochs")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_name = config['model']['name']
    is_se3 = (model_name == 'OctoSE3')
    hidden_dim = config['model'].get('hidden_dim', 768)

    print("=" * 60)
    print(f"  SE(3)-VLA Training")
    print(f"  Model: {model_name}")
    print(f"  Device: {device}")
    print(f"  Config: {args.config}")
    print("=" * 60)

    # Create mock backbone
    print("\n[1/4] Creating model...")
    backbone = MockOctoBackbone(hidden_dim=hidden_dim)

    if is_se3:
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
    param_counts = model.count_parameters() if hasattr(model, 'count_parameters') else {}
    print(f"  Total params:     {param_counts.get('total', 'N/A'):,}")
    print(f"  Trainable params: {param_counts.get('trainable', 'N/A'):,}")
    print(f"  Frozen params:    {param_counts.get('frozen', 'N/A'):,}")

    # Create dataloaders
    print("\n[2/4] Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(config)
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples:   {len(val_loader.dataset)}")
    print(f"  Batch size:    {config['training'].get('batch_size', 32)}")

    # Optimizer and scheduler
    print("\n[3/4] Setting up optimizer...")
    trainable_params = model.trainable_parameters()
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 1e-5),
    )

    n_epochs = args.epochs or config['training']['n_epochs']
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs
    )
    print(f"  Learning rate: {config['training']['learning_rate']}")
    print(f"  Epochs: {n_epochs}")

    # Resume if requested
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        print(f"\n  Resuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0)

    # Checkpoint directory
    ckpt_dir = config.get('logging', {}).get('checkpoint_dir', 'checkpoints/')
    os.makedirs(ckpt_dir, exist_ok=True)

    # Training loop
    print("\n[4/4] Starting training...\n")
    best_val_metric = float('inf')
    history = []
    start_time = time.time()

    for epoch in range(start_epoch, n_epochs):
        epoch_start = time.time()

        # Train
        train_loss, train_components = train_one_epoch(
            model, train_loader, optimizer, device, config, is_se3=is_se3
        )

        # Step scheduler
        scheduler.step()

        # Validate
        val_metrics = validate(model, val_loader, device, is_se3=is_se3)

        epoch_time = time.time() - epoch_start

        # Log
        lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch+1:3d}/{n_epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"G-RMSE: {val_metrics['geodesic_rmse']:.4f} | "
            f"R-RMSE: {val_metrics['rotation_rmse']:.4f} | "
            f"T-RMSE: {val_metrics['translation_rmse']:.4f} | "
            f"LR: {lr:.2e} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Track history
        record = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'lr': lr,
            'epoch_time': epoch_time,
            **{f'val_{k}': v for k, v in val_metrics.items()},
            **{f'train_{k}': v for k, v in train_components.items()},
        }
        history.append(record)

        # Save checkpoint
        save_interval = config.get('logging', {}).get('save_interval', 10)
        if (epoch + 1) % save_interval == 0 or (epoch + 1) == n_epochs:
            ckpt_path = os.path.join(ckpt_dir, f"{model_name}_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'config': config,
            }, ckpt_path)
            print(f"  → Saved checkpoint: {ckpt_path}")

        # Save best
        if val_metrics['geodesic_rmse'] < best_val_metric:
            best_val_metric = val_metrics['geodesic_rmse']
            best_path = os.path.join(ckpt_dir, f"{model_name}_best.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'config': config,
            }, best_path)

    # Summary
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"  Training complete!")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Best geodesic RMSE: {best_val_metric:.4f}")
    print(f"  Final checkpoint: {ckpt_dir}/{model_name}_best.pt")
    print("=" * 60)

    # Save training history
    history_path = os.path.join(ckpt_dir, f"{model_name}_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"  Training history saved: {history_path}")


if __name__ == '__main__':
    main()
