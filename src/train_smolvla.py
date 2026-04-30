"""
SmolVLA Training Script for SE(3)-VLA.

Trains SmolVLA-450M (frozen) + SE(3) action heads on real benchmarks.
Supports three head types:
  - flow: Standard SE(3) flow matching (single action)
  - chunk: Geodesic action chunking (temporally coherent chunks)
  - uncertainty: Uncertainty-aware flow matching (multi-sample + conformal)

Usage:
    # Standard SE(3) flow head
    python src/train_smolvla.py --config configs/smolvla_se3.yaml --seed 0

    # Geodesic chunking head
    python src/train_smolvla.py --config configs/smolvla_se3.yaml --seed 0 --head-type chunk

    # Uncertainty-aware head
    python src/train_smolvla.py --config configs/smolvla_se3.yaml --seed 0 --head-type uncertainty

    # Resume from checkpoint
    python src/train_smolvla.py --config configs/smolvla_se3.yaml --seed 0 --resume checkpoints/smolvla_se3/best.pt
"""

import argparse
import yaml
import torch
import torch.nn as nn
import os
import sys
import time
import json
import math
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backbones.smolvla_backbone import SmolVLABackbone, SmolVLAAdapter
from models.se3_action_head import SE3ActionPredictor
from models.geodesic_chunking import GeodesicChunkPredictor
from models.uncertainty_head import UncertaintyAwareFlowHead
from models.geodesic_loss import GeodesicMSELoss
from utils.metrics import geodesic_rmse, rotation_rmse, translation_rmse
from utils.se3_utils import log_se3, inverse_se3


# ---------------------------------------------------------------------------
# Config & Setup
# ---------------------------------------------------------------------------

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Data — LIBERO / MetaWorld / Synthetic adapters
# ---------------------------------------------------------------------------

def create_smolvla_dataloaders(config, seed=42):
    """
    Create dataloaders for SmolVLA training.

    Supports:
    - LIBERO (real benchmark): requires lerobot + libero installed
    - Synthetic (fallback for testing): uses the built-in synthetic dataset

    Returns:
        train_loader, val_loaders dict
    """
    benchmark = config.get('data', {}).get('benchmark', 'synthetic')

    if benchmark in ('libero_spatial', 'libero_object', 'libero_goal', 'libero_long'):
        return _create_libero_dataloaders(config, seed)
    elif benchmark == 'metaworld_mt10':
        return _create_metaworld_dataloaders(config, seed)
    else:
        # Fallback: use synthetic dataset for smoke testing
        print(f"  [INFO] Using synthetic dataset (benchmark='{benchmark}')")
        from training.data_loader import create_dataloaders
        return create_dataloaders(config, seed=seed)


def _create_libero_dataloaders(config, seed):
    """Load LIBERO demonstrations via lerobot."""
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        raise ImportError(
            "LIBERO requires lerobot. Install with:\n"
            "  git clone https://github.com/huggingface/lerobot.git\n"
            "  cd lerobot && pip install -e '.[smolvla]'"
        )

    data_cfg = config.get('data', {})
    benchmark = data_cfg.get('benchmark', 'libero_spatial')
    batch_size = config.get('training', {}).get('batch_size', 32)
    n_train = data_cfg.get('n_train_per_task', 100)
    n_val = data_cfg.get('n_val_per_task', 20)

    # Map benchmark name to LeRobot dataset ID
    dataset_map = {
        'libero_spatial': 'lerobot/libero_spatial',
        'libero_object': 'lerobot/libero_object',
        'libero_goal': 'lerobot/libero_goal',
        'libero_long': 'lerobot/libero_long',
    }
    dataset_id = dataset_map.get(benchmark, 'lerobot/libero_spatial')

    print(f"  [DATA] Loading {dataset_id} from LeRobot...")

    try:
        full_dataset = LeRobotDataset(dataset_id)
    except Exception as e:
        print(f"  [WARN] Could not load {dataset_id}: {e}")
        print(f"  [WARN] Falling back to synthetic dataset")
        from training.data_loader import create_dataloaders
        return create_dataloaders(config, seed=seed)

    # Split into train/val
    total = len(full_dataset)
    n_train_total = min(int(0.8 * total), n_train * 10)  # 10 tasks
    n_val_total = min(total - n_train_total, n_val * 10)

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [n_train_total, n_val_total],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    val_loaders = {
        'combined': torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
    }

    print(f"  [DATA] Train: {n_train_total}, Val: {n_val_total}")
    return train_loader, val_loaders


def _create_metaworld_dataloaders(config, seed):
    """Load MetaWorld MT-10 demonstrations."""
    try:
        import metaworld
    except ImportError:
        print("  [WARN] MetaWorld not installed. Falling back to synthetic.")
        from training.data_loader import create_dataloaders
        return create_dataloaders(config, seed=seed)

    # MetaWorld data loading would go here
    # For now, fallback to synthetic
    print("  [WARN] MetaWorld loader not yet implemented. Using synthetic.")
    from training.data_loader import create_dataloaders
    return create_dataloaders(config, seed=seed)


# ---------------------------------------------------------------------------
# Model Construction
# ---------------------------------------------------------------------------

def build_smolvla_model(config, device, head_type='flow'):
    """
    Build the full SmolVLA + SE(3) head model.

    Args:
        config: full config dict
        device: torch device
        head_type: 'flow' | 'chunk' | 'uncertainty'

    Returns:
        SmolVLAAdapter model
    """
    model_cfg = config.get('model', {})
    hidden_dim = model_cfg.get('hidden_dim', 768)
    backbone_path = model_cfg.get('backbone_path', 'lerobot/smolvla_base')
    freeze_backbone = model_cfg.get('freeze_backbone', True)

    # Load SmolVLA backbone
    print(f"  [MODEL] Loading SmolVLA backbone from '{backbone_path}'...")
    try:
        backbone = SmolVLABackbone.from_pretrained(
            backbone_path,
            device=str(device),
            freeze=freeze_backbone,
        )
    except Exception as e:
        print(f"  [WARN] Could not load pretrained SmolVLA: {e}")
        print(f"  [WARN] Using mock backbone for smoke testing")
        from models.mock_backbone import MockOctoBackbone
        backbone = MockOctoBackbone(hidden_dim=hidden_dim).to(device)

    # Build action head based on type
    head_hidden_dim = model_cfg.get('head_hidden_dim', 256)
    n_layers = model_cfg.get('n_layers', 4)
    source_scale = model_cfg.get('source_scale', 0.1)

    if head_type == 'chunk':
        chunk_size = model_cfg.get('chunk_size', 8)
        n_anchors = model_cfg.get('n_anchors', 2)
        action_head = GeodesicChunkPredictor(
            hidden_dim=hidden_dim,
            chunk_size=chunk_size,
            n_anchors=n_anchors,
            head_hidden_dim=head_hidden_dim,
            n_layers=n_layers,
        )
        print(f"  [MODEL] Head: GeodesicChunkPredictor (K={n_anchors}, H={chunk_size})")

    elif head_type == 'uncertainty':
        n_samples = model_cfg.get('n_uncertainty_samples', 10)
        base_predictor = SE3ActionPredictor(
            hidden_dim=hidden_dim,
            head_hidden_dim=head_hidden_dim,
            n_layers=n_layers,
            source_scale=source_scale,
        )
        action_head = UncertaintyAwareFlowHead(
            base_predictor=base_predictor,
            n_samples=n_samples,
        )
        print(f"  [MODEL] Head: UncertaintyAwareFlowHead (N={n_samples} samples)")

    else:  # 'flow' (default)
        action_head = SE3ActionPredictor(
            hidden_dim=hidden_dim,
            head_hidden_dim=head_hidden_dim,
            n_layers=n_layers,
            source_scale=source_scale,
        )
        print(f"  [MODEL] Head: SE3ActionPredictor (flow matching)")

    # Assemble full model
    model = SmolVLAAdapter(
        backbone=backbone,
        action_head=action_head,
        head_type=head_type,
    ).to(device)

    return model


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def collate_smolvla_batch(batch):
    """
    Collate function for SmolVLA batches.

    Handles both synthetic and real dataset formats.
    """
    if isinstance(batch[0], dict):
        # Synthetic dataset format
        observations = {
            'image': torch.stack([b['observations']['image'] for b in batch]),
            'task_id': torch.stack([b['observations']['task_id'] for b in batch]),
        }
        language = torch.stack([b['language'] for b in batch])
        target_actions = torch.stack([b['target_actions'] for b in batch])
        target_gripper = torch.stack([b['target_gripper'] for b in batch])

        return {
            'observations': observations,
            'language': language,
            'target_actions': target_actions,
            'target_gripper': target_gripper,
        }
    else:
        # LeRobot dataset format (tuple)
        return batch


def train_one_epoch_smolvla(model, dataloader, optimizer, device, config, head_type='flow'):
    """Train for one epoch with SmolVLA."""
    model.train()
    total_loss = 0
    n_batches = 0
    loss_components = {}

    grad_clip = config.get('training', {}).get('gradient_clip_norm', 1.0)

    for batch in tqdm(dataloader, desc="Training", leave=False):
        # Unpack batch
        if isinstance(batch, dict):
            observations = {k: v.to(device) for k, v in batch['observations'].items()}
            language = batch['language'].to(device)
            target_actions = batch['target_actions'].to(device)
            target_gripper = batch.get('target_gripper')
            if target_gripper is not None:
                target_gripper = target_gripper.to(device)
        else:
            # Handle tuple format from LeRobot
            observations, language, target_actions = batch[0], batch[1], batch[2]
            if isinstance(observations, dict):
                observations = {k: v.to(device) for k, v in observations.items()}
            else:
                observations = observations.to(device)
            target_actions = target_actions.to(device)
            target_gripper = None

        optimizer.zero_grad()

        # Forward + loss
        loss, loss_dict = model.compute_loss(
            observations, language, target_actions, target_gripper
        )

        loss.backward()

        # Gradient clipping on trainable params only
        trainable_params = model.trainable_parameters()
        if trainable_params:
            torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)

        optimizer.step()

        total_loss += loss_dict.get('total_loss', loss.item())
        n_batches += 1

        for k, v in loss_dict.items():
            if k not in loss_components:
                loss_components[k] = 0
            loss_components[k] += v

    avg_loss = total_loss / max(n_batches, 1)
    avg_components = {k: v / max(n_batches, 1) for k, v in loss_components.items()}

    return avg_loss, avg_components


@torch.no_grad()
def validate_smolvla(model, dataloader, device, head_type='flow'):
    """Validate on a single dataloader."""
    model.eval()
    all_pred = []
    all_target = []

    for batch in tqdm(dataloader, desc="Validating", leave=False):
        if isinstance(batch, dict):
            observations = {k: v.to(device) for k, v in batch['observations'].items()}
            language = batch['language'].to(device)
            target_actions = batch['target_actions'].to(device)
        else:
            observations, language, target_actions = batch[0], batch[1], batch[2]
            if isinstance(observations, dict):
                observations = {k: v.to(device) for k, v in observations.items()}
            else:
                observations = observations.to(device)
            target_actions = target_actions.to(device)

        h = model.encode(observations, language)

        if head_type == 'uncertainty':
            mean_action, gripper, variance, samples = \
                model.action_head.predict_with_uncertainty(h, n_steps=10)
            pred_actions = mean_action
        elif head_type == 'chunk':
            pred_chunk, pred_gripper = model.action_head(h)
            pred_actions = pred_chunk[:, 0]  # use first action for metrics
        else:
            pred_actions, gripper = model.action_head.predict(h, n_steps=10)

        all_pred.append(pred_actions)
        all_target.append(target_actions)

    all_pred = torch.cat(all_pred, dim=0)
    all_target = torch.cat(all_target, dim=0)

    return {
        'geodesic_rmse': geodesic_rmse(all_pred, all_target).item(),
        'rotation_rmse': rotation_rmse(all_pred, all_target).item(),
        'translation_rmse': translation_rmse(all_pred, all_target).item(),
    }


def validate_all_families_smolvla(model, val_loaders, device, head_type='flow'):
    """Validate on all family loaders."""
    results = {}
    for family_name, loader in val_loaders.items():
        results[family_name] = validate_smolvla(model, loader, device, head_type)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train SmolVLA + SE(3) Action Head")
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--head-type', type=str, default='flow',
                        choices=['flow', 'chunk', 'uncertainty'],
                        help='Action head type')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--run-tag', type=str, default=None,
                        help='Custom tag for checkpoint filenames')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu/mps)')
    args = parser.parse_args()

    config = load_config(args.config)

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    seed_everything(args.seed)

    head_type = args.head_type
    run_tag = args.run_tag or f"smolvla_{head_type}_seed{args.seed}"

    print("=" * 70)
    print(f"  SmolVLA + SE(3) Training")
    print(f"  Config:    {args.config}")
    print(f"  Head type: {head_type}")
    print(f"  Seed:      {args.seed}")
    print(f"  Device:    {device}")
    print(f"  Run tag:   {run_tag}")
    print("=" * 70)

    # ---- Build model ----
    print("\n[1/4] Building model...")
    model = build_smolvla_model(config, device, head_type=head_type)
    param_counts = model.count_parameters()
    print(f"  Total params:     {param_counts['total']:,}")
    print(f"  Trainable params: {param_counts['trainable']:,}")
    print(f"  Frozen params:    {param_counts['frozen']:,}")

    # ---- Create dataloaders ----
    print("\n[2/4] Creating dataloaders...")
    train_loader, val_loaders = create_smolvla_dataloaders(config, seed=args.seed)
    print(f"  Train samples: {len(train_loader.dataset)}")
    for name, loader in val_loaders.items():
        print(f"  Val [{name}]: {len(loader.dataset)}")

    # ---- Optimizer ----
    print("\n[3/4] Setting up optimizer...")
    trainable_params = model.trainable_parameters()
    lr = config.get('training', {}).get('learning_rate', 1e-4)
    wd = config.get('training', {}).get('weight_decay', 1e-5)

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=wd)

    n_epochs = args.epochs or config.get('training', {}).get('n_epochs', 50)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    print(f"  Learning rate: {lr}")
    print(f"  Weight decay:  {wd}")
    print(f"  Epochs:        {n_epochs}")

    # ---- Resume ----
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        print(f"\n  Resuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        print(f"  Resumed at epoch {start_epoch}")

    # ---- Checkpoint dir ----
    ckpt_dir = config.get('logging', {}).get('checkpoint_dir', 'checkpoints/smolvla_se3/')
    os.makedirs(ckpt_dir, exist_ok=True)

    # ---- Training loop ----
    print(f"\n[4/4] Starting training ({n_epochs} epochs)...\n")
    best_val_metric = float('inf')
    history = []
    start_time = time.time()

    for epoch in range(start_epoch, n_epochs):
        epoch_start = time.time()

        # Train
        train_loss, train_components = train_one_epoch_smolvla(
            model, train_loader, optimizer, device, config, head_type=head_type
        )
        scheduler.step()

        # Validate
        family_metrics = validate_all_families_smolvla(
            model, val_loaders, device, head_type=head_type
        )

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']

        # Combined metrics
        cm = family_metrics.get('combined', family_metrics.get(
            list(family_metrics.keys())[0], {}))

        print(
            f"Epoch {epoch+1:3d}/{n_epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"G-RMSE: {cm.get('geodesic_rmse', 0):.4f} | "
            f"R-RMSE: {cm.get('rotation_rmse', 0):.4f} | "
            f"T-RMSE: {cm.get('translation_rmse', 0):.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {epoch_time:.1f}s"
        )

        # History
        record = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'lr': current_lr,
            'epoch_time': epoch_time,
            'per_family': family_metrics,
            **{f'train_{k}': v for k, v in train_components.items()},
        }
        history.append(record)

        # Periodic checkpoint
        save_interval = config.get('logging', {}).get('save_interval', 10)
        if (epoch + 1) % save_interval == 0 or (epoch + 1) == n_epochs:
            ckpt_path = os.path.join(ckpt_dir, f"{run_tag}_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'per_family_metrics': family_metrics,
                'config': config,
                'seed': args.seed,
                'head_type': head_type,
            }, ckpt_path)
            print(f"  → Saved: {ckpt_path}")

        # Best checkpoint
        combined_grmse = cm.get('geodesic_rmse', float('inf'))
        if combined_grmse < best_val_metric:
            best_val_metric = combined_grmse
            best_path = os.path.join(ckpt_dir, f"{run_tag}_best.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'per_family_metrics': family_metrics,
                'config': config,
                'seed': args.seed,
                'head_type': head_type,
            }, best_path)

    # ---- Summary ----
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"  Training complete!")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Best combined geodesic RMSE: {best_val_metric:.4f}")
    print(f"  Best checkpoint: {ckpt_dir}/{run_tag}_best.pt")
    print("=" * 70)

    # Save history
    history_path = os.path.join(ckpt_dir, f"{run_tag}_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"  History: {history_path}")

    # Save final results summary
    results_path = os.path.join(ckpt_dir, f"{run_tag}_results.json")
    with open(results_path, 'w') as f:
        json.dump({
            'config': args.config,
            'seed': args.seed,
            'head_type': head_type,
            'n_epochs': n_epochs,
            'best_epoch': history[-1]['epoch'] if history else 0,
            'best_geodesic_rmse': best_val_metric,
            'total_time_seconds': total_time,
            'param_counts': param_counts,
            'final_metrics': family_metrics,
        }, f, indent=2)
    print(f"  Results: {results_path}")


if __name__ == '__main__':
    main()
