"""
Training script for SE(3)-VLA.

Usage:
    python src/train.py --config configs/octo_se3.yaml
"""

import argparse
import yaml
import torch
import os
from tqdm import tqdm

from models.octo_adapter import OctoSE3, OctoEuclideanBaseline
from models.geodesic_loss import GeodesicMSELoss
from utils.metrics import geodesic_rmse, rotation_rmse


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def train_one_epoch(model, dataloader, optimizer, scheduler, device, config):
    model.train()
    total_loss = 0
    n_batches = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        observations = {k: v.to(device) for k, v in batch['observations'].items()}
        language = batch['language']
        target_actions = batch['target_actions'].to(device)
        target_gripper = batch.get('target_gripper')
        if target_gripper is not None:
            target_gripper = target_gripper.to(device)
        
        optimizer.zero_grad()
        
        loss, loss_dict = model.compute_loss(
            observations, language, target_actions, target_gripper
        )
        
        loss.backward()
        
        # Gradient clipping
        grad_clip = config['training'].get('gradient_clip_norm', 1.0)
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), grad_clip)
        
        optimizer.step()
        
        total_loss += loss_dict['total_loss']
        n_batches += 1
    
    if scheduler is not None:
        scheduler.step()
    
    return total_loss / n_batches


def evaluate(model, dataloader, device):
    model.eval()
    all_pred = []
    all_target = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            observations = {k: v.to(device) for k, v in batch['observations'].items()}
            language = batch['language']
            target_actions = batch['target_actions'].to(device)
            
            h = model.encode(observations, language)
            pred_actions, _ = model.action_predictor.predict(h, n_steps=10)
            
            all_pred.append(pred_actions)
            all_target.append(target_actions)
    
    all_pred = torch.cat(all_pred, dim=0)
    all_target = torch.cat(all_target, dim=0)
    
    g_rmse = geodesic_rmse(all_pred, all_target).item()
    r_rmse = rotation_rmse(all_pred, all_target).item()
    
    return {
        'geodesic_rmse': g_rmse,
        'rotation_rmse': r_rmse,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    
    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    # In practice, load the actual Octo model here
    # For now, use a placeholder
    print(f"Loading model: {config['model']['name']}")
    print(f"Device: {device}")
    
    # Placeholder: in real implementation, load Octo backbone
    # octo_model = load_octo(config['model']['octo_checkpoint'])
    # model = OctoSE3(octo_model, **{k: v for k, v in config['model'].items() if k not in ['name', 'octo_checkpoint']})
    # model = model.to(device)
    
    # optimizer = torch.optim.AdamW(
    #     model.trainable_parameters(),
    #     lr=config['training']['learning_rate'],
    #     weight_decay=config['training']['weight_decay'],
    # )
    
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=config['training']['n_epochs']
    # )
    
    print("Training configuration loaded successfully.")
    print(f"Config: {config}")
    
    # Training loop
    # for epoch in range(config['training']['n_epochs']):
    #     train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, config)
    #     print(f"Epoch {epoch+1}/{config['training']['n_epochs']}, Loss: {train_loss:.4f}")
    #     
    #     if (epoch + 1) % config['logging']['save_interval'] == 0:
    #         torch.save(model.state_dict(), f"checkpoints/epoch_{epoch+1}.pt")


if __name__ == '__main__':
    main()
