"""
Evaluation script for SE(3)-VLA.

Usage:
    python src/evaluate.py --config configs/octo_se3.yaml --checkpoint checkpoints/best.pt
"""

import argparse
import yaml
import torch
import json
import os

from utils.metrics import (
    geodesic_rmse,
    rotation_rmse,
    translation_rmse,
    geodesic_action_ece,
    success_rate_per_rotation_bin,
)


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def evaluate_all(model, dataloader, device, config):
    """Run all evaluation metrics."""
    model.eval()
    
    all_pred = []
    all_target = []
    all_successes = []
    all_uncertainties = []
    
    with torch.no_grad():
        for batch in dataloader:
            observations = {k: v.to(device) for k, v in batch['observations'].items()}
            language = batch['language']
            target_actions = batch['target_actions'].to(device)
            successes = batch['successes'].to(device)
            
            h = model.encode(observations, language)
            pred_actions, gripper = model.action_predictor.predict(h, n_steps=10)
            
            # Compute uncertainty (geodesic distance between multiple samples)
            pred_actions_2, _ = model.action_predictor.predict(h, n_steps=10)
            uncertainty = geodesic_distance(pred_actions, pred_actions_2)
            
            all_pred.append(pred_actions)
            all_target.append(target_actions)
            all_successes.append(successes)
            all_uncertainties.append(uncertainty)
    
    all_pred = torch.cat(all_pred, dim=0)
    all_target = torch.cat(all_target, dim=0)
    all_successes = torch.cat(all_successes, dim=0)
    all_uncertainties = torch.cat(all_uncertainties, dim=0)
    
    results = {}
    
    # Geodesic RMSE
    results['geodesic_rmse'] = geodesic_rmse(all_pred, all_target).item()
    results['rotation_rmse'] = rotation_rmse(all_pred, all_target).item()
    results['translation_rmse'] = translation_rmse(all_pred, all_target).item()
    
    # Geodesic Action-ECE
    ece, bin_data = geodesic_action_ece(all_uncertainties, all_successes)
    results['geodesic_action_ece'] = ece
    results['ece_bins'] = bin_data
    
    # Success rate per rotation bin
    bin_edges, success_rates, counts = success_rate_per_rotation_bin(
        all_pred, all_target, all_successes, n_bins=5
    )
    results['rotation_bins'] = {
        'edges': bin_edges.tolist(),
        'success_rates': success_rates.tolist(),
        'counts': counts.tolist(),
    }
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, default='results/evaluation.json')
    args = parser.parse_args()
    
    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    
    # In real implementation:
    # model = load_model(config, args.checkpoint, device)
    # results = evaluate_all(model, test_loader, device, config)
    # 
    # os.makedirs(os.path.dirname(args.output), exist_ok=True)
    # with open(args.output, 'w') as f:
    #     json.dump(results, f, indent=2)
    # 
    # print(f"Results saved to {args.output}")
    # print(f"Geodesic RMSE: {results['geodesic_rmse']:.4f}")
    # print(f"Rotation RMSE: {results['rotation_rmse']:.4f}")
    # print(f"Translation RMSE: {results['translation_rmse']:.4f}")
    # print(f"Geodesic Action-ECE: {results['geodesic_action_ece']:.4f}")
    
    print("Evaluation script ready. Implement model loading to run.")


if __name__ == '__main__':
    main()
