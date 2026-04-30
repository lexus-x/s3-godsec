import json
import glob
import numpy as np

data = {}
for filepath in glob.glob("/home/user/Desktop/vla_projects/q1/s3-godsec/results/*_eval.json"):
    filename = filepath.split("/")[-1]
    name_parts = filename.replace("_eval.json", "").split("_")
    head = name_parts[0]
    backbone = "_".join(name_parts[1:-1])
    seed = name_parts[-1].replace("seed", "")
    
    with open(filepath, "r") as f:
        data[(head, backbone, seed)] = json.load(f)

print("--- Per-seed table for primary claim ---")
print("| Backbone | Seed | Euclid R-RMSE | SE(3) R-RMSE | Δ (Eucl - SE(3)) | SE(3) ≤ Eucl? |")
print("|----------|------|---------------|--------------|------------------|---------------|")
for backbone in ["scene_id", "mock_cnn"]:
    for seed in ["0", "1", "2"]:
        eucl = data[("OctoEuclideanBaseline", backbone, seed)]["rotation_heavy"]["rotation_rmse"]
        se3 = data[("OctoSE3", backbone, seed)]["rotation_heavy"]["rotation_rmse"]
        delta = eucl - se3
        yesno = "YES" if se3 <= eucl else "NO"
        print(f"| {backbone} | {seed} | {eucl:.4f} | {se3:.4f} | {delta:.4f} | {yesno} |")

print("\n--- Aggregate table ---")
print("| Head | Backbone | Family | G-RMSE | R-RMSE | T-RMSE |")
print("|------|----------|--------|--------|--------|--------|")
for head in ["OctoEuclideanBaseline", "OctoSE3"]:
    for backbone in ["mock_cnn", "scene_id"]:
        for family in ["rotation_heavy", "translation_heavy", "combined"]:
            g_rmse = [data[(head, backbone, seed)][family]["geodesic_rmse"] for seed in ["0", "1", "2"]]
            r_rmse = [data[(head, backbone, seed)][family]["rotation_rmse"] for seed in ["0", "1", "2"]]
            t_rmse = [data[(head, backbone, seed)][family]["translation_rmse"] for seed in ["0", "1", "2"]]
            print(f"| {head} | {backbone} | {family} | {np.mean(g_rmse):.4f} ± {np.std(g_rmse):.4f} | {np.mean(r_rmse):.4f} ± {np.std(r_rmse):.4f} | {np.mean(t_rmse):.4f} ± {np.std(t_rmse):.4f} |")

print("\n--- Honest negative finding ---")
print("| Backbone | Seed | Euclid T-RMSE | SE(3) T-RMSE | Δ (Eucl - SE(3)) |")
print("|----------|------|---------------|--------------|------------------|")
for backbone in ["scene_id", "mock_cnn"]:
    for seed in ["0", "1", "2"]:
        eucl = data[("OctoEuclideanBaseline", backbone, seed)]["translation_heavy"]["translation_rmse"]
        se3 = data[("OctoSE3", backbone, seed)]["translation_heavy"]["translation_rmse"]
        delta = eucl - se3
        print(f"| {backbone} | {seed} | {eucl:.4f} | {se3:.4f} | {delta:.4f} |")
