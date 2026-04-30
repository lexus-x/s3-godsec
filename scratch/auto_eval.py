import os
import glob
import time
import subprocess
import json
import numpy as np

# 1. Wait for sweep to finish
while True:
    with open("logs/phase_a.log", "r") as f:
        log_content = f.read()
    if "Phase A sweep complete" in log_content:
        break
    time.sleep(10)

print("Sweep finished! Starting evaluation...")

# 2. Evaluate
configs = [
    "phase_a_baseline", "phase_a_scale001", "phase_a_scale05", 
    "phase_a_scale10", "phase_a_steps04", "phase_a_steps20", "phase_a_steps50"
]
seeds = [0, 1, 2]

os.makedirs("results", exist_ok=True)

for cfg in configs:
    for seed in seeds:
        tag = f"{cfg}_seed{seed}"
        ckpt = f"checkpoints/{tag}_best.pt"
        out = f"results/{tag}_eval.json"
        
        if not os.path.exists(out):
            print(f"Evaluating {tag}...")
            subprocess.run([
                "python", "src/evaluate.py",
                "--config", f"experiments/phase_a/{cfg}.yaml",
                "--checkpoint", ckpt,
                "--seed", str(seed),
                "--output", out
            ], check=True, env=dict(os.environ, MKL_SERVICE_FORCE_INTEL="1", MKL_THREADING_LAYER="GNU"))

print("Evaluation complete! Generating report...")

# 3. Parse results
results_data = {}
for cfg in configs:
    t_rmses = []
    r_rmses = []
    for seed in seeds:
        with open(f"results/{cfg}_seed{seed}_eval.json", "r") as f:
            data = json.load(f)
        t_rmses.append(data["translation_heavy"]["translation_rmse"])
        r_rmses.append(data["translation_heavy"]["rotation_rmse"])
    
    results_data[cfg] = {
        "t_rmse_mean": np.mean(t_rmses),
        "t_rmse_std": np.std(t_rmses, ddof=1), # sample std
        "r_rmse_mean": np.mean(r_rmses)
    }

# 4. Compute deltas
baseline_t = results_data["phase_a_baseline"]["t_rmse_mean"]
baseline_r = results_data["phase_a_baseline"]["r_rmse_mean"]

for cfg in configs:
    results_data[cfg]["delta_t"] = results_data[cfg]["t_rmse_mean"] - baseline_t

# 5. Find best config
best_config = min(configs, key=lambda c: results_data[c]["t_rmse_mean"])
best_t = results_data[best_config]["t_rmse_mean"]
best_delta = results_data[best_config]["delta_t"]
best_r = results_data[best_config]["r_rmse_mean"]

r_worsen_percent = (best_r - baseline_r) / baseline_r * 100

# 6. Conclusion
if best_t < baseline_t and r_worsen_percent <= 10:
    conclusion = f"T-RMSE is HP-fixable. The config `{best_config}` achieved a lower T-RMSE ({best_t:.4f}, delta: {best_delta:.4f}) without worsening R-RMSE by more than 10% (worsened by {r_worsen_percent:.1f}%)."
elif best_t < baseline_t and r_worsen_percent > 10:
    conclusion = f"T-RMSE is fundamental. Although `{best_config}` improved T-RMSE by {best_delta:.4f}, it severely degraded R-RMSE by {r_worsen_percent:.1f}%, indicating a fundamental trade-off."
else:
    conclusion = f"T-RMSE is fundamental. No configuration significantly improved T-RMSE over the baseline without severe trade-offs. Best was `{best_config}` with delta {best_delta:.4f}."

# 7. Write Markdown
md = ["# Phase A: T-RMSE Investigation\n"]
md.append("### Translation-Heavy Family Results\n")
md.append("| Config | Mean T-RMSE | Std T-RMSE | Mean R-RMSE | Δ vs Baseline T-RMSE |")
md.append("|---|---|---|---|---|")

for cfg in configs:
    d = results_data[cfg]
    md.append(f"| `{cfg}` | {d['t_rmse_mean']:.4f} | {d['t_rmse_std']:.4f} | {d['r_rmse_mean']:.4f} | {d['delta_t']:.4f} |")

md.append(f"\n**Best Config:** `{best_config}` (Δ T-RMSE: {best_delta:.4f})")
md.append(f"\n**Crucial Check:** Baseline R-RMSE was {baseline_r:.4f}. Best config R-RMSE is {best_r:.4f} (change: {r_worsen_percent:.1f}%).")
md.append(f"\n### Conclusion\n{conclusion}\n")

with open("reports/T_RMSE_INVESTIGATION.md", "w") as f:
    f.write("\n".join(md))

print("Report generated at reports/T_RMSE_INVESTIGATION.md")
