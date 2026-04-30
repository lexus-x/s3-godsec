# Phase A: T-RMSE Investigation

### Translation-Heavy Family Results

| Config | Mean T-RMSE | Std T-RMSE | Mean R-RMSE | Δ vs Baseline T-RMSE |
|---|---|---|---|---|
| `phase_a_baseline` | 0.4241 | 0.0514 | 0.2407 | 0.0000 |
| `phase_a_scale001` | 0.4129 | 0.0231 | 0.1983 | -0.0111 |
| `phase_a_scale05` | 0.4275 | 0.0120 | 0.2088 | 0.0034 |
| `phase_a_scale10` | 0.4124 | 0.0137 | 0.1999 | -0.0116 |
| `phase_a_steps04` | 0.4226 | 0.0443 | 0.2390 | -0.0015 |
| `phase_a_steps20` | 0.4325 | 0.0525 | 0.2394 | 0.0084 |
| `phase_a_steps50` | 0.4228 | 0.0431 | 0.2365 | -0.0013 |

**Best Config:** `phase_a_scale10` (Δ T-RMSE: -0.0116)

**Crucial Check:** Baseline R-RMSE was 0.2407. Best config R-RMSE is 0.1999 (change: -17.0%).

### Conclusion
T-RMSE is HP-fixable. The config `phase_a_scale10` achieved a lower T-RMSE (0.4124, delta: -0.0116) without worsening R-RMSE by more than 10% (worsened by -17.0%).
