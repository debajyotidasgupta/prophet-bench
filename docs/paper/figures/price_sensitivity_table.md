# Price-sensitivity ablation

| Perturbation | Spearman ρ | Kendall τ | N |
|---|---|---|---|
| V × 2 | 0.997 | 0.983 | 16 |
| C × 2 (higher penalty) | 0.962 | 0.883 | 16 |
| κ × 2 (heavier calib weight) | 0.971 | 0.883 | 16 |
| κ × 0.5 (lighter calib) | 0.997 | 0.983 | 16 |
| δ_pass × 2 | 1.000 | 1.000 | 16 |
| δ_pass × 0.5 | 1.000 | 1.000 | 16 |
| asymmetric: C × 3 | 0.950 | 0.867 | 16 |