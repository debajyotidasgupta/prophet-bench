# MGSM cross-correlation

N agents = 10

| agent | MGSM acc | PROPHET acc | PROPHET ECE | PROPHET payoff/task |
|---|---|---|---|---|
| openai-gpt-5.2 | 0.967 | 1.000 | 0.010 | 127.8 |
| anthropic-claude-opus-4.7 | 0.958 | 1.000 | 0.024 | 123.2 |
| google-gemini-3-flash-preview | 0.958 | 1.000 | 0.000 | 89.7 |
| google-gemini-3.1-pro-preview | 0.958 | 1.000 | 0.000 | 4.8 |
| openai-gpt-5-mini | 0.950 | 1.000 | 0.006 | 118.8 |
| anthropic-claude-haiku-4.5 | 0.925 | 1.000 | 0.018 | 89.7 |
| meta-llama-llama-4-maverick | 0.925 | 1.000 | 0.009 | 103.1 |
| openai-gpt-5 | 0.925 | 0.950 | 0.022 | 81.7 |
| openai-gpt-5-nano | 0.808 | 1.000 | 0.032 | 89.6 |
| deepseek-deepseek-v3.2 | 0.542 | 1.000 | 0.021 | 132.1 |

## Correlations

- spearman_mgsm_vs_prophet_acc = 0.261
- pearson_mgsm_vs_prophet_acc = -0.089
- spearman_mgsm_vs_prophet_ece = -0.539
- pearson_mgsm_vs_prophet_ece = -0.434
- spearman_mgsm_vs_prophet_payoff = -0.042
- pearson_mgsm_vs_prophet_payoff = -0.310
- n_agents = 10