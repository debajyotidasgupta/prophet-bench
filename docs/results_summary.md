# PROPHET — Initial Results Summary

**Methodology**: 12 task families × 20 tasks each = 240 tasks per model, seed=42, market_seed=42.

**Cost**: ~$8 USD on OpenRouter across 13 paid agents + 4 baselines.

**Findings**:

1. **The first-place leader changes under every axis.**
   - Best by accuracy: **Google Gemini 3 Flash Preview** (97.0%)
   - Best by ECE: **Google Gemini 3 Flash Preview** (0.027)
   - Best by net payoff: **Anthropic Claude Opus 4.7** ($23,686 PROPHET-cents)

2. **Two Pareto-optimal points** on (ECE, net payoff):
   - **Gemini 3 Flash Preview**: highest accuracy + best ECE + 2nd-best payoff
   - **Claude Opus 4.7**: highest payoff + lower accuracy + moderate ECE

3. **Spearman rank correlations between axes are NOT 1.0**:
   - ρ(accuracy, ECE) = 0.89
   - ρ(accuracy, payoff) = 0.79
   - ρ(ECE, payoff) = 0.60 ← significant divergence

4. **Behavioral diversity in commitment style**:
   - DeepSeek-V3.2: nearly always QUOTE (205/240)
   - Claude Opus 4.7: TAKE-heavy with calibrated QUOTE
   - GPT-5 / Gemini-3.1-Pro-Preview: 100% commit (no PASS) — fully ambitious
   - Llama-4-Scout: 13.8% PASS rate (most cautious of mid-tier)

5. **Per-family heterogeneity**: 10/12 families have different leaders on accuracy vs ECE — calibration is family-specific, not a universal trait of the model.

6. **Open vs closed**: DeepSeek-V3.2 (#4 by payoff) and Llama-4-Maverick (#7) compete with closed frontier on payoff despite lower accuracy. Cost-per-quality favours open weights for risk-aware buyers.

**Statistical rigor**: Bootstrap 95% CIs (B=10⁴) on every metric. Pairwise tests (paired t / Wilcoxon / McNemar / ECE-permutation) with Holm-Bonferroni correction.

See `results/openrouter_matrix/leaderboard/leaderboard.md` for the up-to-date table.
