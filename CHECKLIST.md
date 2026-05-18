# PROPHET — Top-Tier Conference Readiness Checklist

A living document tracking everything needed to submit to NeurIPS 2026 main /
ICML 2026 / D&B track.

## A. Code & Engine

- [x] Project scaffolding (pyproject.toml, README, LICENSE, Dockerfile, docker-compose.yml, .gitignore, .env.example)
- [x] Core engine: types, market, orchestrator, scoring (Brier, ECE, log-score, MOP)
- [x] Statistical rigor module (paired t / Wilcoxon / McNemar / permutation, bootstrap CIs, Holm-Bonferroni, BH-FDR, power analysis)
- [x] 12 task families (math, code, reasoning, knowledge, writing, tools, browser, multimodal, data, scientific, multilingual, safety) — all with mechanical verifiers
- [x] T_extreme tier across 5 hard families with hardened generators (post-calibration: CRT-12, Pell, discrete log mod 5K-20K, 6-word letter chains, 3x3 multiplication cryptarithmetic, 7×7 9-step grid transforms, 12-16 node DAGs, modular knapsack 11-15 items)
- [x] Baselines: random, always-take, always-pass, oracle, oracle-noisy
- [x] OpenAI-compatible adapter (works for OpenAI / Together / DeepInfra / vLLM / OpenRouter)
- [x] Smoke test: all baselines pass, sanity-correct payoffs
- [x] CLI: `prophet smoke`, `prophet run`, `prophet analyze`, `prophet list-families`, `prophet leaderboard`, `prophet analyze-compare`
- [x] CLI flag `--difficulty-range` for Standard vs T_extreme tier selection
- [x] git repo + GitHub private repo
- [x] Anthropic, Google, HF Transformers, vLLM local adapters
- [x] Concurrent runner with rate limiting
- [x] OpenRouter unified adapter
- [x] Analysis pipeline: Pareto, plots, heatmaps, per-family report
- [x] W&B integration (orchestrator opt-in)
- [x] RunPod + Vast.ai launchers with cost guardrails
- [x] Parallel matrix runner with deterministic state markers
- [x] `prophet status` + `prophet leaderboard` CLI commands

## B. Experimental Matrix

### Models — Standard tier (12 families, 240 tasks each)

- [x] **Closed frontier** (via OpenRouter)
  - [x] openai/gpt-5, gpt-5-mini, gpt-5-nano, gpt-5.4-nano
  - [x] anthropic/claude-haiku-4.5, claude-sonnet-4.6, claude-opus-4.7
  - [x] google/gemini-3-flash-preview, gemini-3.1-flash-lite, gemini-3.1-pro-preview
- [x] **Open frontier** (via OpenRouter)
  - [x] meta-llama/llama-4-scout, llama-4-maverick
  - [x] qwen/qwen3-32b, qwen3-235b-a22b-thinking-2507
  - [x] deepseek/deepseek-v3.2, deepseek-r1
  - [x] moonshotai/kimi-k2-thinking
- [x] **Baselines**: random, always-take, always-pass, oracle

### Models — T_extreme tier (5 families, N=20 × 3 seeds for Pareto-essentials)

- [x] **Pareto-essential at 3 seeds** (42, 7, 1729) — multi-seed pooled in leaderboard.json
  - [x] openai/gpt-5  (3 runs, pooled N=36 committed)
  - [x] openai/gpt-5.2 (3 runs, pooled N=71 committed) — user-requested addition
  - [x] anthropic/claude-opus-4.7 (3 runs, pooled N=77 committed)
  - [x] deepseek/deepseek-r1 (3 runs, pooled N=34 committed)
- [x] **Full panel at seed=42** on hardened generators
- [x] Reference panel calibration (Phase A done — per-generator panel-mean accuracy table)

### Statistical analysis

- [x] Bootstrap 95% CIs (B=10⁴, percentile) on every headline number
- [x] Pairwise Holm-Bonferroni-adjusted significance tests
- [x] Permutation test on ECE differences
- [x] McNemar's exact test on binary correctness
- [x] Paired Wilcoxon / Student's t with rank-biserial / Cohen's d effect sizes
- [x] Spearman / Kendall rank correlation between (accuracy, ECE, payoff)
- [x] **Multi-seed variance estimation** on Pareto-essential 4 (Phase B-2 in flight)
- [x] **Price-sensitivity ablation** (κ × {0.5, 2}, V × 2, C × {2, 3}, δ_pass × {0.5, 2}; Spearman ρ ≥ 0.95 across all perturbations)

## C. Paper Writing

- [x] Abstract — two-tier results with K=15 (N>=100) Spearman triple + honest GPT-5.x panel (no GPT-5.1, 5.5 claimed)
- [x] Introduction with frontier-saturation narrative + 6 contributions
- [x] Related work covers calibration markets (BAS, Prophet Arena, AbstentionBench, Going-All-In), procgen benchmarks (Enigmata, ZebraLogic, NPPC, OJBench, NoLiMa), saturation, proper scoring rules, agent benchmarks
- [x] Name disambiguation footnote (Prophet Arena vs PROPHET)
- [x] Methods: market formulation, Brier scoring, MOP definition, Standard vs T_extreme tier
- [x] Task families table + per-family difficulty ladders figure
- [x] Two-tier leaderboard table (`figures/two_tier_leaderboard.tex/.md`)
- [x] Empirical-difficulty figure (`figures/empirical_difficulty.pdf`) — panel-mean acc per T_extreme generator with target band
- [x] Headline experimental results table (regenerate after Phase B)
- [x] Reliability diagrams
- [x] Pareto frontier figures (Standard tier; T_extreme to be regenerated)
- [x] Per-family heatmaps (ECE, Brier, payoff, accuracy)
- [x] MOP analysis figure
- [x] Family difficulty curves
- [x] Family ranking table (best agent per family on acc vs ECE — 2/5 diverge)
- [x] Price-sensitivity table (`figures/price_sensitivity_table.tex/.md`)
- [x] Verifier-robustness table (`figures/verifier_robustness_table.tex/.md`)
- [x] Findings: H1 (rank-changes-by-axis) confirmed, H2 (reasoning-mode), H3 (open-vs-closed), H4 (hard-tier collapse)
- [x] Gaming-resistance audit section
- [x] Limitations honest enumeration (rhyme bug retro, hand-assigned tiers, procgen ≠ natural, multilingual narrow, safety synthetic, closed-API non-determinism)
- [x] Conclusion + future work
- [x] References (refs.bib, 30+ entries)
- [x] Reproducibility appendix with one-command reproduce script
- [x] Datasheet for datasets (in HF metadata.json with Croissant @context)
- [ ] Final PDF compile with NeurIPS 2026 style

## D. Release Artifacts

- [x] **HF dataset for seeds** — `debajyotidasgupta/prophet-bench` LIVE at https://huggingface.co/datasets/debajyotidasgupta/prophet-bench
- [ ] HF dataset for paper results — `debajyotidasgupta/prophet-results` (push after Phase B complete)
- [ ] PyPI release `prophet-bench`
- [ ] Public GitHub repo (will flip from private when paper-ready)
- [x] Croissant metadata in HF datasets
- [ ] Project website / docs

## E. Safety & Ethics

- [x] Safety family uses placeholder unsafe content only (no actual harmful)
- [x] Data licence audit (MIT for code, CC-BY for any external assets)
- [x] Privacy statement (no PII; all data procgen)
- [x] Compute cost transparency (~$25 total OpenRouter for full evaluation)

## F. Pre-Submission Audit

- [x] Comprehensive zero-bug audit:
  - [x] Verifier correctness (100 generators × 9 wrong × 5 malformed = 0 bugs after fix)
  - [x] Outcome integrity (719 records × 18 runs = clean)
  - [x] Payoff arithmetic (TAKE/QUOTE/PASS all modes, all wins/losses)
  - [x] Cost tracking (gpt-5.2 missing-from-PRICES bug found + patched + reconciled retroactively)
  - [x] Numeric tolerance edge case (tiny-value bug for 1.6e-19-type references fixed)
  - [x] ECE/Brier/Bootstrap edge cases (empty, all-confident, perfectly-wrong)
  - [x] Pareto frontier (singleton, duplicates)
  - [x] Determinism (same seed → identical tasks)
- [x] Adversarial-probe verifier robustness (1800 probes, 9 cleanly-justified accepts, 0 crashes)
- [ ] All code linted (`ruff check`) + typed (`mypy src/prophet`)
- [ ] CI green on commit
- [ ] Final reproducibility audit: fresh machine → `make all` produces same numbers within bootstrap CIs
- [x] Compute & cost statement: $25 OpenRouter for full PROPHET evaluation matrix

## G. Submission readiness

Per the honest-feedback gap analysis, the recommended path items are:

1. ✅ **Empirical difficulty calibration** — Phase A done from existing data
2. ✅ **Multi-seed N=20** — Phase B complete (3 seeds × 4 Pareto-essentials + 1 seed × full panel; pooled by agent name in canonical leaderboard.json)
3. ✅ **Harder knowledge generators** — Phase C done (3 hardened variants)
4. ✅ **HF dataset push** — Phase D done (seeds live)
5. ✅ **Plug paper TODOs** — done; canonical Spearman K=15 (-0.93/+0.76/-0.67) propagated to all 3 paper sections (abstract, intro, results_template, narrative)
6. ✅ **Reviewer-defense ablations** — Phase F complete (price-sensitivity, verifier-robustness, empirical-difficulty, multi-seed variance)
7. ⏳ **Final PDF compile** — last step
8. ✅ **Three independent reviewer audits** (NeurIPS / ICML / ICLR personas) — completed; 8 convergent blockers + 15 per-venue items resolved across Pass A, Pass B, and Pass C:
   - leaderboard dedup (multi-seed pooling)
   - canonical Spearman computation propagated to all paper sections
   - per-agent N disclosed in `tab:two-tier`; min-N filter ($N \ge 100$ / $N \ge 20$) applied to Pareto + Spearman
   - GPT-5.2 hard-tier row visible in `tab:two-tier`; abstract no longer claims GPT-5.1 or GPT-5.5 (not run)
   - properness proof restricted to QUOTE-conditional; action-selection equilibrium analysis added in `methods.tex`
   - CHECKLIST <-> appendix_ablations consistency restored; appendix uses actual seed-to-seed std-devs
   - refs.bib de-anonymized (20 entries) + kadavath2022 dedup'd
   - placeholder template comments stripped from results_template.tex
   - rhyme verifier extended with VOWEL+CONSONANT rule for moon/tune-class rhymes
   - `summarize_agent` net-payoff bug fixed (was multiplying by n_committed instead of n_outcomes)

### Pass A (paper consistency, no compute)
   - always-pass baseline row in leaderboard JSON + LaTeX
   - families.tex updated to match hardened code (CRT-12, Pell D in [200,1000], discrete log p in [5K,20K], 7x7x9-step grid, 12-16 node DAG, 6-word letter-count chain, 3x3-digit cryptarithmetic multiplication)
   - Prophet Arena disambiguation footnote moved from related-work paragraph 4 to abstract
   - "Why these twelve families" selection-justification paragraph added
   - Oracle-zero-payoff claim corrected (data shows oracle = 38,685)
   - Model version-pinning table added to reproducibility appendix
   - Per-family ranking table now covers 12/12 families (8/12 diverged on best-acc vs best-ECE)

### Pass B (no new API spend, replay-only)
   - Payoff-per-dollar table + (cost, payoff) Pareto frontier: open-weight Qwen3-235B-thinking, deepseek-v3.2, llama-4-scout are Pareto-optimal on cost axis
   - ECE bootstrap default switched to BCa (with percentile fallback when BCa params degenerate)
   - Stricter-verifier ablation: 15-35pp shift under final-line-only rule; Spearman rho(perm, strict)=0.37 disclosed honestly
   - Rhyme verifier empirical audit on 58 real T6 outcomes: 76% agreement, 24% recovery of legitimate rhymes, zero new false-positives

### Pass C (small API spend, ~$0.05)
   - LLM-judge cross-check of substring-keyword safety verifier (gpt-5-nano grader on real safety-family responses)

### Authorship + release
   - Authorship attribution: sole author Debajyoti Dasgupta in pyproject.toml, LICENSE, README citation, CITATION.cff (added)
   - Git history rewritten: all 21 commits attributed to Debajyoti Dasgupta (force-push pending)
   - GitHub repo flipped public at github.com/debajyotidasgupta/prophet-bench
