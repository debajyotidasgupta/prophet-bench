# PROPHET — Top-Tier Conference Readiness Checklist

A living document tracking everything needed to submit to NeurIPS 2026
Datasets & Benchmarks Track.

## A. Code & Engine

- [x] Project scaffolding (pyproject.toml, README, LICENSE, Dockerfile, docker-compose.yml, .gitignore, .env.example)
- [x] Core engine: types, market, orchestrator, scoring (Brier, ECE, log-score, MOP)
- [x] Statistical rigor module (paired t / Wilcoxon / McNemar / permutation, bootstrap CIs, Holm-Bonferroni, BH-FDR, power analysis)
- [x] Math family (8 difficulty tiers, mechanical verifier)
- [x] Baselines: random, always-take, always-pass, oracle, oracle-noisy
- [x] OpenAI-compatible adapter (works for OpenAI / Together / DeepInfra / vLLM / OpenRouter)
- [x] Smoke test: all baselines pass, sanity-correct payoffs
- [x] CLI: `prophet smoke`, `prophet run`, `prophet analyze`, `prophet list-families`
- [x] Tests passing (28+ tests; will grow as families land)
- [x] git repo + GitHub private repo (`debajyotidasgupta/prophet-bench`)
- [x] 11 additional task families implemented (12 total)
- [x] Anthropic, Google, HF Transformers, vLLM local adapters
- [x] Concurrent runner with rate limiting
- [x] OpenRouter unified adapter (all closed + most open in one provider)
- [x] Analysis pipeline: Pareto, plots, heatmaps, per-family report
- [x] W&B integration (orchestrator opt-in)
- [x] RunPod launcher with cost guardrails
- [x] Reproducibility script `scripts/run_full_eval.sh`
- [x] CI on GitHub Actions (test + lint + smoke)
- [x] Parallel matrix runner with deterministic state markers
- [x] `prophet status` + `prophet leaderboard` CLI commands
- [x] Auto narrative generator for paper

## B. Experimental Matrix

### Models (≥ 12 for headline)

- [x] **Closed frontier** (via OpenRouter)
  - [x] openrouter:openai/gpt-5-nano
  - [x] openrouter:openai/gpt-5.4-nano
  - [x] openrouter:openai/gpt-5-mini
  - [ ] openrouter:openai/gpt-5
  - [x] openrouter:anthropic/claude-haiku-4.5
  - [ ] openrouter:anthropic/claude-sonnet-4.6
  - [ ] openrouter:anthropic/claude-opus-4.7
  - [x] openrouter:google/gemini-3-flash-preview
  - [x] openrouter:google/gemini-3.1-flash-lite
  - [ ] openrouter:google/gemini-3.1-pro-preview
- [x] **Open frontier** (via OpenRouter)
  - [x] meta-llama/llama-4-scout
  - [ ] meta-llama/llama-4-maverick (running)
  - [ ] qwen/qwen3-32b (running)
  - [ ] qwen/qwen3-235b-a22b-thinking-2507 (running)
  - [ ] deepseek/deepseek-v3.2 (running)
- [ ] **Reasoning variants** (with/without thinking) — to-do
- [x] **Baselines**: random, always-take, always-pass, oracle

### Families × Models

- [ ] Full matrix: 12 families × N agents × seed × n_tasks
- [ ] Per-family heatmaps generated
- [ ] Cross-family calibration transfer analysis

### Statistical analysis

- [ ] Bootstrap CIs on every headline number
- [ ] Pairwise Holm-Bonferroni-adjusted significance tests
- [ ] Permutation test on ECE differences
- [ ] Power analysis appendix
- [ ] Ablations: with/without reasoning mode, with/without thinking tokens, with/without explicit confidence head

## C. Paper Writing

- [ ] Abstract finalised
- [ ] Introduction with frontier-saturation narrative + 5 contributions
- [ ] Related work covers calibration, reliability, scoring rules, agent benchmarks
- [ ] Methods: market formulation, scoring, MOP definition
- [ ] Task families table + per-family difficulty ladders figure
- [ ] Headline experimental results table
- [ ] Reliability diagrams
- [ ] Pareto frontier figures
- [ ] Per-family heatmaps
- [ ] MOP analysis figure
- [ ] Findings: H1-H5 confirmed/refuted
- [ ] Gaming-resistance audit section
- [ ] Limitations honest enumeration
- [ ] Conclusion + future work
- [ ] References complete
- [ ] Reproducibility appendix with one-command reproduce script
- [ ] Datasheet for datasets

## D. Release Artifacts

- [ ] HF dataset for seeds & reference scores (`debajyotidasgupta/prophet-seeds`)
- [ ] HF dataset for reference panel scoring (`debajyotidasgupta/prophet-reference`)
- [ ] HF dataset for paper results (`debajyotidasgupta/prophet-results`)
- [ ] PyPI release `prophet-bench`
- [ ] Public GitHub repo (will flip from private when paper-ready)
- [ ] Croissant metadata for each HF dataset
- [ ] Project website / docs

## E. Safety & Ethics

- [ ] Safety family uses placeholder unsafe content only (no actual harmful)
- [ ] Data licence audit for any external assets
- [ ] Privacy statement (no PII)
- [ ] Compute cost transparency (RunPod $ spent reported)

## F. Pre-Submission Audit

- [ ] Berkeley RDI attack family run; each documented passed / blocked
- [ ] 3 frontier-model red team
- [ ] All code linted (`ruff check`) + typed (`mypy src/prophet`)
- [ ] CI green on commit
- [ ] Final reproducibility audit: fresh machine → `make all` produces same numbers within bootstrap CIs
- [ ] Compute & cost statement: dollar amount per full evaluation
