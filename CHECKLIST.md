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
- [ ] 11 additional task families implemented (in-flight)
- [ ] Anthropic, Google, HF Transformers, vLLM local adapters (in-flight)
- [ ] Concurrent runner with rate limiting (in-flight)
- [ ] HF Inference adapter
- [ ] Analysis pipeline: Pareto, plots, heatmaps, per-family report
- [ ] W&B integration end-to-end
- [ ] RunPod launcher with cost guardrails
- [ ] Reproducibility script `scripts/run_full_eval.sh`
- [ ] CI on GitHub Actions

## B. Experimental Matrix

### Models (≥ 12 for headline)

- [ ] **Closed frontier** (via APIs)
  - [ ] openai:gpt-5
  - [ ] openai:gpt-5-mini
  - [ ] openai:gpt-5-nano
  - [ ] anthropic:claude-opus-4-7
  - [ ] anthropic:claude-sonnet-4-6
  - [ ] anthropic:claude-haiku-4-5
  - [ ] google:gemini-3.1-pro
  - [ ] google:gemini-3-flash
- [ ] **Open frontier** (vLLM / Together / DeepInfra)
  - [ ] meta-llama/Llama-4-70B-Instruct
  - [ ] meta-llama/Llama-4-8B-Instruct
  - [ ] Qwen/Qwen3-32B-Instruct
  - [ ] Qwen/Qwen3-14B-Instruct
  - [ ] Qwen/Qwen3-7B-Instruct
  - [ ] DeepSeek-V3.2 (Together / DeepInfra)
  - [ ] Mistral-Large-3 (DeepInfra)
  - [ ] Kimi-K2-Thinking variants if available
- [ ] **Reasoning-mode** variants on/off comparison (Qwen3 thinking-tokens; o-series)
- [ ] **Baselines**: random, always-take, always-pass, oracle, oracle-noisy

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
