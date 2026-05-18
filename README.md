# PROPHET

**P**robabilistic **R**eliability via **O**utcome-**P**ricing for **H**onest **E**valuation of agen**T** calibration.

A benchmark for measuring the calibration of LLM-based agents under economic pressure. Agents must price their own ability to succeed at heterogeneous tasks; the leaderboard is a Pareto frontier of expected payoff, expected calibration error (ECE), abstention precision, and Model Overreach Point (MOP) localization.

## Why this benchmark exists

By May 2026 every static-resettable LLM benchmark we built has saturated or is saturating. The capability that has *not* saturated and is barely measured is **calibrated, persistent agency**: knowing in advance how likely you are to succeed, walking away from tasks you shouldn't take, and pricing your effort in a way that survives heterogeneous workloads.

PROPHET turns evaluation into a marketplace:

- **Take($T)** — claim the task; succeed → +V, fail → −C.
- **Quote(P̂)** — declare your probability of success; scored on a proper scoring rule (Brier-based).
- **Pass** — decline outright at a small research cost δ.

Across 12 task families spanning code, math, knowledge, reasoning, writing, browsing, tool-use, multimodal, data analysis, scientific reading, multilingual, and safety, PROPHET measures whether a system can be *trustably autonomous*.

## Quickstart

```bash
# 1. Install
pip install -e ".[api,dev]"

# 2. Copy .env.example to .env and fill in tokens
cp .env.example .env

# 3. Run a tiny smoke test (CPU, no API key needed)
prophet smoke

# 4. Run a single family with a closed-model API baseline
prophet run --agent openai:gpt-5-mini --families math --n 50

# 5. Run full benchmark on an open-source model
prophet run --agent vllm:Qwen/Qwen3-7B-Instruct --families all --n 100

# 6. Compute Pareto + plots
prophet analyze results/runs/<run-id>
```

## Repo structure

```
prophet/
├── src/prophet/
│   ├── engine/          # Market maker, orchestrator, scoring
│   ├── families/        # 12 task families with procgen + verifier
│   ├── agents/          # Open + closed model adapters
│   ├── procgen/         # Procedural task generators
│   ├── analysis/        # Pareto frontier, calibration curves
│   └── cli.py           # `prophet` CLI
├── tests/
├── docs/paper/          # NeurIPS submission
├── scripts/
│   ├── runpod_launch.py # GPU pod launcher with cost guardrails
│   └── *.sh             # Reproducibility scripts
├── data/                # Seeds + reference scores
├── results/             # (gitignored) experiment outputs
└── configs/             # YAML run configs
```

## Citation

```bibtex
@inproceedings{prophet2026,
  title={PROPHET: Probabilistic Reliability via Outcome-Pricing for Honest Evaluation of Agent Calibration},
  author={Dasgupta, Debajyoti},
  booktitle={NeurIPS 2026 Datasets and Benchmarks Track},
  year={2026}
}
```

## License

MIT. See `LICENSE`.
