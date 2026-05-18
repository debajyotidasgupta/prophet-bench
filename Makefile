.PHONY: install test smoke lint format clean run-baselines run-paper run-paper-hard paper finalize-paper

PY ?= .venv/bin/python
PIP ?= .venv/bin/pip
PROPHET ?= .venv/bin/prophet

install:
	python3 -m venv .venv
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e ".[api,dev]"

test:
	$(PY) -m pytest -x -q

smoke:
	$(PROPHET) smoke --n 50

lint:
	$(PY) -m ruff check src tests
	$(PY) -m mypy src/prophet || true

format:
	$(PY) -m ruff check --fix src tests

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache **/__pycache__ build dist *.egg-info

run-baselines:
	$(PROPHET) run --agent baseline:random --families all --n 50
	$(PROPHET) run --agent baseline:always-take --families all --n 50
	$(PROPHET) run --agent baseline:always-pass --families all --n 50
	$(PROPHET) run --agent baseline:oracle --families all --n 50
	$(PROPHET) analyze-compare results/runs

paper:
	cd docs/paper && tectonic prophet.tex 2>/dev/null || (cd docs/paper && pdflatex prophet.tex && bibtex prophet && pdflatex prophet.tex && pdflatex prophet.tex)

# Full reproducibility: run every baseline + API model + analysis. Use a budget cap.
run-paper:
	bash scripts/run_full_eval.sh 30 100 42 results/paper

# Hard tier — multi-seed N=20 on Pareto-essentials (~$15, 2-3h wall-clock).
run-paper-hard:
	bash scripts/run_hard_tier_phase_b.sh

# Plug all paper TODOs + regen figures + tables after a fresh matrix.
finalize-paper:
	PYTHONPATH=src $(PY) scripts/finalize_paper.py --runs-dir results/openrouter_matrix/runs
	PYTHONPATH=src $(PY) scripts/make_paper_figures.py --runs-dir results/openrouter_matrix/runs --out-dir docs/paper/figures
	$(PY) scripts/build_two_tier_leaderboard.py
	$(PY) scripts/plug_paper_todos.py
	$(PY) scripts/price_sensitivity_ablation.py
	$(PY) scripts/empirical_difficulty_figure.py
	$(PY) scripts/family_pass_rate_heatmap.py
	$(PY) scripts/payoff_per_dollar.py
	$(PY) scripts/stricter_verifier_ablation.py
	@if [ -d results/openrouter_hard_v2/runs ]; then PYTHONPATH=src $(PY) scripts/multi_seed_variance.py; fi
	$(MAKE) paper
