.PHONY: install test smoke lint format clean run-baselines run-paper paper

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
	cd docs/paper && pdflatex prophet.tex && bibtex prophet && pdflatex prophet.tex && pdflatex prophet.tex

# Full reproducibility: run every baseline + API model + analysis. Use a budget cap.
run-paper:
	bash scripts/run_full_eval.sh 30 100 42 results/paper
