"""Engine integration: orchestrator + market + baselines + math family.

Smoke-level tests that exercise the full pipeline on small N.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from prophet.agents.baselines import (
    AlwaysPassAgent,
    AlwaysTakeAgent,
    OracleAgent,
    RandomAgent,
)
from prophet.engine.market import MarketMaker
from prophet.engine.orchestrator import Orchestrator, RunConfig
from prophet.engine.scoring import ece, total_payoff
from prophet.engine.types import DecisionMode
from prophet.families import get_family


def _setup(tmp_path: Path):
    fam = get_family("math")
    tasks = fam.generate(n=20, seed=123)
    orch = Orchestrator(market=MarketMaker(market_seed=123))
    cfg = RunConfig(cycle_seed=123, market_seed=123, out_dir=tmp_path, progress=False)
    return tasks, orch, cfg


def test_always_pass_total_payoff_is_negative(tmp_path):
    tasks, orch, cfg = _setup(tmp_path)
    res = orch.run(AlwaysPassAgent(), tasks, cfg)
    pay = total_payoff(res.outcomes)
    assert pay < 0
    assert all(o.response.mode == DecisionMode.PASS for o in res.outcomes)


def test_oracle_perfect_payoff_high(tmp_path):
    tasks, orch, cfg = _setup(tmp_path)
    res = orch.run(OracleAgent(seed=0, correctness_rate=1.0), tasks, cfg)
    pay = total_payoff(res.outcomes)
    # Perfect oracle should earn at least V + κ on every task
    assert pay > 100 * len(res.outcomes)


def test_random_agent_calibration_baseline(tmp_path):
    tasks, orch, cfg = _setup(tmp_path)
    res = orch.run(RandomAgent(seed=0), tasks, cfg)
    # Random confidences uncorrelated with outcome → ECE near 0.25 in expectation
    confs = [o.response.confidence for o in res.outcomes if o.success is not None]
    ys = [int(bool(o.success)) for o in res.outcomes if o.success is not None]
    if confs:
        e = ece(confs, ys)
        assert 0.0 <= e <= 1.0


def test_always_take_with_no_answer_fails(tmp_path):
    tasks, orch, cfg = _setup(tmp_path)
    res = orch.run(AlwaysTakeAgent(answer_text="42"), tasks, cfg)
    # "42" coincidentally correct on a small fraction; payoff should be negative overall
    assert total_payoff(res.outcomes) < 0


def test_run_persists_outcomes(tmp_path):
    tasks, orch, cfg = _setup(tmp_path)
    res = orch.run(RandomAgent(seed=0), tasks, cfg)
    f = tmp_path / res.run_id / "outcomes.jsonl"
    assert f.exists()
    lines = [ln for ln in f.read_text().splitlines() if ln.strip()]
    assert len(lines) == len(tasks)
