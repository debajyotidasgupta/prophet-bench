"""Per-family generator + verifier tests.

Each of code, knowledge, reasoning, writing must:
  * generate exactly N tasks for a given seed,
  * be deterministic across two calls with the same seed,
  * span the full difficulty range when n is large enough,
  * earn a strongly positive net payoff for the OracleAgent.

The OracleAgent uses `task.reference_answer` as its response, so passing
through it ↔ verifier(reference_answer) is True. This pins down the
mechanical-correctness contract of every family.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from prophet.agents.baselines import OracleAgent
from prophet.engine.market import MarketMaker
from prophet.engine.orchestrator import Orchestrator, RunConfig
from prophet.engine.scoring import total_payoff
from prophet.families import get_family


FAMILIES = ["code", "knowledge", "reasoning", "writing"]


# -------------------------------------------------------------------------
# Shared fixtures
# -------------------------------------------------------------------------

def _generate(name: str, n: int = 20, seed: int = 13):
    return get_family(name).generate(n=n, seed=seed)


def _run_oracle(tmp_path: Path, name: str, n: int = 20, seed: int = 13):
    fam = get_family(name)
    tasks = fam.generate(n=n, seed=seed)
    orch = Orchestrator(market=MarketMaker(market_seed=seed))
    cfg = RunConfig(cycle_seed=seed, market_seed=seed, out_dir=tmp_path, progress=False)
    return orch.run(OracleAgent(seed=seed, correctness_rate=1.0), tasks, cfg)


# -------------------------------------------------------------------------
# 1) Count is exactly N
# -------------------------------------------------------------------------

@pytest.mark.parametrize("name", FAMILIES)
def test_family_generates_exactly_n(name):
    tasks = _generate(name, n=12, seed=7)
    assert len(tasks) == 12
    assert all(t.family == name for t in tasks)


# -------------------------------------------------------------------------
# 2) Determinism — same seed → same prompts and reference answers
# -------------------------------------------------------------------------

@pytest.mark.parametrize("name", FAMILIES)
def test_family_deterministic_by_seed(name):
    a = _generate(name, n=10, seed=99)
    b = _generate(name, n=10, seed=99)
    assert [t.prompt for t in a] == [t.prompt for t in b]
    assert [t.reference_answer for t in a] == [t.reference_answer for t in b]
    assert [t.task_id for t in a] == [t.task_id for t in b]
    # Different seed should produce different content
    c = _generate(name, n=10, seed=100)
    assert [t.prompt for t in a] != [t.prompt for t in c]


# -------------------------------------------------------------------------
# 3) Tier coverage — large N hits 6+ distinct difficulty levels
# -------------------------------------------------------------------------

@pytest.mark.parametrize("name", FAMILIES)
def test_family_covers_multiple_tiers(name):
    tasks = _generate(name, n=120, seed=1)
    levels = {round(t.difficulty, 2) for t in tasks}
    # At least 6 distinct difficulty tiers should appear with N=120
    assert len(levels) >= 6, f"{name} only saw tiers: {levels}"
    # All in [0,1]
    assert all(0.0 <= t.difficulty <= 1.0 for t in tasks)
    # Coverage spans low and high end
    assert min(levels) <= 0.25
    assert max(levels) >= 0.7


# -------------------------------------------------------------------------
# 4) Oracle baseline → strongly positive payoff
# -------------------------------------------------------------------------

@pytest.mark.parametrize("name", FAMILIES)
def test_family_oracle_earns_positive_payoff(tmp_path, name):
    res = _run_oracle(tmp_path, name, n=20, seed=13)
    pay = total_payoff(res.outcomes)
    # With perfect oracle, each task earns roughly V + κ ≈ 100..180
    # Require average per-task net > 100 (much greater than 0)
    assert pay > 100 * len(res.outcomes), (
        f"{name} oracle payoff was {pay} for {len(res.outcomes)} tasks"
    )
    # All tasks should have a successful outcome
    n_ok = sum(1 for o in res.outcomes if o.success is True)
    assert n_ok == len(res.outcomes), (
        f"{name} oracle missed {len(res.outcomes) - n_ok} task(s)"
    )
