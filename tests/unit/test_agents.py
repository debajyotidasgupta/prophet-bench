"""Agent unit tests: parsing, baselines, ConcurrentRunner.

These tests must NOT make real API calls. We test:
  - `parse_response` over malformed / well-formed model output;
  - deterministic baselines produce their expected mode;
  - `ConcurrentRunner` produces identical results to a sequential run (same
    ordering, same content).
  - `build_agent` dispatches correctly for baseline schemes (without touching
    optional SDKs).
"""

from __future__ import annotations

from typing import Any

import pytest

from prophet.agents import (
    AlwaysPassAgent,
    AlwaysTakeAgent,
    ConcurrentRunner,
    OracleAgent,
    RandomAgent,
    build_agent,
)
from prophet.agents.base import parse_response
from prophet.engine.market import MarketMaker
from prophet.engine.types import (
    AgentResponse,
    DecisionMode,
    MarketOffer,
    Task,
)
from prophet.families import get_family


# ---------- parse_response -----------------------------------------------


def _task(tid: str = "t1") -> Task:
    return Task(
        task_id=tid,
        family="math",
        difficulty=0.5,
        prompt="prompt",
        verifier=lambda s: s.strip() == "42",
        reference_answer="42",
    )


def test_parse_response_well_formed():
    raw = (
        "Reasoning here.\n"
        "<decision>\n"
        "mode: QUOTE\n"
        "confidence: 0.73\n"
        "answer: 42\n"
        "</decision>\n"
    )
    r = parse_response(_task(), raw)
    assert r.mode == DecisionMode.QUOTE
    assert r.confidence == pytest.approx(0.73)
    assert r.answer == "42"


def test_parse_response_no_signal_at_all_defaults_to_pass():
    # Only when the raw text has no usable signal AND no meaningful last line.
    task = _task()
    r = parse_response(task, "   \n\n   \n\n")
    assert r.mode == DecisionMode.PASS
    assert r.confidence == 0.5
    assert r.answer is None


def test_parse_response_missing_block_uses_last_line():
    r = parse_response(_task(), "I don't know.")
    assert r.mode == DecisionMode.PASS
    assert r.confidence == 0.5
    assert r.answer is None
    assert r.metadata.get("parse_error") == "no_decision_block"


def test_parse_response_garbled_mode_defaults_to_pass():
    raw = "<decision>\nmode: MAYBE\nconfidence: 0.4\nanswer: 7\n</decision>"
    r = parse_response(_task(), raw)
    assert r.mode == DecisionMode.PASS


def test_parse_response_clamps_high_confidence():
    raw_high = "<decision>\nmode: TAKE\nconfidence: 9.5\nanswer: 7\n</decision>"
    assert parse_response(_task(), raw_high).confidence == 1.0


def test_parse_response_unparseable_confidence_falls_back():
    # The regex doesn't match a leading "-"; parser falls back to 0.5.
    raw_neg = "<decision>\nmode: TAKE\nconfidence: -X\nanswer: 7\n</decision>"
    r = parse_response(_task(), raw_neg)
    assert r.confidence == 0.5


def test_parse_response_take_without_answer_returns_none():
    raw = "<decision>\nmode: TAKE\nconfidence: 0.9\nanswer: \n</decision>"
    r = parse_response(_task(), raw)
    assert r.mode == DecisionMode.TAKE
    assert r.answer is None


def test_parse_response_is_case_insensitive_mode():
    raw = "<decision>\nmode: take\nconfidence: 0.5\nanswer: x\n</decision>"
    r = parse_response(_task(), raw)
    assert r.mode == DecisionMode.TAKE


# ---------- Baselines produce expected modes -----------------------------


def _offer() -> MarketOffer:
    return MarketOffer(
        task_id="t1",
        family="math",
        V_success=100,
        C_failure=100,
        kappa_calib=50,
        delta_pass=2,
    )


def test_always_pass_returns_pass():
    a = AlwaysPassAgent()
    r = a.respond(_task(), _offer())
    assert r.mode == DecisionMode.PASS
    assert r.answer is None


def test_always_take_returns_take_with_answer():
    a = AlwaysTakeAgent(answer_text="42")
    r = a.respond(_task(), _offer())
    assert r.mode == DecisionMode.TAKE
    assert r.answer == "42"
    assert 0.0 <= r.confidence <= 1.0


def test_random_agent_emits_only_valid_modes():
    a = RandomAgent(seed=0)
    modes_seen: set[DecisionMode] = set()
    for i in range(40):
        r = a.respond(_task(f"t{i}"), _offer())
        modes_seen.add(r.mode)
        assert 0.0 <= r.confidence <= 1.0
    # With seed 0 over 40 draws, we should see at least 2 distinct modes
    assert len(modes_seen) >= 2
    assert modes_seen.issubset({DecisionMode.TAKE, DecisionMode.QUOTE, DecisionMode.PASS})


def test_oracle_uses_reference_answer():
    a = OracleAgent(seed=0, correctness_rate=1.0)
    r = a.respond(_task(), _offer())
    assert r.mode == DecisionMode.QUOTE
    assert r.answer == "42"


def test_oracle_noisy_can_lie():
    a = OracleAgent(seed=1, correctness_rate=0.0)  # always wrong
    r = a.respond(_task(), _offer())
    assert r.answer != "42"


# ---------- build_agent dispatcher ---------------------------------------


def test_build_agent_baseline_random():
    a = build_agent("baseline:random", seed=0)
    assert isinstance(a, RandomAgent)


def test_build_agent_baseline_always_take():
    a = build_agent("baseline:always-take")
    assert isinstance(a, AlwaysTakeAgent)


def test_build_agent_baseline_always_pass():
    a = build_agent("baseline:always-pass")
    assert isinstance(a, AlwaysPassAgent)


def test_build_agent_baseline_oracle():
    a = build_agent("baseline:oracle")
    assert isinstance(a, OracleAgent)


def test_build_agent_unknown_baseline_raises():
    with pytest.raises(ValueError):
        build_agent("baseline:nope")


def test_build_agent_missing_colon_raises():
    with pytest.raises(ValueError):
        build_agent("noscheme")


def test_build_agent_unknown_scheme_raises():
    with pytest.raises(ValueError):
        build_agent("alienprovider:foo")


# ---------- ConcurrentRunner: parity with sequential ---------------------


class _DeterministicAgent:
    """A deterministic agent whose response depends only on task.task_id.

    We intentionally do NOT use RandomAgent here because it stores an rng with
    intra-call state, which would yield different sequential vs parallel
    interleavings. The point of the test is *output equality*, so we want a
    pure function of (task_id).
    """

    def __init__(self) -> None:
        self.name = "test:deterministic"
        self.family_support = None

    def respond(self, task: Task, offer: MarketOffer) -> AgentResponse:
        # Pick mode/confidence/answer from a hash of task_id so it's reproducible.
        h = abs(hash(task.task_id))
        modes = [DecisionMode.TAKE, DecisionMode.QUOTE, DecisionMode.PASS]
        mode = modes[h % 3]
        confidence = ((h // 3) % 1000) / 1000.0
        answer = None if mode == DecisionMode.PASS else f"ans-{task.task_id}"
        return AgentResponse(
            task_id=task.task_id,
            mode=mode,
            confidence=confidence,
            answer=answer,
            reasoning=None,
            tokens_in=10,
            tokens_out=5,
            wall_time_s=0.0,
            cost_usd=0.0,
        )


def _gen_tasks(n: int = 12):
    return get_family("math").generate(n=n, seed=42)


def _seq_run(agent, tasks, market):
    return [(t, market.offer(t), agent.respond(t, market.offer(t))) for t in tasks]


def test_concurrent_runner_matches_sequential_ordering_and_content():
    tasks = _gen_tasks(16)
    market = MarketMaker(market_seed=42)
    agent = _DeterministicAgent()

    seq = _seq_run(agent, tasks, market)
    runner = ConcurrentRunner(_DeterministicAgent(), max_concurrency=4)
    par = runner.run_batch(tasks, market)

    assert len(seq) == len(par)
    for (ts, os_, rs), (tp, op, rp) in zip(seq, par, strict=False):
        assert ts.task_id == tp.task_id  # ordering preserved
        assert rs.mode == rp.mode
        assert rs.confidence == rp.confidence
        assert rs.answer == rp.answer


def test_concurrent_runner_with_rate_limit_preserves_ordering():
    tasks = _gen_tasks(8)
    market = MarketMaker(market_seed=42)
    runner = ConcurrentRunner(_DeterministicAgent(), max_concurrency=2, per_sec_limit=200.0)
    par = runner.run_batch(tasks, market)
    # Ordering preserved by task_id
    assert [r.task_id for (_t, _o, r) in par] == [t.task_id for t in tasks]


def test_concurrent_runner_single_concurrency_acts_like_sequential():
    tasks = _gen_tasks(6)
    market = MarketMaker(market_seed=42)
    runner = ConcurrentRunner(_DeterministicAgent(), max_concurrency=1)
    par = runner.run_batch(tasks, market)
    assert [r.task_id for (_t, _o, r) in par] == [t.task_id for t in tasks]


def test_concurrent_runner_handles_empty_batch():
    market = MarketMaker(market_seed=42)
    runner = ConcurrentRunner(_DeterministicAgent(), max_concurrency=4)
    par = runner.run_batch([], market)
    assert par == []


# ---------- Lazy-import safety -------------------------------------------


def test_agents_init_module_importable_without_optional_sdks():
    # The agents package must import cleanly without forcing transformers / vllm.
    import importlib
    m = importlib.import_module("prophet.agents")
    assert hasattr(m, "build_agent")
    assert hasattr(m, "ConcurrentRunner")


def test_anthropic_module_importable_without_api_key(monkeypatch):
    """Importing the module should not require an API key."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    import importlib
    m = importlib.import_module("prophet.agents.anthropic_agent")
    assert hasattr(m, "AnthropicAgent")


def test_google_module_importable_without_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    import importlib
    m = importlib.import_module("prophet.agents.google_agent")
    assert hasattr(m, "GoogleAgent")


def test_transformers_module_importable_without_torch():
    import importlib
    m = importlib.import_module("prophet.agents.transformers_agent")
    assert hasattr(m, "TransformersAgent")


def test_vllm_local_module_importable_without_vllm():
    import importlib
    m = importlib.import_module("prophet.agents.vllm_local")
    assert hasattr(m, "VLLMLocalAgent")
