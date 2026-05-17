"""Deterministic baselines.

Used for:
  • smoke-testing the pipeline (no API key needed);
  • lower-bound (RandomAgent) and upper-bound (OracleAgent) leaderboard anchors;
  • sanity-checking that always-Pass / always-Take strategies score badly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from prophet.engine.types import (
    AgentResponse,
    DecisionMode,
    MarketOffer,
    Task,
)


@dataclass
class RandomAgent:
    name: str = "baseline:random"
    seed: int = 0
    family_support: set[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    _rng: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def respond(self, task: Task, offer: MarketOffer) -> AgentResponse:
        # np.random.Generator.choice over a heterogeneous Python list will
        # coerce to numpy strings and lose Enum identity; index instead.
        modes = (DecisionMode.TAKE, DecisionMode.QUOTE, DecisionMode.PASS)
        mode = modes[int(self._rng.integers(0, len(modes)))]
        confidence = float(self._rng.random())
        answer = None
        if mode != DecisionMode.PASS:
            answer = "random answer"
        return AgentResponse(
            task_id=task.task_id,
            mode=mode,
            confidence=confidence,
            answer=answer,
            reasoning="random baseline",
            tokens_in=0,
            tokens_out=0,
            wall_time_s=0.0,
            cost_usd=0.0,
        )


@dataclass
class AlwaysTakeAgent:
    name: str = "baseline:always-take"
    confidence: float = 0.99
    answer_text: str = "<no answer>"
    family_support: set[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def respond(self, task: Task, offer: MarketOffer) -> AgentResponse:
        return AgentResponse(
            task_id=task.task_id,
            mode=DecisionMode.TAKE,
            confidence=self.confidence,
            answer=self.answer_text,
            reasoning=None,
        )


@dataclass
class AlwaysPassAgent:
    name: str = "baseline:always-pass"
    family_support: set[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def respond(self, task: Task, offer: MarketOffer) -> AgentResponse:
        return AgentResponse(
            task_id=task.task_id,
            mode=DecisionMode.PASS,
            confidence=0.5,
            answer=None,
            reasoning=None,
        )


@dataclass
class OracleAgent:
    """Cheats by reading `task.reference_answer`.

    Used as the upper bound. NOT a real adapter — never list on leaderboard.
    Calibration is perfect by construction; this defines maximum payoff.
    """

    name: str = "baseline:oracle"
    family_support: set[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    correctness_rate: float = 1.0  # set <1.0 to simulate a noisy oracle
    seed: int = 0
    _rng: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def respond(self, task: Task, offer: MarketOffer) -> AgentResponse:
        will_be_correct = self._rng.random() < self.correctness_rate
        answer = task.reference_answer or "<oracle has no reference>"
        if not will_be_correct:
            answer = "<oracle deliberately wrong>"
        return AgentResponse(
            task_id=task.task_id,
            mode=DecisionMode.QUOTE,
            confidence=self.correctness_rate if not will_be_correct else self.confidence,
            answer=answer,
            reasoning=None,
        )
