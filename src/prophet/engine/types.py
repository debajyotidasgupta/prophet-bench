"""Core dataclasses + protocols used across engine, families, and agents.

These types define the contract between PROPHET's components. Agents see Task
and MarketOffer, produce AgentResponse. The engine reconciles into Outcome.
"""

from __future__ import annotations

import enum
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


class DecisionMode(str, enum.Enum):
    """How an agent committed at the market.

    - TAKE: the agent attempted; success → +V, failure → −C.
    - QUOTE: the agent declared P̂ and attempted; calibration is scored on top.
    - PASS: the agent declined; only the research cost δ is paid.

    Note: in PROPHET v1, every attempt is also implicitly a Quote — the agent
    always declares P̂ — so TAKE vs QUOTE collapses to "is V/C the full stake
    or scaled by the quote." We model the two modes explicitly to allow
    future variants where the market hides the calibration payoff from
    risk-neutral takers.
    """

    TAKE = "take"
    QUOTE = "quote"
    PASS = "pass"


@dataclass(slots=True)
class Task:
    """A single procedurally-generated benchmark instance.

    Mechanical verification (`verifier`) is the gold standard for primary
    metrics. When a family cannot avoid an LLM-judge (e.g. open-ended writing),
    it should set `judge_required=True` and provide a judging callable.
    """

    task_id: str
    family: str
    difficulty: float  # reference panel's empirical difficulty, in [0, 1]
    prompt: str
    verifier: Callable[[str], bool]
    metadata: dict[str, Any] = field(default_factory=dict)
    judge_required: bool = False
    reference_answer: str | None = None  # for grading / debugging only
    estimated_seconds: float = 30.0  # reference panel wall-clock estimate


@dataclass(slots=True)
class MarketOffer:
    """The market's offer for a single task.

    All values are denominated in *pseudo-cents* (100 = $1.00). We keep things
    integer-ish to make payoff math reproducible; conversion to real dollars
    is a configurable post-hoc scale factor (see `configs/payoff.yaml`).
    """

    task_id: str
    family: str
    V_success: float  # reward on success (Take or Quote+success)
    C_failure: float  # fine on failure
    kappa_calib: float  # calibration payoff coefficient (Brier-scaled)
    delta_pass: float  # research cost to pass
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentResponse:
    """An agent's reply for a single (task, offer) pair.

    The agent always provides a confidence in [0, 1]; this is the proper
    scoring-rule signal. The agent's *mode* selects how that confidence is
    used in the payoff:
      - TAKE: full V/C stake; no calibration term applied.
      - QUOTE: full V/C stake on outcome AND calibration term applied.
      - PASS: only -delta_pass; answer/confidence are not graded.

    `tokens_in`, `tokens_out`, and `wall_time_s` are populated by the
    orchestrator. The agent should populate `reasoning` if it produced any.
    """

    task_id: str
    mode: DecisionMode
    confidence: float
    answer: str | None = None
    reasoning: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    wall_time_s: float = 0.0
    cost_usd: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Outcome:
    """The graded outcome of one task instance."""

    task_id: str
    family: str
    difficulty: float
    response: AgentResponse
    offer: MarketOffer
    success: bool | None  # None iff agent passed
    payoff_attempt: float
    payoff_calib: float
    payoff_total: float
    judge_score: float | None = None  # for open-ended families
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class TaskFamily(Protocol):
    """Protocol implemented by every task family.

    Conventions:
      - `name` is a stable kebab-case identifier ("math", "tool-use", "browser").
      - `generate(n, seed, difficulty_range)` returns *exactly* n tasks.
      - `reference_score(task, response)` allows LLM-judge fallback for
        open-ended families. Mechanical verifiers should not need this.
    """

    name: str
    description: str

    def generate(
        self,
        n: int,
        seed: int,
        difficulty_range: tuple[float, float] = (0.0, 1.0),
    ) -> list[Task]: ...

    def reference_score(self, task: Task, response: AgentResponse) -> float | None:
        """Return float in [0,1] when family needs LLM-judge, else None."""
        ...


@runtime_checkable
class Agent(Protocol):
    """Protocol every agent adapter implements."""

    name: str
    family_support: set[str] | None  # None = supports all families

    def respond(self, task: Task, offer: MarketOffer) -> AgentResponse: ...


def aggregate_payoff(outcomes: Iterable[Outcome]) -> float:
    """Sum payoff_total over an iterable of outcomes."""
    return sum(o.payoff_total for o in outcomes)
