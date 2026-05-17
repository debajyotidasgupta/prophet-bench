"""PROPHET: Probabilistic Reliability via Outcome-Pricing for Honest Evaluation of agenT calibration.

A benchmark that turns LLM-agent evaluation into a marketplace: agents commit
(Take), price their belief (Quote with a proper scoring rule), or abstain
(Pass). Headline outputs are a Pareto frontier of expected payoff, ECE,
abstention precision, and Model Overreach Point localization.

This package is research code. APIs are stable for the v1 paper but may evolve
in v2 as task families are added.
"""

from prophet._version import __version__  # noqa: F401
from prophet.engine.types import (  # noqa: F401
    AgentResponse,
    MarketOffer,
    Outcome,
    Task,
)

__all__ = [
    "AgentResponse",
    "MarketOffer",
    "Outcome",
    "Task",
    "__version__",
]
