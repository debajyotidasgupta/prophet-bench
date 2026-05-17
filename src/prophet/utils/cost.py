"""Cost tracking + provider price tables.

We keep an in-process counter and a small JSON sidecar per run; this is the
primary guardrail against runaway API spend. The orchestrator enforces a
hard `PROPHET_MAX_RUN_COST_USD` cap.

Prices are updated as of May 2026. We err on the side of *over*-estimating
to keep guardrails conservative. Add new models as needed.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

log = logging.getLogger("prophet.cost")


@dataclass(slots=True)
class TokenPrice:
    """USD per million tokens."""

    in_per_m: float
    out_per_m: float


# Conservative reference prices — May 2026.
# Update keys to match the literal model id used by each provider.
PRICES: dict[str, TokenPrice] = {
    # OpenAI (May 2026)
    "openai:gpt-5": TokenPrice(in_per_m=2.50, out_per_m=10.00),
    "openai:gpt-5-mini": TokenPrice(in_per_m=0.20, out_per_m=0.80),
    "openai:gpt-5-nano": TokenPrice(in_per_m=0.05, out_per_m=0.20),
    "openai:gpt-4o": TokenPrice(in_per_m=2.50, out_per_m=10.00),
    "openai:gpt-4o-mini": TokenPrice(in_per_m=0.15, out_per_m=0.60),
    # Anthropic
    "anthropic:claude-opus-4-7": TokenPrice(in_per_m=15.00, out_per_m=75.00),
    "anthropic:claude-sonnet-4-6": TokenPrice(in_per_m=3.00, out_per_m=15.00),
    "anthropic:claude-haiku-4-5": TokenPrice(in_per_m=0.80, out_per_m=4.00),
    # Google
    "google:gemini-3.1-pro": TokenPrice(in_per_m=2.50, out_per_m=15.00),
    "google:gemini-3-flash": TokenPrice(in_per_m=0.30, out_per_m=2.50),
    "google:gemini-3-flash-lite": TokenPrice(in_per_m=0.10, out_per_m=0.40),
    # Together / open weights via API (approximate)
    "together:meta-llama/Llama-4-70B-Instruct": TokenPrice(in_per_m=0.90, out_per_m=0.90),
    "together:Qwen/Qwen3-32B-Instruct": TokenPrice(in_per_m=0.80, out_per_m=0.80),
    "together:Qwen/Qwen3-7B-Instruct": TokenPrice(in_per_m=0.20, out_per_m=0.20),
    # vLLM local (operating cost only; model itself is free)
    "vllm:Qwen/Qwen3-7B-Instruct": TokenPrice(in_per_m=0.0, out_per_m=0.0),
    "vllm:Qwen/Qwen3-32B-Instruct": TokenPrice(in_per_m=0.0, out_per_m=0.0),
    "vllm:meta-llama/Llama-4-8B-Instruct": TokenPrice(in_per_m=0.0, out_per_m=0.0),
    "hf:default": TokenPrice(in_per_m=0.0, out_per_m=0.0),
    # Baselines (free)
    "baseline:random": TokenPrice(in_per_m=0.0, out_per_m=0.0),
    "baseline:always-take": TokenPrice(in_per_m=0.0, out_per_m=0.0),
    "baseline:always-pass": TokenPrice(in_per_m=0.0, out_per_m=0.0),
    "baseline:oracle": TokenPrice(in_per_m=0.0, out_per_m=0.0),
}


def estimate_cost(provider_model: str, tokens_in: int, tokens_out: int) -> float:
    """Estimate cost in USD for an API call."""
    p = PRICES.get(provider_model)
    if p is None:
        p = PRICES.get(f"{provider_model.split(':')[0]}:default")
    if p is None:
        log.debug("No price entry for %s — returning 0 (treat as free)", provider_model)
        return 0.0
    return (tokens_in / 1_000_000.0) * p.in_per_m + (tokens_out / 1_000_000.0) * p.out_per_m


class CostMeter:
    """Thread-safe running cost meter with hard cap."""

    def __init__(self, cap_usd: float | None = None) -> None:
        self.cap = cap_usd
        self.total = 0.0
        self._lock = threading.Lock()
        self.tripped = False

    def add(self, amount: float) -> None:
        with self._lock:
            self.total += max(0.0, amount)
            if self.cap is not None and self.total > self.cap and not self.tripped:
                self.tripped = True
                log.warning("CostMeter cap reached: $%.4f > $%.4f", self.total, self.cap)

    def check(self) -> bool:
        with self._lock:
            return not self.tripped

    def __repr__(self) -> str:
        return f"CostMeter(total=${self.total:.4f}, cap=${self.cap})"
