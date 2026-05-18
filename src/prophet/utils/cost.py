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
    "baseline:oracle-noisy": TokenPrice(in_per_m=0.0, out_per_m=0.0),
    # OpenRouter (live May 2026; queried from /models endpoint)
    "openrouter:openai/gpt-5-nano": TokenPrice(0.05, 0.40),
    "openrouter:openai/gpt-5-mini": TokenPrice(0.25, 2.00),
    "openrouter:openai/gpt-5": TokenPrice(1.25, 10.00),
    "openrouter:openai/gpt-5.4": TokenPrice(2.50, 15.00),
    "openrouter:openai/gpt-5.4-mini": TokenPrice(0.75, 4.50),
    "openrouter:openai/gpt-5.4-nano": TokenPrice(0.20, 1.25),
    "openrouter:openai/gpt-5.5": TokenPrice(5.00, 30.00),
    "openrouter:openai/gpt-5.3-chat": TokenPrice(1.75, 14.00),
    "openrouter:openai/gpt-5.1": TokenPrice(1.25, 10.00),
    "openrouter:openai/gpt-5.2": TokenPrice(1.75, 14.00),
    "openrouter:openai/gpt-5.2-pro": TokenPrice(21.00, 168.00),
    "openrouter:anthropic/claude-haiku-4.5": TokenPrice(1.00, 5.00),
    "openrouter:anthropic/claude-sonnet-4.6": TokenPrice(3.00, 15.00),
    "openrouter:anthropic/claude-sonnet-4.5": TokenPrice(3.00, 15.00),
    "openrouter:anthropic/claude-opus-4.7": TokenPrice(5.00, 25.00),
    "openrouter:anthropic/claude-opus-4.6": TokenPrice(5.00, 25.00),
    "openrouter:google/gemini-3.1-pro-preview": TokenPrice(2.00, 12.00),
    "openrouter:google/gemini-3.1-flash-lite": TokenPrice(0.25, 1.50),
    "openrouter:google/gemini-3-flash-preview": TokenPrice(0.50, 3.00),
    "openrouter:google/gemini-2.5-pro": TokenPrice(1.25, 10.00),
    "openrouter:google/gemini-2.5-flash-lite": TokenPrice(0.10, 0.40),
    "openrouter:meta-llama/llama-4-maverick": TokenPrice(0.15, 0.60),
    "openrouter:meta-llama/llama-4-scout": TokenPrice(0.08, 0.30),
    "openrouter:meta-llama/llama-3.3-70b-instruct": TokenPrice(0.10, 0.32),
    "openrouter:qwen/qwen3-32b": TokenPrice(0.08, 0.28),
    "openrouter:qwen/qwen3-14b": TokenPrice(0.10, 0.24),
    "openrouter:qwen/qwen3-8b": TokenPrice(0.05, 0.40),
    "openrouter:qwen/qwen3-235b-a22b-2507": TokenPrice(0.071, 0.100),
    "openrouter:qwen/qwen3-235b-a22b-thinking-2507": TokenPrice(0.150, 1.495),
    "openrouter:qwen/qwen3.6-flash": TokenPrice(0.188, 1.125),
    "openrouter:deepseek/deepseek-v3.2": TokenPrice(0.252, 0.378),
    "openrouter:deepseek/deepseek-v3.2-exp": TokenPrice(0.270, 0.410),
    "openrouter:deepseek/deepseek-r1": TokenPrice(0.700, 2.500),
    "openrouter:deepseek/deepseek-r1-distill-qwen-32b": TokenPrice(0.290, 0.290),
    "openrouter:moonshotai/kimi-k2-thinking": TokenPrice(0.600, 2.500),
    "openrouter:moonshotai/kimi-k2.5": TokenPrice(0.400, 1.900),
    "openrouter:moonshotai/kimi-k2.6": TokenPrice(0.730, 3.490),
    "openrouter:x-ai/grok-4.3": TokenPrice(1.250, 2.500),
    "openrouter:mistralai/mistral-large-2512": TokenPrice(0.500, 1.500),
    "openrouter:microsoft/phi-4": TokenPrice(0.065, 0.140),
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
