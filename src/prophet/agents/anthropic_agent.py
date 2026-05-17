"""Anthropic Claude agent adapter.

Uses the official `anthropic` Python SDK. Model id is the literal Claude
identifier (e.g. "claude-opus-4-7", "claude-sonnet-4-6", "claude-haiku-4-5").

The SDK is imported lazily so this module can be imported on machines that
don't have `anthropic` installed — calls will then raise at construction
time when the missing dependency is actually required.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

from prophet.agents.base import AgentBase
from prophet.utils.cost import estimate_cost

log = logging.getLogger("prophet.agents.anthropic")

ENV_KEY = "ANTHROPIC_API_KEY"


@dataclass(slots=True)
class AnthropicAgent(AgentBase):
    """Agent over the Anthropic Messages API."""

    model: str = "claude-haiku-4-5"
    api_key_override: str | None = None
    name: str = ""
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout_s: float = 90.0
    family_support: set[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"anthropic:{self.model}"
        self._api_key = self.api_key_override or os.environ.get(ENV_KEY)
        if not self._api_key:
            log.warning(
                "No Anthropic API key found (env=%s) — calls will fail.",
                ENV_KEY,
            )
        try:
            from anthropic import Anthropic
        except Exception as e:  # pragma: no cover - import guard
            raise RuntimeError(
                "anthropic package missing; install with `pip install prophet-bench[api]`"
            ) from e
        self._client = Anthropic(
            api_key=self._api_key or "EMPTY",
            timeout=self.timeout_s,
            max_retries=2,
        )

    def _complete(self, prompt: str) -> tuple[str, int, int, float]:
        resp = self._client.messages.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        # Concatenate text blocks (vision/tool blocks are ignored here).
        chunks: list[str] = []
        for block in resp.content or []:
            text = getattr(block, "text", None)
            if text:
                chunks.append(text)
        text = "".join(chunks)
        usage = getattr(resp, "usage", None)
        tok_in = int(getattr(usage, "input_tokens", 0) or 0) if usage else 0
        tok_out = int(getattr(usage, "output_tokens", 0) or 0) if usage else 0
        cost = estimate_cost(self.name, tok_in, tok_out)
        return text, tok_in, tok_out, cost
