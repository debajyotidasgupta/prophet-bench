"""Google Gemini agent adapter.

Uses the official `google-genai` Python SDK (new unified Client API). Model id
is the literal Gemini identifier (e.g. "gemini-3.1-pro", "gemini-3-flash").

The SDK is imported lazily so this module can be imported on machines that
don't have `google-genai` installed — calls will then raise at construction
time when the missing dependency is actually required.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

from prophet.agents.base import AgentBase
from prophet.utils.cost import estimate_cost

log = logging.getLogger("prophet.agents.google")

ENV_KEY = "GOOGLE_API_KEY"
ENV_KEY_ALT = "GEMINI_API_KEY"


@dataclass(slots=True)
class GoogleAgent(AgentBase):
    """Agent over the Google Gemini API (new google-genai SDK)."""

    model: str = "gemini-3-flash"
    api_key_override: str | None = None
    name: str = ""
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout_s: float = 90.0
    family_support: set[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"google:{self.model}"
        self._api_key = (
            self.api_key_override
            or os.environ.get(ENV_KEY)
            or os.environ.get(ENV_KEY_ALT)
        )
        if not self._api_key:
            log.warning(
                "No Google API key found (env=%s or %s) — calls will fail.",
                ENV_KEY,
                ENV_KEY_ALT,
            )
        try:
            from google import genai
        except Exception as e:  # pragma: no cover - import guard
            raise RuntimeError(
                "google-genai package missing; install with `pip install prophet-bench[api]`"
            ) from e
        self._client = genai.Client(api_key=self._api_key or "EMPTY")

    def _complete(self, prompt: str) -> tuple[str, int, int, float]:
        config: dict[str, Any] = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
        }
        resp = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )
        # `response.text` is a convenience accessor that joins parts.
        text = getattr(resp, "text", None) or ""
        if not text:
            # Fallback: walk candidates → content → parts → text.
            candidates = getattr(resp, "candidates", None) or []
            for cand in candidates:
                content = getattr(cand, "content", None)
                parts = getattr(content, "parts", None) if content else None
                for part in parts or []:
                    t = getattr(part, "text", None)
                    if t:
                        text += t
        usage = getattr(resp, "usage_metadata", None)
        tok_in = int(getattr(usage, "prompt_token_count", 0) or 0) if usage else 0
        tok_out = int(getattr(usage, "candidates_token_count", 0) or 0) if usage else 0
        cost = estimate_cost(self.name, tok_in, tok_out)
        return text, tok_in, tok_out, cost
