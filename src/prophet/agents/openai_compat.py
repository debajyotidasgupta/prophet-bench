"""OpenAI-compatible agent.

Works for:
  - OpenAI (api.openai.com)
  - Together.ai (api.together.xyz, OpenAI-compatible)
  - DeepInfra, Fireworks, Groq, OpenRouter (all OpenAI-compatible)
  - vLLM with --served-model-name (local OpenAI-compatible server)
  - Hugging Face Inference Endpoints (OpenAI-compatible mode)

The `provider` arg selects the base_url + the env-var name for the API key.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

from prophet.agents.base import AgentBase
from prophet.utils.cost import estimate_cost

log = logging.getLogger("prophet.agents.openai")

PROVIDER_BASEURLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "together": "https://api.together.xyz/v1",
    "deepinfra": "https://api.deepinfra.com/v1/openai",
    "fireworks": "https://api.fireworks.ai/inference/v1",
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "hf-inference": "https://api-inference.huggingface.co/v1",  # text-generation router
    "vllm": "http://localhost:8000/v1",
}

PROVIDER_ENVKEY: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "together": "TOGETHER_API_KEY",
    "deepinfra": "DEEPINFRA_API_KEY",
    "fireworks": "FIREWORKS_API_KEY",
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "hf-inference": "HUGGINGFACE_TOKEN",
    "vllm": "VLLM_API_KEY",  # often "EMPTY"
}


@dataclass(slots=True)
class OpenAICompatAgent(AgentBase):
    """Agent over any OpenAI-compatible chat completions endpoint."""

    model: str = ""
    provider: str = "openai"
    base_url_override: str | None = None
    api_key_override: str | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)
    name: str = ""
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout_s: float = 90.0
    family_support: set[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"{self.provider}:{self.model}"
        self._base_url = self.base_url_override or PROVIDER_BASEURLS.get(self.provider, "")
        env_key = PROVIDER_ENVKEY.get(self.provider, "OPENAI_API_KEY")
        self._api_key = self.api_key_override or os.environ.get(env_key) or "EMPTY"
        if self.provider != "vllm" and (not self._api_key or self._api_key == "EMPTY"):
            log.warning(
                "No API key for provider %s (env=%s) — calls will likely fail.",
                self.provider,
                env_key,
            )
        try:
            from openai import OpenAI
        except Exception as e:
            raise RuntimeError(
                "openai package missing; install with `pip install prophet-bench[api]`"
            ) from e
        self._client = OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=self.timeout_s,
            max_retries=2,
        )

    def _complete(self, prompt: str) -> tuple[str, int, int, float]:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            extra_body=self.extra_body or None,
        )
        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        tok_in = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
        tok_out = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
        cost = estimate_cost(self.name, tok_in, tok_out)
        return text, tok_in, tok_out, cost
