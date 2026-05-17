"""Agent adapters: open + closed model wrappers, plus a deterministic baseline.

Adapters implement the Agent protocol from `prophet.engine.types`. Each adapter
should report token usage and per-call cost so the orchestrator can enforce
budget guardrails.

URI scheme for the CLI:
  openai:gpt-5-mini
  anthropic:claude-opus-4-7
  google:gemini-3.1-pro
  vllm:Qwen/Qwen3-7B-Instruct          (local vLLM server)
  hf:Qwen/Qwen3-7B-Instruct            (transformers via accelerate)
  hf-inference:Qwen/Qwen3-7B           (HF Inference API)
  together:meta-llama/Llama-4-70B
  baseline:random
  baseline:always-take
  baseline:always-pass
  baseline:oracle                       (cheats; reference for upper bound)
"""

from __future__ import annotations

from prophet.agents.baselines import (
    AlwaysPassAgent,
    AlwaysTakeAgent,
    OracleAgent,
    RandomAgent,
)
from prophet.agents.base import AgentBase

__all__ = [
    "AgentBase",
    "AlwaysPassAgent",
    "AlwaysTakeAgent",
    "OracleAgent",
    "RandomAgent",
    "build_agent",
]


def build_agent(uri: str, **kwargs):  # noqa: ANN201 — Agent protocol
    """Construct an agent from a URI string.

    Lazy-imports backend modules so missing optional deps don't break basic
    usage (e.g. an `openai` import failure shouldn't prevent `baseline:random`
    from working).
    """
    if ":" not in uri:
        raise ValueError(f"Agent URI must include ':' — got {uri!r}")
    scheme, identifier = uri.split(":", 1)
    scheme = scheme.lower()

    if scheme == "baseline":
        if identifier == "random":
            return RandomAgent(**kwargs)
        if identifier == "always-take":
            return AlwaysTakeAgent(**kwargs)
        if identifier == "always-pass":
            return AlwaysPassAgent(**kwargs)
        if identifier == "oracle":
            return OracleAgent(**kwargs)
        raise ValueError(f"Unknown baseline {identifier!r}")

    if scheme in {"openai", "deepinfra", "together", "groq", "fireworks", "hf-inference", "openrouter", "vllm"}:
        from prophet.agents.openai_compat import OpenAICompatAgent
        return OpenAICompatAgent(model=identifier, provider=scheme, **kwargs)

    if scheme == "anthropic":
        from prophet.agents.anthropic_agent import AnthropicAgent
        return AnthropicAgent(model=identifier, **kwargs)

    if scheme == "google":
        from prophet.agents.google_agent import GoogleAgent
        return GoogleAgent(model=identifier, **kwargs)

    if scheme == "hf":
        from prophet.agents.transformers_agent import TransformersAgent
        return TransformersAgent(model_id=identifier, **kwargs)

    raise ValueError(f"Unknown agent scheme {scheme!r}")
