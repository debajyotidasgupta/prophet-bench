"""vLLM in-process local agent adapter.

Uses the `vllm` Python API directly (`from vllm import LLM, SamplingParams`),
so the model is loaded once into GPU memory and reused across calls within
the same Python process. For an *external* vLLM OpenAI-compatible server, use
`OpenAICompatAgent(provider="vllm", ...)` instead.

The `vllm` package is imported lazily so this module can be imported on
machines that don't have it installed.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

from prophet.agents.base import AgentBase
from prophet.utils.cost import estimate_cost

log = logging.getLogger("prophet.agents.vllm_local")

ENV_KEY = "HUGGINGFACE_TOKEN"


@dataclass(slots=True)
class VLLMLocalAgent(AgentBase):
    """Agent that runs a model locally via the vLLM Python API."""

    model_id: str = "Qwen/Qwen3-7B-Instruct"
    name: str = ""
    temperature: float = 0.0
    max_tokens: int = 512
    timeout_s: float = 300.0
    family_support: set[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # vLLM engine knobs.
    dtype: str = "auto"
    trust_remote_code: bool = True
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 8192
    tensor_parallel_size: int = 1

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"vllm:{self.model_id}"
        if not os.environ.get(ENV_KEY):
            log.info(
                "No HUGGINGFACE_TOKEN set — gated/private models will fail to load."
            )
        try:
            from vllm import LLM
        except Exception as e:  # pragma: no cover - import guard
            raise RuntimeError(
                "vllm package missing; install with `pip install prophet-bench[vllm]`"
            ) from e
        log.info("Loading vLLM engine for %s", self.model_id)
        self._llm = LLM(
            model=self.model_id,
            dtype=self.dtype,
            trust_remote_code=self.trust_remote_code,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            tensor_parallel_size=self.tensor_parallel_size,
        )
        # Reuse vLLM's tokenizer for chat templating + token counting.
        self._tokenizer = self._llm.get_tokenizer()

    def _format_prompt(self, prompt: str) -> str:
        """Use the model's chat template when available, else plain prompt."""
        if getattr(self._tokenizer, "chat_template", None):
            messages = [{"role": "user", "content": prompt}]
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return f"System: You are a helpful assistant.\nUser: {prompt}\nAssistant:"

    def _complete(self, prompt: str) -> tuple[str, int, int, float]:
        from vllm import SamplingParams

        formatted = self._format_prompt(prompt)
        params = SamplingParams(
            temperature=max(self.temperature, 0.0),
            max_tokens=self.max_tokens,
        )
        outputs = self._llm.generate([formatted], params)
        if not outputs:
            return "", 0, 0, 0.0
        result = outputs[0]
        # Prompt token count: prefer the engine's prompt_token_ids when present.
        prompt_token_ids = getattr(result, "prompt_token_ids", None)
        if prompt_token_ids is not None:
            tok_in = len(prompt_token_ids)
        else:  # pragma: no cover - fallback
            tok_in = len(self._tokenizer.encode(formatted))
        completion = result.outputs[0] if result.outputs else None
        if completion is None:
            return "", tok_in, 0, 0.0
        text = completion.text or ""
        token_ids = getattr(completion, "token_ids", None)
        tok_out = len(token_ids) if token_ids is not None else int(
            len(self._tokenizer.encode(text)) if text else 0
        )
        cost = estimate_cost(self.name, tok_in, tok_out)
        return text, tok_in, tok_out, cost
