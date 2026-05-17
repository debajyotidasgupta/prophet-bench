"""Hugging Face Transformers local-inference agent adapter.

Loads a model with `transformers.AutoModelForCausalLM` + `accelerate`'s
`device_map="auto"` and runs greedy / sampled generation in-process. Suitable
for testing local CPU/GPU inference without needing an external server.

Both `torch` and `transformers` are imported lazily so this module can be
imported on machines that don't have them installed — construction will raise
a helpful error when the missing optional dep is actually required.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

from prophet.agents.base import AgentBase
from prophet.utils.cost import estimate_cost

log = logging.getLogger("prophet.agents.transformers")

ENV_KEY = "HUGGINGFACE_TOKEN"


@dataclass(slots=True)
class TransformersAgent(AgentBase):
    """Agent that runs a HF model locally via the `transformers` library."""

    model_id: str = "Qwen/Qwen3-7B-Instruct"
    name: str = ""
    temperature: float = 0.0
    max_tokens: int = 512
    timeout_s: float = 300.0
    family_support: set[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # Optional knobs (left as field defaults to play well with slots=True).
    device_map: str = "auto"
    torch_dtype: str = "auto"
    trust_remote_code: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"hf:{self.model_id}"
        self._token = os.environ.get(ENV_KEY)
        if not self._token:
            log.info(
                "No HUGGINGFACE_TOKEN set — gated/private models will fail to load."
            )
        try:
            import torch  # noqa: F401  - imported to fail-fast if missing
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as e:  # pragma: no cover - import guard
            raise RuntimeError(
                "torch + transformers missing; install with "
                "`pip install prophet-bench[local]`"
            ) from e
        log.info("Loading tokenizer for %s", self.model_id)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            token=self._token,
            trust_remote_code=self.trust_remote_code,
        )
        log.info("Loading model %s (device_map=%s)", self.model_id, self.device_map)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map=self.device_map,
            torch_dtype=self.torch_dtype,
            token=self._token,
            trust_remote_code=self.trust_remote_code,
        )
        # Some tokenizers lack a pad token (causes warnings during batch gen).
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

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
        import torch

        formatted = self._format_prompt(prompt)
        inputs = self._tokenizer(formatted, return_tensors="pt")
        # Move inputs to the model's first device.
        try:
            device = next(self._model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
        except Exception:  # pragma: no cover - device move best-effort
            pass
        tok_in = int(inputs["input_ids"].shape[-1])

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_tokens,
            "do_sample": self.temperature > 0,
            "temperature": max(self.temperature, 1e-5),
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **gen_kwargs)
        # Strip prompt prefix so we only decode generated tokens.
        gen_ids = output_ids[0, tok_in:]
        tok_out = int(gen_ids.shape[-1])
        text = self._tokenizer.decode(gen_ids, skip_special_tokens=True)
        cost = estimate_cost(self.name, tok_in, tok_out)
        return text, tok_in, tok_out, cost
