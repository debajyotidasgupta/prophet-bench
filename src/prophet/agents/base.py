"""Base class for PROPHET agents.

Agent adapters do two things:
  1. Send the task prompt + market offer to a model.
  2. Parse the model output into an AgentResponse (mode, confidence, answer).

Output parsing uses a *structured* protocol the model is asked to follow:

    <think> ... (optional reasoning) ... </think>
    <decision>
      mode: TAKE | QUOTE | PASS
      confidence: 0.00 - 1.00
      answer: ...                # required for TAKE / QUOTE
    </decision>

If the model deviates, we attempt graceful fallbacks (regex pulls + JSON
schema retries). A failing-to-parse response is treated as PASS with
confidence 0.5 — this conservatively penalises malformed agents.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from prophet.engine.types import (
    AgentResponse,
    DecisionMode,
    MarketOffer,
    Task,
)

log = logging.getLogger("prophet.agents")


PROMPT_INSTRUCTIONS = """You are participating in PROPHET, a calibration benchmark.

For each task you may choose ONE of:
  - TAKE: commit to attempting. If your answer is correct, you earn V_success; otherwise you pay C_failure.
  - QUOTE: declare your probability of success P̂ in [0,1]. You attempt the task; you earn V·y - C·(1-y) and additionally κ·(1 - 4·(P̂ - y)²), where y is whether you got it right.
  - PASS: decline. You pay only δ_pass.

Always end with EXACTLY this block (no markdown, no extra text after it):

<decision>
mode: <TAKE | QUOTE | PASS>
confidence: <float in [0,1] — your P̂ that your answer would be correct>
answer: <your final answer; empty if PASS>
</decision>

Important:
  • Be honest in your confidence; over-claiming and under-claiming are both penalised.
  • Reasoning is allowed before the <decision> block.
  • Do NOT pad with markdown after </decision>. The parser reads only the first <decision>…</decision>.
"""


def build_prompt(task: Task, offer: MarketOffer) -> str:
    return (
        f"{PROMPT_INSTRUCTIONS}\n"
        f"---\n"
        f"Task family: {task.family}\n"
        f"Market offer: V_success={offer.V_success:.1f}, "
        f"C_failure={offer.C_failure:.1f}, κ={offer.kappa_calib:.1f}, "
        f"δ_pass={offer.delta_pass:.1f}\n"
        f"---\n"
        f"{task.prompt}\n"
        f"---\n"
        f"Respond now."
    )


_DECISION_RE = re.compile(
    r"<decision>\s*(.*?)\s*</decision>", re.DOTALL | re.IGNORECASE
)
_MODE_RE = re.compile(r"mode\s*:\s*(TAKE|QUOTE|PASS)", re.IGNORECASE)
_CONF_RE = re.compile(r"confidence\s*:\s*([0-9]*\.?[0-9]+)")
_ANS_RE = re.compile(r"answer\s*:\s*(.*?)(?=\Z|\n[a-zA-Z]+\s*:)", re.DOTALL | re.IGNORECASE)


def parse_response(task: Task, raw: str) -> AgentResponse:
    """Parse the raw model output into an AgentResponse.

    On parse failure → PASS with confidence 0.5 (logged).
    """
    m = _DECISION_RE.search(raw)
    if not m:
        log.warning("No <decision> block on task %s — defaulting to PASS", task.task_id)
        return AgentResponse(
            task_id=task.task_id,
            mode=DecisionMode.PASS,
            confidence=0.5,
            answer=None,
            reasoning=raw[:4000],
            metadata={"parse_error": "no_decision_block"},
        )
    body = m.group(1)
    mode_m = _MODE_RE.search(body)
    conf_m = _CONF_RE.search(body)
    ans_m = _ANS_RE.search(body)

    mode_str = mode_m.group(1).upper() if mode_m else "PASS"
    try:
        mode = DecisionMode[mode_str]
    except KeyError:
        mode = DecisionMode.PASS

    try:
        confidence = float(conf_m.group(1)) if conf_m else 0.5
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    answer = None
    if mode != DecisionMode.PASS and ans_m:
        answer = ans_m.group(1).strip()
        if not answer:
            answer = None
    # Pre-decision reasoning (everything before the decision block)
    reasoning = raw[: m.start()].strip()[-4000:]
    return AgentResponse(
        task_id=task.task_id,
        mode=mode,
        confidence=confidence,
        answer=answer,
        reasoning=reasoning,
    )


@dataclass(slots=True)
class AgentBase:
    """Convenience base — implement `_complete(prompt) -> (text, tokens_in, tokens_out, cost)` in subclasses."""

    name: str = "agent"
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout_s: float = 60.0
    family_support: set[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def _complete(self, prompt: str) -> tuple[str, int, int, float]:
        raise NotImplementedError

    def respond(self, task: Task, offer: MarketOffer) -> AgentResponse:
        prompt = build_prompt(task, offer)
        t0 = time.time()
        try:
            text, tok_in, tok_out, cost = self._complete(prompt)
        except Exception as e:
            log.exception("Agent %s failed on %s: %s", self.name, task.task_id, e)
            return AgentResponse(
                task_id=task.task_id,
                mode=DecisionMode.PASS,
                confidence=0.5,
                answer=None,
                reasoning=None,
                tokens_in=0,
                tokens_out=0,
                wall_time_s=time.time() - t0,
                cost_usd=0.0,
                metadata={"error": repr(e)},
            )
        elapsed = time.time() - t0
        resp = parse_response(task, text)
        resp.tokens_in = tok_in
        resp.tokens_out = tok_out
        resp.wall_time_s = elapsed
        resp.cost_usd = cost
        return resp
