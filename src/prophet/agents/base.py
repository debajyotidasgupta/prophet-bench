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
_CONF_RE = re.compile(r"confidence\s*[:=]\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)
_ANS_RE = re.compile(r"answer\s*:\s*(.*?)(?=\Z|\n[a-zA-Z]+\s*:)", re.DOTALL | re.IGNORECASE)
# Fallbacks for models that don't emit <decision> tags.
_ANS_LINE_RE = re.compile(r"(?:^|\n)\s*answer\s*[:=]\s*([^\n]+)", re.IGNORECASE)
_FINAL_ANSWER_RE = re.compile(r"(?:^|\n)\s*(?:final\s+answer|answer)\s*[:=]\s*([^\n]+)", re.IGNORECASE)
_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def parse_response(task: Task, raw: str) -> AgentResponse:
    """Parse the raw model output into an AgentResponse.

    Robust to several common deviations from the structured-output protocol:
      * decision block missing entirely (use trailing "Answer: ..." line)
      * <think>…</think> blocks present (stripped before extraction)
      * confidence/mode reported on separate lines outside <decision>
      * answer wrapped in \\boxed{…}

    On total parse failure → PASS with confidence 0.5 (logged).
    """
    cleaned = _THINK_RE.sub("", raw)
    m = _DECISION_RE.search(cleaned)
    if m:
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
            answer = ans_m.group(1).strip() or None
        reasoning = cleaned[: m.start()].strip()[-4000:]
        return AgentResponse(
            task_id=task.task_id,
            mode=mode,
            confidence=confidence,
            answer=answer,
            reasoning=reasoning,
        )

    # ---- Fallback: no <decision> block. Try to salvage. ----
    mode_m = _MODE_RE.search(cleaned)
    conf_m = _CONF_RE.search(cleaned)
    ans_m = _FINAL_ANSWER_RE.search(cleaned) or _ANS_LINE_RE.search(cleaned)
    boxed_m = _BOXED_RE.search(cleaned)
    have_signal = bool(mode_m or conf_m or ans_m or boxed_m)
    REFUSAL_PHRASES = (
        "i don't know",
        "i do not know",
        "i cannot",
        "i can't",
        "i won't",
        "i refuse",
        "i'm unable",
        "i am unable",
        "no answer",
        "i'm not sure",
        "not sure",
        "unsure",
        "n/a",
        "unable to answer",
    )
    if not have_signal:
        # last-ditch: take the last non-empty line as the answer (heuristic)
        last_lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
        if last_lines:
            answer = last_lines[-1][:500]
            answer_lower = answer.lower()
            if any(p in answer_lower for p in REFUSAL_PHRASES):
                return AgentResponse(
                    task_id=task.task_id,
                    mode=DecisionMode.PASS,
                    confidence=0.5,
                    answer=None,
                    reasoning=cleaned[-4000:],
                    metadata={"parse_error": "no_decision_block"},
                )
            return AgentResponse(
                task_id=task.task_id,
                mode=DecisionMode.QUOTE,
                confidence=0.5,
                answer=answer,
                reasoning=cleaned[-4000:],
                metadata={"parse_error": "no_decision_block_used_last_line"},
            )
        log.warning("No usable signal on task %s — defaulting to PASS", task.task_id)
        return AgentResponse(
            task_id=task.task_id,
            mode=DecisionMode.PASS,
            confidence=0.5,
            answer=None,
            reasoning=raw[:4000],
            metadata={"parse_error": "no_decision_block"},
        )
    # Reconstruct
    if mode_m:
        try:
            mode = DecisionMode[mode_m.group(1).upper()]
        except KeyError:
            mode = DecisionMode.QUOTE
    else:
        mode = DecisionMode.QUOTE  # default when an answer is detected
    try:
        confidence = float(conf_m.group(1)) if conf_m else 0.7
    except (TypeError, ValueError):
        confidence = 0.7
    confidence = max(0.0, min(1.0, confidence))
    if ans_m:
        answer = ans_m.group(1).strip()
    elif boxed_m:
        answer = boxed_m.group(1).strip()
    else:
        answer = None
    if mode == DecisionMode.PASS:
        answer = None
    return AgentResponse(
        task_id=task.task_id,
        mode=mode,
        confidence=confidence,
        answer=answer,
        reasoning=cleaned[-4000:],
        metadata={"parse_error": "soft_recovery"},
    )


@dataclass
class AgentBase:
    """Convenience base — implement `_complete(prompt) -> (text, tokens_in, tokens_out, cost)` in subclasses."""

    name: str = "agent"
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout_s: float = 120.0
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
