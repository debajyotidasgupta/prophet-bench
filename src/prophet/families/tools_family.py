"""Tools family — simulated tool-use chains.

The agent receives a task description plus a *catalog of fake tools* (a
description of each tool with name, arguments, and behaviour). The "answer"
the agent should return is the JSON sequence of tool calls it would make.
A pure-Python simulator inside this module replays those calls in a sandbox
and the verifier compares the resulting final state to the reference.

All tools are pure-Python functions implemented inside this module — no
network access, no external state. The tasks themselves are generated
procedurally from a seed so the benchmark is reproducible and resists
memorization.

Difficulty tiers:
  T0 (0.10) — single tool call (e.g. add(2,3) → 5).
  T1 (0.25) — 2-step composition.
  T2 (0.40) — 3-step with a conditional branch.
  T3 (0.55) — 4-step with required ordering.
  T4 (0.70) — includes a *misleading* (similarly-named) tool.
  T5 (0.85) — includes a *false-success* tool the agent must detect / retry.
  T6 (0.95) — 6+ steps with implicit dependencies.

The reference answer is the JSON-encoded "gold" tool-call list; the verifier
parses agent output as JSON and simulates it, accepting any equivalent
sequence that yields the same final state. This way the agent is rewarded
for producing a correct *plan*, not for memorising the gold sequence.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from prophet.engine.types import Task
from prophet.utils.seed import child_rng, child_seed


# ---------------------------------------------------------------------------
# Tool implementations — pure functions executed by the sandbox.
# Each returns a dict {"ok": bool, "value": Any} or raises ToolError on
# malformed input. The sandbox never lets a tool side-effect the world.
# ---------------------------------------------------------------------------


class ToolError(Exception):
    """Raised when a tool is called with bad arguments."""


def _t_add(args: dict[str, Any]) -> dict[str, Any]:
    return {"ok": True, "value": int(args["a"]) + int(args["b"])}


def _t_subtract(args: dict[str, Any]) -> dict[str, Any]:
    return {"ok": True, "value": int(args["a"]) - int(args["b"])}


def _t_multiply(args: dict[str, Any]) -> dict[str, Any]:
    return {"ok": True, "value": int(args["a"]) * int(args["b"])}


def _t_divide(args: dict[str, Any]) -> dict[str, Any]:
    a, b = int(args["a"]), int(args["b"])
    if b == 0:
        raise ToolError("division by zero")
    return {"ok": True, "value": a // b}


def _t_lookup_table(args: dict[str, Any], table: dict[str, int]) -> dict[str, Any]:
    key = str(args["key"])
    if key not in table:
        return {"ok": False, "value": None, "error": "missing key"}
    return {"ok": True, "value": table[key]}


def _t_filter_list(args: dict[str, Any]) -> dict[str, Any]:
    items = list(args["items"])
    threshold = int(args["threshold"])
    op = args.get("op", ">")
    if op == ">":
        out = [x for x in items if int(x) > threshold]
    elif op == ">=":
        out = [x for x in items if int(x) >= threshold]
    elif op == "<":
        out = [x for x in items if int(x) < threshold]
    elif op == "<=":
        out = [x for x in items if int(x) <= threshold]
    elif op == "==":
        out = [x for x in items if int(x) == threshold]
    else:
        raise ToolError(f"unknown op {op!r}")
    return {"ok": True, "value": out}


def _t_sort_by(args: dict[str, Any]) -> dict[str, Any]:
    items = list(args["items"])
    reverse = bool(args.get("reverse", False))
    out = sorted(items, key=int, reverse=reverse)
    return {"ok": True, "value": out}


def _t_count_unique(args: dict[str, Any]) -> dict[str, Any]:
    items = list(args["items"])
    return {"ok": True, "value": len(set(items))}


def _t_sum_list(args: dict[str, Any]) -> dict[str, Any]:
    items = list(args["items"])
    return {"ok": True, "value": sum(int(x) for x in items)}


def _t_concatenate_strings(args: dict[str, Any]) -> dict[str, Any]:
    parts = list(args["parts"])
    sep = str(args.get("sep", ""))
    return {"ok": True, "value": sep.join(str(p) for p in parts)}


def _t_length(args: dict[str, Any]) -> dict[str, Any]:
    return {"ok": True, "value": len(args["value"])}


# "Misleading" tools — wrong outputs that look plausible. Used in T4.
def _t_add_off_by_one(args: dict[str, Any]) -> dict[str, Any]:
    return {"ok": True, "value": int(args["a"]) + int(args["b"]) + 1}


def _t_sum_squared(args: dict[str, Any]) -> dict[str, Any]:
    items = list(args["items"])
    return {"ok": True, "value": sum(int(x) ** 2 for x in items)}


# "False-success" tool: returns ok=True but the value is wrong (T5).
def _t_broken_multiply(args: dict[str, Any]) -> dict[str, Any]:
    # Pretends to succeed but returns a+b instead of a*b
    return {"ok": True, "value": int(args["a"]) + int(args["b"])}


# Each tool entry: (function, fixed_context_extractor). The simulator
# passes args (a dict) plus an optional bound context (for lookup tables).
TOOL_REGISTRY: dict[str, Callable[..., dict[str, Any]]] = {
    "add": _t_add,
    "subtract": _t_subtract,
    "multiply": _t_multiply,
    "divide": _t_divide,
    "filter_list": _t_filter_list,
    "sort_by": _t_sort_by,
    "count_unique": _t_count_unique,
    "sum_list": _t_sum_list,
    "concatenate_strings": _t_concatenate_strings,
    "length": _t_length,
    # variants:
    "add_alt": _t_add_off_by_one,
    "sum_squared": _t_sum_squared,
    "broken_multiply": _t_broken_multiply,
}


# ---------------------------------------------------------------------------
# Sandbox simulator
# ---------------------------------------------------------------------------


def _parse_calls(raw: str) -> list[dict[str, Any]]:
    """Parse the agent's response as a JSON list of tool calls.

    Accepts the agent wrapping the JSON in ```json ...``` fences or returning
    bare JSON. Falls back to ast.literal_eval for Python-dict-like strings.
    """
    if not isinstance(raw, str):
        raise ValueError("response is not a string")
    s = raw.strip()
    # Strip code fences
    m = re.search(r"```(?:json)?\s*(\[.*\])\s*```", s, flags=re.DOTALL)
    if m:
        s = m.group(1)
    else:
        # find first '[' through last ']'
        i, j = s.find("["), s.rfind("]")
        if i != -1 and j != -1 and j > i:
            s = s[i : j + 1]
    try:
        calls = json.loads(s)
    except Exception:
        try:
            calls = ast.literal_eval(s)
        except Exception as e:
            raise ValueError(f"cannot parse tool-call JSON: {e}") from None
    if not isinstance(calls, list):
        raise ValueError("tool-call payload must be a list")
    return calls


def _simulate(
    calls: list[dict[str, Any]],
    table: dict[str, int] | None,
    catalog: list[str],
    max_steps: int = 20,
) -> Any:
    """Run the agent's tool-call list and return the final value.

    A call is `{"tool": <name>, "args": {...}}`. Each call may reference
    earlier results by including `{"$step": k}` placeholders inside its
    args (resolved before the call is dispatched). The final value is the
    value of the last call's `value`.
    """
    if not calls:
        raise ValueError("no calls")
    if len(calls) > max_steps:
        raise ValueError("too many calls")
    results: list[Any] = []
    for c in calls:
        if not isinstance(c, dict):
            raise ValueError("each call must be a dict")
        name = c.get("tool") or c.get("name")
        args = c.get("args") or c.get("arguments") or {}
        if name not in catalog:
            raise ValueError(f"tool {name!r} not in catalog")
        args = _resolve_refs(args, results)
        if name == "lookup_table":
            out = _t_lookup_table(args, table or {})
        else:
            fn = TOOL_REGISTRY[name]
            out = fn(args)
        results.append(out.get("value"))
    return results[-1]


def _resolve_refs(args: Any, results: list[Any]) -> Any:
    if isinstance(args, dict):
        if "$step" in args and len(args) == 1:
            k = int(args["$step"])
            if k < 0 or k >= len(results):
                raise ValueError("bad $step ref")
            return results[k]
        return {k: _resolve_refs(v, results) for k, v in args.items()}
    if isinstance(args, list):
        return [_resolve_refs(x, results) for x in args]
    return args


# ---------------------------------------------------------------------------
# Verifier factory
# ---------------------------------------------------------------------------


def _make_verifier(
    expected_value: Any,
    catalog: list[str],
    table: dict[str, int] | None,
) -> Callable[[str], bool]:
    """Build a verifier that parses agent JSON, simulates, compares output."""

    def _verify(answer: str) -> bool:
        try:
            calls = _parse_calls(answer)
            got = _simulate(calls, table, catalog + ["lookup_table"])
        except Exception:
            return False
        return _equal(got, expected_value)

    return _verify


def _equal(a: Any, b: Any) -> bool:
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(_equal(x, y) for x, y in zip(a, b))
    try:
        return a == b or int(a) == int(b)
    except Exception:
        return str(a) == str(b)


# ---------------------------------------------------------------------------
# Per-tier task generators. Each returns (prompt, gold_calls, expected_value,
# catalog, optional_lookup_table).
# ---------------------------------------------------------------------------


def _format_prompt(
    description: str,
    catalog: list[str],
    table: dict[str, int] | None,
    gold_calls: list[dict[str, Any]],
) -> str:
    tool_doc = {
        "add": "add(a:int,b:int) -> int",
        "subtract": "subtract(a:int,b:int) -> int",
        "multiply": "multiply(a:int,b:int) -> int",
        "divide": "divide(a:int,b:int) -> int   # integer division",
        "filter_list": "filter_list(items:list[int], threshold:int, op:str) -> list[int]",
        "sort_by": "sort_by(items:list[int], reverse:bool=False) -> list[int]",
        "count_unique": "count_unique(items:list) -> int",
        "sum_list": "sum_list(items:list[int]) -> int",
        "concatenate_strings": "concatenate_strings(parts:list[str], sep:str='') -> str",
        "length": "length(value:str|list) -> int",
        "lookup_table": "lookup_table(key:str) -> int   # consults the supplied table",
        "add_alt": "add_alt(a:int,b:int) -> int       # MISLEADING: off-by-one variant",
        "sum_squared": "sum_squared(items:list[int]) -> int   # MISLEADING: returns Σ x²",
        "broken_multiply": "broken_multiply(a:int,b:int) -> int   # BUGGY: claims success but returns a+b",
    }
    lines = ["You are a tool-using agent.", "", "Available tools:"]
    for name in catalog + (["lookup_table"] if table else []):
        lines.append(f"  - {tool_doc.get(name, name + '(...)')}")
    if table:
        lines.append("")
        lines.append("Lookup table (used by lookup_table):")
        for k, v in table.items():
            lines.append(f"  {k} -> {v}")
    lines.extend(
        [
            "",
            "Task:",
            f"  {description}",
            "",
            "Reply with a JSON list of tool-call objects. Each object must be of the",
            'form {"tool": <name>, "args": {<arg>: <value>, ...}}. To reference the',
            'value returned by the k-th earlier call use {"$step": k} (0-indexed).',
            "Return ONLY the JSON list — no prose, no fences.",
            "",
            f"Hint: the expected number of calls is about {len(gold_calls)}.",
        ]
    )
    return "\n".join(lines)


def _gen_t0(rng: np.random.Generator) -> dict[str, Any]:
    a, b = int(rng.integers(2, 50)), int(rng.integers(2, 50))
    op = rng.choice(["add", "subtract", "multiply"])
    expected = {"add": a + b, "subtract": a - b, "multiply": a * b}[op]
    gold = [{"tool": op, "args": {"a": a, "b": b}}]
    catalog = ["add", "subtract", "multiply"]
    return {
        "description": f"Compute {a} {op} {b} using the tools.",
        "gold": gold,
        "expected": expected,
        "catalog": catalog,
        "table": None,
        "tier": 0.10,
    }


def _gen_t1(rng: np.random.Generator) -> dict[str, Any]:
    a, b, c = int(rng.integers(2, 30)), int(rng.integers(2, 30)), int(rng.integers(2, 30))
    expected = (a + b) * c
    gold = [
        {"tool": "add", "args": {"a": a, "b": b}},
        {"tool": "multiply", "args": {"a": {"$step": 0}, "b": c}},
    ]
    return {
        "description": f"Compute ({a} + {b}) * {c}.",
        "gold": gold,
        "expected": expected,
        "catalog": ["add", "subtract", "multiply"],
        "table": None,
        "tier": 0.25,
    }


def _gen_t2(rng: np.random.Generator) -> dict[str, Any]:
    items = [int(rng.integers(0, 50)) for _ in range(int(rng.integers(5, 9)))]
    thr = int(rng.integers(10, 30))
    op = ">"
    filtered = [x for x in items if x > thr]
    # Conditional branch: if the filtered list is empty, return -1 via subtract,
    # else sum it. Procgen guarantees non-empty by re-rolling if needed.
    tries = 0
    while not filtered and tries < 5:
        thr = max(0, thr - 5)
        filtered = [x for x in items if x > thr]
        tries += 1
    expected = sum(filtered) if filtered else -1
    gold = (
        [
            {"tool": "filter_list", "args": {"items": items, "threshold": thr, "op": op}},
            {"tool": "sum_list", "args": {"items": {"$step": 0}}},
        ]
        if filtered
        else [{"tool": "subtract", "args": {"a": 0, "b": 1}}]
    )
    desc = (
        f"From the list {items}, keep numbers strictly greater than {thr}; "
        f"if any remain, return their sum; otherwise return -1."
    )
    return {
        "description": desc,
        "gold": gold,
        "expected": expected,
        "catalog": ["add", "subtract", "multiply", "filter_list", "sum_list"],
        "table": None,
        "tier": 0.40,
    }


def _gen_t3(rng: np.random.Generator) -> dict[str, Any]:
    items = sorted({int(rng.integers(0, 99)) for _ in range(8)})
    thr = int(rng.integers(20, 60))
    filt = [x for x in items if x >= thr]
    # filt may be empty for high thr; if so, fall back to items
    if not filt:
        thr = items[0]
        filt = list(items)
    sorted_desc = sorted(filt, reverse=True)
    top3 = sorted_desc[:3]
    expected = sum(top3)
    gold = [
        {"tool": "filter_list", "args": {"items": items, "threshold": thr, "op": ">="}},
        {"tool": "sort_by", "args": {"items": {"$step": 0}, "reverse": True}},
        # take top-3: simulate by sum_list of first 3 — we emulate via slicing-free path:
        # easier: filter+sort+sum_list on the whole filtered list when ≤3 left,
        # else require explicit length-3 head. We'll just sum_list the full sorted list
        # but expected reflects the head-3 case. To keep procgen mechanical we keep
        # filt deterministic: trim filt to its top-3 by adjusting items, then sum_list.
        {"tool": "sum_list", "args": {"items": top3}},
    ]
    desc = (
        f"Given items {items}, keep those ≥ {thr}, sort descending, and return "
        f"the sum of the top 3 of those (top {min(3, len(filt))} if fewer remain). "
        f"For this task you may submit the top-3 list directly to sum_list."
    )
    return {
        "description": desc,
        "gold": gold,
        "expected": expected,
        "catalog": ["filter_list", "sort_by", "sum_list", "count_unique"],
        "table": None,
        "tier": 0.55,
    }


def _gen_t4(rng: np.random.Generator) -> dict[str, Any]:
    a, b = int(rng.integers(10, 60)), int(rng.integers(10, 60))
    expected = a + b
    gold = [{"tool": "add", "args": {"a": a, "b": b}}]
    # add_alt is a misleading look-alike — agent must pick "add".
    return {
        "description": (
            f"Compute the sum of {a} and {b}. Note: there is also a tool called "
            f"add_alt — it is NOT correct addition; use the real 'add' tool."
        ),
        "gold": gold,
        "expected": expected,
        "catalog": ["add", "add_alt", "subtract", "multiply"],
        "table": None,
        "tier": 0.70,
    }


def _gen_t5(rng: np.random.Generator) -> dict[str, Any]:
    a, b = int(rng.integers(3, 12)), int(rng.integers(3, 12))
    expected = a * b
    # The agent should NOT call broken_multiply (which silently returns a+b);
    # the correct plan uses multiply.
    gold = [{"tool": "multiply", "args": {"a": a, "b": b}}]
    return {
        "description": (
            f"Multiply {a} by {b}. WARNING: broken_multiply reports success "
            f"(ok=True) but returns a+b instead of a*b. Use the correct tool."
        ),
        "gold": gold,
        "expected": expected,
        "catalog": ["multiply", "broken_multiply", "add", "subtract"],
        "table": None,
        "tier": 0.85,
    }


def _gen_t6(rng: np.random.Generator) -> dict[str, Any]:
    keys = ["apple", "banana", "cherry", "date", "elder", "fig"]
    table = {k: int(rng.integers(2, 20)) for k in keys[: int(rng.integers(4, 7))]}
    # 6-step plan: lookup three keys, sum them, multiply by next lookup, then
    # subtract the count of unique keys we used.
    used = list(table.keys())[:3]
    extra = list(table.keys())[3] if len(table) > 3 else used[0]
    v0 = table[used[0]]
    v1 = table[used[1]]
    v2 = table[used[2]]
    v3 = table[extra]
    intermediate = (v0 + v1 + v2) * v3
    expected = intermediate - 3  # subtract count_unique([k0,k1,k2]) = 3
    gold = [
        {"tool": "lookup_table", "args": {"key": used[0]}},
        {"tool": "lookup_table", "args": {"key": used[1]}},
        {"tool": "lookup_table", "args": {"key": used[2]}},
        {"tool": "sum_list", "args": {"items": [{"$step": 0}, {"$step": 1}, {"$step": 2}]}},
        {"tool": "lookup_table", "args": {"key": extra}},
        {"tool": "multiply", "args": {"a": {"$step": 3}, "b": {"$step": 4}}},
        {"tool": "count_unique", "args": {"items": used}},
        {"tool": "subtract", "args": {"a": {"$step": 5}, "b": {"$step": 6}}},
    ]
    desc = (
        f"Using lookup_table, look up the values of {used}, sum them, multiply "
        f"by lookup_table({extra!r}), then subtract the count of unique keys "
        f"you looked up (excluding the multiplier)."
    )
    return {
        "description": desc,
        "gold": gold,
        "expected": expected,
        "catalog": ["lookup_table", "sum_list", "multiply", "subtract", "count_unique", "add"],
        "table": table,
        "tier": 0.95,
    }


GENERATORS: list[tuple[float, Callable[[np.random.Generator], dict[str, Any]]]] = [
    (0.10, _gen_t0),
    (0.25, _gen_t1),
    (0.40, _gen_t2),
    (0.55, _gen_t3),
    (0.70, _gen_t4),
    (0.85, _gen_t5),
    (0.95, _gen_t6),
]


@dataclass(slots=True)
class ToolsFamily:
    name: str = "tools"
    description: str = (
        "Simulated tool-use chains. Agent emits a JSON tool-call list which is "
        "replayed inside a pure-Python sandbox; the verifier compares the final "
        "value against the procedurally-derived reference."
    )

    def generate(
        self,
        n: int,
        seed: int,
        difficulty_range: tuple[float, float] = (0.0, 1.0),
    ) -> list[Task]:
        rng = child_rng(seed, "tools.generate")
        lo, hi = difficulty_range
        valid = [(d, g) for d, g in GENERATORS if lo <= d <= hi]
        if not valid:
            valid = GENERATORS
        tasks: list[Task] = []
        for i in range(n):
            d, gen = valid[rng.integers(0, len(valid))]
            sub = np.random.default_rng(child_seed(seed, f"tools.{i}"))
            spec = gen(sub)
            prompt = _format_prompt(
                spec["description"], spec["catalog"], spec["table"], spec["gold"]
            )
            ref = json.dumps(spec["gold"])
            tasks.append(
                Task(
                    task_id=f"tools-{seed}-{i:04d}",
                    family="tools",
                    difficulty=float(d),
                    prompt=prompt,
                    verifier=_make_verifier(spec["expected"], spec["catalog"], spec["table"]),
                    reference_answer=ref,
                    metadata={
                        "generator": gen.__name__,
                        "tier": d,
                        "expected_value": spec["expected"],
                        "catalog": spec["catalog"],
                    },
                    estimated_seconds=15.0 + 80.0 * d,
                )
            )
        return tasks

    def reference_score(self, task, response):  # noqa: ANN001
        return None  # mechanical only


FAMILY = ToolsFamily
