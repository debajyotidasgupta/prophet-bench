"""Reasoning family — abstract reasoning, deduction, and ARC-AGI-2-style grids.

We mix logical / arithmetic puzzles with small grid-transform problems.
For grid tasks, the agent answers in JSON:

    {"output": [[1,2,3], [4,5,6], ...]}

and the verifier parses-and-compares as nested lists. For non-grid tasks
the verifier is a normalized synonym / numeric match.

Difficulty tiers:
  T0 (0.10) — number sequence completion ("2,4,6,?,10").
  T1 (0.20) — symbol substitution cipher ("A→1, B→2, C→3 ; CBA = ?").
  T2 (0.30) — small grid recolor: replace one fixed colour with another.
  T3 (0.45) — logical deduction (transitive ordering).
  T4 (0.60) — grid transform: rotate 90, reflect, transpose, count.
  T5 (0.75) — multi-step deduction with red-herring constraints.
  T6 (0.90) — novel-rule discovery: two input→output examples, derive rule,
              apply to a third grid.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable

import numpy as np

from prophet.engine.types import Task
from prophet.utils.seed import child_rng, child_seed


# -------------------------------------------------------------------------
# Verifier helpers
# -------------------------------------------------------------------------

def _normalize(s: str) -> str:
    s = s.strip()
    s = re.sub(r"```(?:json)?\s*", "", s)
    s = s.replace("```", "")
    return s.strip().rstrip(".,;:")


def _exact_text(expected: str) -> Callable[[str], bool]:
    norm = _normalize(expected).lower()

    def _verify(answer: str) -> bool:
        if not isinstance(answer, str):
            return False
        cand = _normalize(answer).lower()
        if cand == norm:
            return True
        # Allow trailing extra words ("the answer is X")
        return cand.endswith(" " + norm) or cand.endswith(norm)

    return _verify


def _numeric_match(expected: int) -> Callable[[str], bool]:
    def _verify(answer: str) -> bool:
        if not isinstance(answer, str):
            return False
        m = re.findall(r"-?\d+", answer.replace(",", ""))
        if not m:
            return False
        # Prefer the last numeric token
        for tok in reversed(m):
            try:
                if int(tok) == expected:
                    return True
            except Exception:
                continue
        return False
    return _verify


def _grid_match(expected_grid: list[list[int]]) -> Callable[[str], bool]:
    """Parse JSON-ish output and compare to expected nested-list grid."""
    def _verify(answer: str) -> bool:
        if not isinstance(answer, str):
            return False
        s = _normalize(answer)
        # Find first JSON object; greedy match between the first '{' and last '}'
        first = s.find("{")
        last = s.rfind("}")
        candidates: list[str] = []
        if first != -1 and last != -1 and last > first:
            candidates.append(s[first:last + 1])
        # Also try a bracketed list directly
        lb = s.find("[")
        rb = s.rfind("]")
        if lb != -1 and rb != -1 and rb > lb:
            candidates.append(s[lb:rb + 1])
        candidates.append(s)
        for cand in candidates:
            try:
                parsed = json.loads(cand)
            except Exception:
                continue
            grid = parsed["output"] if isinstance(parsed, dict) and "output" in parsed else parsed
            if not isinstance(grid, list):
                continue
            try:
                # coerce all to int
                norm_grid = [[int(x) for x in row] for row in grid]
            except Exception:
                continue
            if norm_grid == expected_grid:
                return True
        return False

    return _verify


# -------------------------------------------------------------------------
# T0 — number sequence completion
# -------------------------------------------------------------------------

def _gen_t0_sequence(rng: np.random.Generator) -> tuple[str, str, Callable[[str], bool]]:
    kind = int(rng.integers(0, 3))
    if kind == 0:
        # Arithmetic
        a0 = int(rng.integers(1, 20))
        d = int(rng.integers(2, 8))
        seq = [a0 + d * i for i in range(5)]
        miss = int(rng.integers(2, 4))  # mask position 2 or 3
        ans = seq[miss]
        display = [str(x) if i != miss else "?" for i, x in enumerate(seq)]
        prompt = (
            f"Complete the sequence by replacing '?' with the correct integer:\n"
            f"  {', '.join(display)}\nReturn one integer."
        )
        return prompt, str(ans), _numeric_match(ans)
    if kind == 1:
        # Geometric (small ratio so values stay tractable)
        a0 = int(rng.integers(1, 5))
        r = int(rng.integers(2, 4))
        seq = [a0 * (r ** i) for i in range(5)]
        miss = int(rng.integers(2, 4))
        ans = seq[miss]
        display = [str(x) if i != miss else "?" for i, x in enumerate(seq)]
        prompt = (
            f"Complete the sequence (each term is multiplied by a fixed integer):\n"
            f"  {', '.join(display)}\nReturn one integer."
        )
        return prompt, str(ans), _numeric_match(ans)
    # Squares
    start = int(rng.integers(1, 8))
    seq = [(start + i) ** 2 for i in range(5)]
    miss = int(rng.integers(2, 4))
    ans = seq[miss]
    display = [str(x) if i != miss else "?" for i, x in enumerate(seq)]
    prompt = (
        f"Complete the sequence (each term is a perfect square):\n"
        f"  {', '.join(display)}\nReturn one integer."
    )
    return prompt, str(ans), _numeric_match(ans)


# -------------------------------------------------------------------------
# T1 — symbol substitution
# -------------------------------------------------------------------------

def _gen_t1_substitution(rng: np.random.Generator) -> tuple[str, str, Callable[[str], bool]]:
    # Map A..F to k..k+5 in some permutation, then ask for a 3-letter word's digits
    letters = list("ABCDEF")
    perm = list(rng.permutation(np.arange(1, 7)))
    mapping = dict(zip(letters, [int(v) for v in perm]))
    # pick a 3-letter "word" from these letters
    word_idx = rng.choice(len(letters), size=3, replace=False)
    word = "".join(letters[int(i)] for i in word_idx)
    digits = "".join(str(mapping[c]) for c in word)
    mapping_str = ", ".join(f"{c}→{mapping[c]}" for c in letters)
    prompt = (
        f"Given the substitution cipher {mapping_str}, what digit string corresponds "
        f"to the word \"{word}\"? Reply with the digit string only (e.g. '123')."
    )
    return prompt, digits, _exact_text(digits)


# -------------------------------------------------------------------------
# T2 — simple grid recolor
# -------------------------------------------------------------------------

def _random_grid(rng: np.random.Generator, h: int, w: int, palette: list[int]) -> list[list[int]]:
    return [[palette[int(rng.integers(0, len(palette)))] for _ in range(w)] for _ in range(h)]


def _gen_t2_recolor(rng: np.random.Generator) -> tuple[str, str, Callable[[str], bool]]:
    h, w = 3, 3
    palette = [0, 1, 2]
    grid = _random_grid(rng, h, w, palette)
    # Guarantee at least one cell of source value
    src = int(rng.integers(0, len(palette)))
    tgt = int(rng.integers(0, len(palette)))
    while tgt == src:
        tgt = int(rng.integers(0, len(palette)))
    grid[0][0] = src
    out = [[(tgt if v == src else v) for v in row] for row in grid]
    prompt = (
        f"Rule: replace every cell with value {src} by value {tgt}; leave all other "
        f"cells unchanged.\n\nInput grid (as JSON):\n"
        f"{json.dumps(grid)}\n\n"
        f'Reply with a JSON object of the form {{"output": [[...], [...], ...]}} '
        f"containing the resulting grid."
    )
    return prompt, json.dumps({"output": out}), _grid_match(out)


# -------------------------------------------------------------------------
# T3 — logical deduction (transitive ordering)
# -------------------------------------------------------------------------

def _gen_t3_deduction(rng: np.random.Generator) -> tuple[str, str, Callable[[str], bool]]:
    names = ["Alice", "Bob", "Carol", "Dan", "Eve", "Finn"]
    chosen = list(rng.choice(names, size=4, replace=False))
    # Random permutation = ranking from tallest [0] to shortest [3]
    rank = list(rng.permutation(chosen))
    # Build statements
    statements = [
        f"{rank[0]} is taller than {rank[1]}.",
        f"{rank[1]} is taller than {rank[2]}.",
        f"{rank[2]} is taller than {rank[3]}.",
    ]
    rng.shuffle(statements)
    question_kind = int(rng.integers(0, 2))
    if question_kind == 0:
        ans = rank[-1]
        q = "Who is shortest?"
    else:
        ans = rank[0]
        q = "Who is tallest?"
    prompt = (
        "Given the following statements:\n"
        + "\n".join(f"  - {s}" for s in statements)
        + f"\n\n{q} Reply with a single name."
    )
    return prompt, ans, _exact_text(ans)


# -------------------------------------------------------------------------
# T4 — grid transform (rotate / reflect / count)
# -------------------------------------------------------------------------

def _rotate_cw(grid: list[list[int]]) -> list[list[int]]:
    h = len(grid)
    w = len(grid[0]) if h else 0
    return [[grid[h - 1 - r][c] for r in range(h)] for c in range(w)]


def _reflect_h(grid: list[list[int]]) -> list[list[int]]:
    return [list(reversed(row)) for row in grid]


def _transpose(grid: list[list[int]]) -> list[list[int]]:
    h = len(grid)
    w = len(grid[0]) if h else 0
    return [[grid[r][c] for r in range(h)] for c in range(w)]


def _gen_t4_transform(rng: np.random.Generator) -> tuple[str, str, Callable[[str], bool]]:
    h = int(rng.integers(3, 5))
    w = int(rng.integers(3, 5))
    palette = [0, 1, 2, 3]
    grid = _random_grid(rng, h, w, palette)
    kind = int(rng.integers(0, 4))
    if kind == 0:
        out = _rotate_cw(grid)
        rule = "Rotate the grid 90 degrees clockwise."
    elif kind == 1:
        out = _reflect_h(grid)
        rule = "Reflect the grid horizontally (left-right mirror)."
    elif kind == 2:
        out = _transpose(grid)
        rule = "Transpose the grid (swap rows and columns)."
    else:
        # Recolor: swap two values
        a, b = 1, 2
        out = [[(b if v == a else (a if v == b else v)) for v in row] for row in grid]
        rule = f"Swap colours {a} and {b}: every {a} becomes {b} and every {b} becomes {a}."
    prompt = (
        f"Rule: {rule}\n\nInput grid (as JSON):\n{json.dumps(grid)}\n\n"
        f'Reply with a JSON object of the form {{"output": [[...], [...]]}} '
        f"containing the resulting grid."
    )
    return prompt, json.dumps({"output": out}), _grid_match(out)


# -------------------------------------------------------------------------
# T5 — multi-step deduction with red-herrings
# -------------------------------------------------------------------------

def _gen_t5_multistep(rng: np.random.Generator) -> tuple[str, str, Callable[[str], bool]]:
    names = ["Alice", "Bob", "Carol", "Dan"]
    fruits = ["apple", "banana", "cherry", "date"]
    rng.shuffle(names)
    rng.shuffle(fruits)
    # The truth: names[i] likes fruits[i]
    mapping = dict(zip(names, fruits))
    # Build 3 true statements + 1 red-herring (about colors)
    true_stmts = [
        f"{n} likes the {f}." for n, f in mapping.items()
    ]
    rng.shuffle(true_stmts)
    true_stmts = true_stmts[:3]  # leave one to deduce
    herring = (
        f"The walls of the cafeteria are painted in two colours; "
        f"this fact is not related to anyone's preferences."
    )
    statements = true_stmts + [herring]
    rng.shuffle(statements)
    # Pick someone to ask about whose preference is NOT in true_stmts
    used_names = set()
    for s in true_stmts:
        for n in names:
            if n in s:
                used_names.add(n)
                break
    missing = [n for n in names if n not in used_names]
    target = missing[0] if missing else names[-1]
    ans = mapping[target]
    prompt = (
        "Each of the following people prefers a different fruit, drawn from "
        "{apple, banana, cherry, date}. Given:\n"
        + "\n".join(f"  - {s}" for s in statements)
        + f"\n\nWhich fruit does {target} prefer? Reply with one word."
    )
    return prompt, ans, _exact_text(ans)


# -------------------------------------------------------------------------
# T6 — novel-rule discovery (2 examples → apply to 3rd)
# -------------------------------------------------------------------------

def _gen_t6_rule_discovery(rng: np.random.Generator) -> tuple[str, str, Callable[[str], bool]]:
    # Rule choices: shift-down-by-1, double-each-value, mirror-vertically
    rules = [
        ("shift colors by +1 (mod 4)", lambda g: [[(v + 1) % 4 for v in row] for row in g]),
        ("reflect horizontally", _reflect_h),
        ("rotate 90 degrees clockwise", _rotate_cw),
        ("swap colors 0 and 1", lambda g: [[(1 if v == 0 else (0 if v == 1 else v)) for v in row] for row in g]),
    ]
    rule_idx = int(rng.integers(0, len(rules)))
    _, fn = rules[rule_idx]
    palette = [0, 1, 2, 3]
    # Two example pairs (3x3) + 1 test grid (3x3)
    ex1 = _random_grid(rng, 3, 3, palette)
    ex2 = _random_grid(rng, 3, 3, palette)
    test = _random_grid(rng, 3, 3, palette)
    out_test = fn(test)
    prompt = (
        "Two input → output examples are shown below. Discover the underlying rule, "
        "then apply it to the test input.\n\n"
        f"Example 1:\n  Input:  {json.dumps(ex1)}\n  Output: {json.dumps(fn(ex1))}\n\n"
        f"Example 2:\n  Input:  {json.dumps(ex2)}\n  Output: {json.dumps(fn(ex2))}\n\n"
        f"Test input:\n  {json.dumps(test)}\n\n"
        f'Reply with a JSON object of the form {{"output": [[...], [...], [...]]}} '
        f"containing the transformed grid."
    )
    return prompt, json.dumps({"output": out_test}), _grid_match(out_test)


# -------------------------------------------------------------------------
# Tier table
# -------------------------------------------------------------------------

GENERATORS: list[tuple[float, Callable[[np.random.Generator], tuple[str, str, Callable[[str], bool]]]]] = [
    (0.10, _gen_t0_sequence),
    (0.20, _gen_t1_substitution),
    (0.30, _gen_t2_recolor),
    (0.45, _gen_t3_deduction),
    (0.60, _gen_t4_transform),
    (0.75, _gen_t5_multistep),
    (0.90, _gen_t6_rule_discovery),
]


@dataclass(slots=True)
class ReasoningFamily:
    name: str = "reasoning"
    description: str = (
        "Abstract reasoning: sequence completion, ciphers, ARC-AGI-style grid "
        "transforms, deductive puzzles, and rule discovery."
    )

    def generate(
        self,
        n: int,
        seed: int,
        difficulty_range: tuple[float, float] = (0.0, 1.0),
    ) -> list[Task]:
        rng = child_rng(seed, "reasoning.generate")
        lo, hi = difficulty_range
        valid = [(d, g) for d, g in GENERATORS if lo <= d <= hi]
        if not valid:
            valid = GENERATORS
        tasks: list[Task] = []
        for i in range(n):
            d, gen = valid[rng.integers(0, len(valid))]
            sub_rng = np.random.default_rng(child_seed(seed, f"reasoning.{i}"))
            prompt, expected, verifier = gen(sub_rng)
            tasks.append(
                Task(
                    task_id=f"reasoning-{seed}-{i:04d}",
                    family="reasoning",
                    difficulty=float(d),
                    prompt=prompt,
                    verifier=verifier,
                    reference_answer=expected,
                    metadata={"generator": gen.__name__, "tier": d},
                    estimated_seconds=20.0 + 80.0 * d,
                )
            )
        return tasks

    def reference_score(self, task, response):  # noqa: ANN001
        return None  # mechanical only


FAMILY = ReasoningFamily
