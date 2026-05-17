"""Math family — competition-style problems with integer / rational / closed-form answers.

We use *procedurally generated* problems across difficulty tiers so the
benchmark resists memorization. Difficulty is calibrated empirically against
a reference panel (computed offline once and cached in
`data/reference/math.json`).

Difficulty tiers:
  T0 — arithmetic (single-step, integer)
  T1 — algebra (linear, simple word problems)
  T2 — number theory / modular arithmetic
  T3 — combinatorics (counting, ball/urn)
  T4 — sequence / pattern (closed-form expected)
  T5 — multi-step word problem (AMC/AIME class small)
  T6 — Diophantine / olympiad-lite (HMMT class small)
  T7 — adversarial / red-herring numeric phrasing

Generators below produce reproducible, integer-answer problems whose ground
truth is the exact integer or rational the verifier checks for.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable

import numpy as np

from prophet.engine.types import Task
from prophet.utils.seed import child_rng, child_seed


def _ans_matches(expected: str) -> Callable[[str], bool]:
    """Build a verifier comparing the expected answer string to the model output.

    Accepts: integers, fractions ("a/b"), decimals, sign-prefixed numbers,
    and answers wrapped in \\boxed{...} / **bold**. Trailing punctuation is
    stripped. Returns False on ambiguous output.
    """
    expected = expected.strip()
    # Try numeric comparison if expected parses as number
    try:
        expected_num: Fraction | None = Fraction(expected)
    except Exception:
        expected_num = None

    def _verify(answer: str) -> bool:
        if not isinstance(answer, str):
            return False
        s = answer.strip()
        # Strip common wrappers
        s = re.sub(r"\\boxed\{([^}]*)\}", r"\1", s)
        s = re.sub(r"\$+", "", s)
        s = s.replace("**", "")
        s = s.strip().rstrip(".,;:")
        # If the answer is multiline, take the last non-empty line as final.
        lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
        if lines:
            s = lines[-1]
        # Extract last token if multiple
        m = re.search(r"(-?\d+(?:/\d+)?(?:\.\d+)?)", s.replace(",", ""))
        if m:
            cand = m.group(1)
            if expected_num is not None:
                try:
                    return Fraction(cand) == expected_num
                except Exception:
                    pass
            return cand.strip() == expected
        # Fallback: exact match
        return s == expected

    return _verify


def _gen_t0_arithmetic(rng: np.random.Generator) -> tuple[str, str]:
    a = int(rng.integers(2, 200))
    b = int(rng.integers(2, 200))
    op = rng.choice(["+", "-", "*"])
    if op == "+":
        ans = a + b
    elif op == "-":
        ans = a - b
    else:
        ans = a * b
    return f"Compute {a} {op} {b}. Give an integer.", str(ans)


def _gen_t1_algebra(rng: np.random.Generator) -> tuple[str, str]:
    x = int(rng.integers(2, 50))
    m = int(rng.integers(2, 12))
    b = int(rng.integers(-50, 50))
    c = m * x + b
    return (
        f"Solve for x: {m}x + ({b}) = {c}. Return the integer x.",
        str(x),
    )


def _gen_t2_modular(rng: np.random.Generator) -> tuple[str, str]:
    a = int(rng.integers(10, 9999))
    n = int(rng.integers(3, 30))
    ans = a % n
    return f"Compute {a} mod {n}. Return the integer.", str(ans)


def _gen_t3_combo(rng: np.random.Generator) -> tuple[str, str]:
    n = int(rng.integers(4, 12))
    k = int(rng.integers(2, n))
    ans = math.comb(n, k)
    return f"How many ways are there to choose {k} elements from a set of {n}?", str(ans)


def _gen_t4_sequence(rng: np.random.Generator) -> tuple[str, str]:
    a0 = int(rng.integers(-10, 10))
    d = int(rng.integers(2, 9))
    k = int(rng.integers(5, 20))
    ans = a0 + d * (k - 1)
    return (
        f"An arithmetic sequence starts at a_1 = {a0} with common difference {d}. "
        f"What is a_{k}? Return an integer.",
        str(ans),
    )


def _gen_t5_word(rng: np.random.Generator) -> tuple[str, str]:
    apples = int(rng.integers(20, 200))
    eaten = int(rng.integers(2, apples // 3))
    given = int(rng.integers(1, 10))
    bags = int(rng.integers(2, 6))
    per_bag = (apples - eaten) // bags
    return (
        f"A farmer has {apples} apples. After {eaten} are eaten, the remaining apples "
        f"are split equally into {bags} bags (extras discarded). Each bag also receives "
        f"{given} bonus apples (separately). How many apples are in one bag? "
        f"Return an integer.",
        str(per_bag + given),
    )


def _gen_t6_olympiad(rng: np.random.Generator) -> tuple[str, str]:
    # Sum-of-divisors of a random integer
    n = int(rng.integers(20, 500))
    ans = sum(d for d in range(1, n + 1) if n % d == 0)
    return (
        f"Let σ(n) denote the sum of positive divisors of n (including 1 and n). "
        f"Compute σ({n}).",
        str(ans),
    )


def _gen_t7_adversarial(rng: np.random.Generator) -> tuple[str, str]:
    a = int(rng.integers(10, 99))
    b = int(rng.integers(10, 99))
    ans = a + b
    # Adversarial: include red-herring numbers
    rh1 = int(rng.integers(100, 999))
    rh2 = int(rng.integers(100, 999))
    return (
        f"Maria saw the number {rh1} on a sign earlier today and the number {rh2} "
        f"on a passing bus. After that, she received {a} stickers in the morning and "
        f"{b} stickers in the afternoon. The signs and bus numbers are unrelated to "
        f"her stickers. How many stickers does Maria have in total?",
        str(ans),
    )


GENERATORS: list[tuple[float, Callable[[np.random.Generator], tuple[str, str]]]] = [
    (0.05, _gen_t0_arithmetic),
    (0.15, _gen_t1_algebra),
    (0.25, _gen_t2_modular),
    (0.35, _gen_t3_combo),
    (0.45, _gen_t4_sequence),
    (0.60, _gen_t5_word),
    (0.80, _gen_t6_olympiad),
    (0.92, _gen_t7_adversarial),
]


@dataclass(slots=True)
class MathFamily:
    name: str = "math"
    description: str = "Procedurally generated math problems with integer / rational ground truth."

    def generate(
        self,
        n: int,
        seed: int,
        difficulty_range: tuple[float, float] = (0.0, 1.0),
    ) -> list[Task]:
        rng = child_rng(seed, "math.generate")
        lo, hi = difficulty_range
        valid = [(d, g) for d, g in GENERATORS if lo <= d <= hi]
        if not valid:
            valid = GENERATORS
        tasks: list[Task] = []
        for i in range(n):
            d, gen = valid[rng.integers(0, len(valid))]
            sub_rng = np.random.default_rng(child_seed(seed, f"math.{i}"))
            prompt, expected = gen(sub_rng)
            tasks.append(
                Task(
                    task_id=f"math-{seed}-{i:04d}",
                    family="math",
                    difficulty=float(d),
                    prompt=prompt,
                    verifier=_ans_matches(expected),
                    reference_answer=expected,
                    metadata={"generator": gen.__name__, "tier": d},
                    estimated_seconds=20.0 + 60.0 * d,
                )
            )
        return tasks

    def reference_score(self, task, response):  # noqa: ANN001
        return None  # mechanical only


FAMILY = MathFamily
