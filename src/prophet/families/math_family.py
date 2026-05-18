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
  T8 — system of 3 linear equations with integer solutions (return x)
  T9 — small Diophantine: ax + by = c with gcd(a,b)=1 (return any valid x)

Generators below produce reproducible, integer-answer problems whose ground
truth is the exact integer or rational the verifier checks for.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction

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


def _gen_t8_linear_system3(rng: np.random.Generator) -> tuple[str, str]:
    """System of 3 linear equations in (x, y, z) with integer solutions.

    Construct ground truth (x, y, z) first, then pick three integer coefficient
    rows; compute the RHS so the system is consistent by construction. We
    additionally require the coefficient matrix to be full-rank (det != 0) so
    the solution is unique. The agent is asked to return ``x`` only — a single
    integer.
    """
    x = int(rng.integers(-9, 10))
    y = int(rng.integers(-9, 10))
    z = int(rng.integers(-9, 10))
    # Re-sample until we get a non-singular coefficient matrix.
    for _ in range(200):
        coeffs: list[tuple[int, int, int]] = []
        while len(coeffs) < 3:
            a = int(rng.integers(-5, 6))
            b = int(rng.integers(-5, 6))
            c = int(rng.integers(-5, 6))
            if a == 0 and b == 0 and c == 0:
                continue
            if (a, b, c) in coeffs:
                continue
            coeffs.append((a, b, c))
        # Check determinant != 0 (integer 3x3 cofactor expansion).
        m = coeffs
        det = (
            m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
        )
        if det != 0:
            break
    else:
        # Fallback: identity matrix (always invertible).
        coeffs = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    rhs = [a * x + b * y + c * z for a, b, c in coeffs]
    lines = []
    for (a, b, c), r in zip(coeffs, rhs):
        lines.append(f"  {a}x + ({b})y + ({c})z = {r}")
    eq_block = "\n".join(lines)
    prompt = (
        "Solve the following system of three linear equations for integer x, y, z:\n"
        f"{eq_block}\n"
        "Return only the integer value of x."
    )
    return prompt, str(x)


def _gen_t9_diophantine(rng: np.random.Generator) -> tuple[str, str]:
    """Solve ax + by = c with small a, b, c and gcd(a, b) = 1.

    To keep a unique ground truth, we additionally require x in [0, b - 1].
    Existence is guaranteed by gcd(a, b) = 1.
    """
    # Sample a, b with gcd = 1, both in [2, 19].
    while True:
        a = int(rng.integers(2, 20))
        b = int(rng.integers(2, 20))
        if a != b and math.gcd(a, b) == 1:
            break
    # Pick a target c not too small / not too large.
    c = int(rng.integers(20, 200))
    # Find the smallest non-negative x with (c - a*x) % b == 0; existence by gcd=1.
    x = 0
    while x < b:
        if (c - a * x) % b == 0:
            break
        x += 1
    # By Bezout / gcd(a, b) = 1 we must find such x in [0, b - 1]; assert defensively.
    assert (c - a * x) % b == 0, "Unexpected: no solution found within [0, b)"
    prompt = (
        f"Find a non-negative integer x with 0 ≤ x < {b} such that "
        f"{a}*x + {b}*y = {c} has an integer solution y. "
        f"Return the integer x only."
    )
    return prompt, str(x)


# --------------------------------------------------------------------------
# T_extreme (d ≥ 0.97) — designed to be HARD for frontier models in 2026.
# Targets: <15% accuracy on Gemini 3 Flash, Claude Opus 4.7, GPT-5.
# Each generator yields a tightly-verifiable integer answer.
# --------------------------------------------------------------------------

_PRIMES_50_200 = [
    p for p in range(50, 200)
    if all(p % d != 0 for d in range(2, int(p**0.5) + 1))
]


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


# Larger prime banks for genuinely hard arithmetic in 2026
_PRIMES_5K_20K = [p for p in range(5000, 20000) if _is_prime(p)]
_PRIMES_1K_5K = [p for p in range(1000, 5000) if _is_prime(p)]


def _gen_text_mult_order(rng: np.random.Generator) -> tuple[str, str]:
    """Multiplicative order ord_p(a) for primes p in [1000, 5000].

    Larger primes push the order range to 4-digit values, defeating
    pattern-matching on small p, and require actual modular exponentiation
    rather than mental arithmetic. Per AICrypto / modular-exp transformer
    studies, frontier accuracy drops below 10% at p ≳ 10³ without tools.
    """
    p = _PRIMES_1K_5K[int(rng.integers(0, len(_PRIMES_1K_5K)))]
    while True:
        a = int(rng.integers(2, p))
        if math.gcd(a, p) == 1:
            break
    order = 1
    val = a % p
    while val != 1:
        val = (val * a) % p
        order += 1
        if order > p:
            order = -1
            break
    prompt = (
        f"Find the multiplicative order of {a} modulo {p}, i.e. the smallest "
        f"positive integer k such that {a}^k ≡ 1 (mod {p}). "
        f"Return the integer k only."
    )
    return prompt, str(order)


def _gen_text_crt12(rng: np.random.Generator) -> tuple[str, str]:
    """12-modulus Chinese Remainder Theorem.

    Mixed prime-power and coprime composite moduli. The product of moduli
    grows to ~10^15, so the answer is regularly 11-13 digits long, forcing
    an iterative big-integer CRT lift that frontier LLMs cannot reliably
    perform mentally (cf. arXiv 2511.00763 accuracy-cliff at long chains).
    """
    moduli_pool = [3, 4, 5, 7, 8, 9, 11, 13, 16, 17, 19, 23, 25, 27, 29, 31, 32, 37, 41, 43, 47]
    chosen: list[int] = []
    pool = list(moduli_pool)
    rng.shuffle(pool)
    for m in pool:
        if all(math.gcd(m, c) == 1 for c in chosen):
            chosen.append(m)
        if len(chosen) == 12:
            break
    if len(chosen) < 12:
        chosen = [3, 4, 5, 7, 11, 13, 17, 19, 23, 25, 29, 31]
    remainders = [int(rng.integers(0, m)) for m in chosen]
    x = remainders[0]
    M = chosen[0]
    for r, m in zip(remainders[1:], chosen[1:]):
        for t in range(m):
            if (x + t * M) % m == r:
                x = x + t * M
                M = M * m
                break
    constraints = "\n".join(
        f"  x ≡ {r} (mod {m})" for r, m in zip(remainders, chosen)
    )
    prompt = (
        "Find the smallest non-negative integer x satisfying all of the "
        f"following 12 congruences:\n{constraints}\n"
        "Return the integer x only."
    )
    return prompt, str(x)


def _gen_text_pell(rng: np.random.Generator) -> tuple[str, str]:
    """Smallest positive y in the Pell equation x² - Dy² = 1 for non-square D.

    D ∈ [200, 1000]. The fundamental y can range from single digits into
    the billions when D is close to a perfect square or has long continued-
    fraction period. We bound the period at 200 to keep the answer
    representable but still well past mental-arithmetic feasibility for
    frontier models (cf. OlymMATH NT hard, arXiv 2503.21380).
    """
    while True:
        D = int(rng.integers(200, 1000))
        s = int(round(D ** 0.5))
        if s * s != D:
            break

    # Solve via continued fractions: standard algorithm.
    m0, d0, a0 = 0, 1, int(D ** 0.5)
    a = a0
    h_prev, h_cur = 1, a0
    k_prev, k_cur = 0, 1
    if h_cur * h_cur - D * k_cur * k_cur == 1:
        y = k_cur
    else:
        m, d = m0, d0
        for _ in range(200):
            m = d * a - m
            d = (D - m * m) // d
            a = (a0 + m) // d
            h_prev, h_cur = h_cur, a * h_cur + h_prev
            k_prev, k_cur = k_cur, a * k_cur + k_prev
            if h_cur * h_cur - D * k_cur * k_cur == 1:
                y = k_cur
                break
        else:
            y = k_cur  # fallback (should not hit)
    prompt = (
        f"Consider the Pell equation x^2 - {D}*y^2 = 1, with x and y positive integers. "
        f"Find the smallest positive integer y satisfying this equation. "
        f"Return the integer y only."
    )
    return prompt, str(y)


def _gen_text_lattice_paths(rng: np.random.Generator) -> tuple[str, str]:
    """Count monotone lattice paths from (0,0) to (n,n) avoiding k forbidden cells.

    n ∈ [8, 12], k ∈ [6, 10] — both bumped from prior pilot (n=5-7, k=2-5).
    The DP recurrence is mechanical but counting through interacting
    forbidden regions requires consistent state-tracking; frontier LLMs
    overcount past n≈8 (cf. BeyondBench Hard, arXiv 2509.24210).
    """
    n = int(rng.integers(8, 13))
    k = int(rng.integers(6, 11))
    cells = [(r, c) for r in range(n + 1) for c in range(n + 1)
             if (r, c) not in {(0, 0), (n, n)}]
    rng.shuffle(cells)
    forbidden = sorted(cells[:k])
    forbidden_set = set(forbidden)
    # DP: dp[r][c] = number of monotone paths from (0,0) to (r,c)
    dp = [[0] * (n + 1) for _ in range(n + 1)]
    dp[0][0] = 1
    for r in range(n + 1):
        for c in range(n + 1):
            if (r, c) in forbidden_set:
                dp[r][c] = 0
                continue
            if r == 0 and c == 0:
                continue
            from_top = dp[r - 1][c] if r > 0 else 0
            from_left = dp[r][c - 1] if c > 0 else 0
            dp[r][c] = from_top + from_left
    count = dp[n][n]
    coords = ", ".join(f"({r},{c})" for r, c in forbidden)
    prompt = (
        f"Count the number of monotone lattice paths from (0, 0) to ({n}, {n}) "
        f"that only step Right (r→r+1) or Up (c→c+1) by one unit, and that "
        f"avoid all of the following forbidden grid cells: {coords}. "
        f"Both (0, 0) and ({n}, {n}) are allowed. Return the integer count only."
    )
    return prompt, str(count)


def _gen_text_discrete_log(rng: np.random.Generator) -> tuple[str, str]:
    """Discrete log: find x in [1, p-1] with g**x ≡ h (mod p).

    Per AICrypto (arXiv 2507.09580) and modular-exp transformer limits
    (arXiv 2506.23679), p in [5×10³, 2×10⁴] keeps the search space
    explicitly intractable for non-tool-using frontier models (<5% acc).
    """
    p = _PRIMES_5K_20K[int(rng.integers(0, len(_PRIMES_5K_20K)))]
    # Find a primitive root g (need g^k != 1 for any k < p-1 dividing p-1)
    phi = p - 1
    # Factor phi (it's at most 198)
    def _prime_factors(n: int) -> list[int]:
        f: list[int] = []
        d = 2
        while d * d <= n:
            if n % d == 0:
                f.append(d)
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            f.append(n)
        return f
    factors = _prime_factors(phi)
    g = 2
    while g < p:
        if all(pow(g, phi // q, p) != 1 for q in factors):
            break
        g += 1
    if g >= p:
        g = 2  # fallback (shouldn't happen for primes in this range)
    # Pick x in [1, p-1], compute h = g^x mod p, then ask the LLM for x.
    x = int(rng.integers(1, p))
    h = pow(g, x, p)
    prompt = (
        f"Solve the discrete logarithm: find the integer x with 1 ≤ x ≤ {p-1} "
        f"such that {g}^x ≡ {h} (mod {p}). "
        f"Return the integer x only."
    )
    return prompt, str(x)


def _gen_text_quadratic_residue(rng: np.random.Generator) -> tuple[str, str]:
    """Modular square root in p ∈ [1000, 5000]: find smaller root r with r²≡a (mod p)."""
    p = _PRIMES_1K_5K[int(rng.integers(0, len(_PRIMES_1K_5K)))]
    r = int(rng.integers(1, (p - 1) // 2 + 1))
    a = (r * r) % p
    prompt = (
        f"Find the smaller integer r with 1 ≤ r ≤ {(p-1)//2} such that "
        f"r^2 ≡ {a} (mod {p}). Return the integer r only."
    )
    return prompt, str(r)


def _gen_text_combinatorial_count(rng: np.random.Generator) -> tuple[str, str]:
    """Count integers in [1, N] coprime to BOTH m1 and m2 (a 2-modulus
    inclusion-exclusion problem). Larger N (10⁴-10⁵) and 2-modulus
    constraint defeat the naive single-Euler-totient pattern.
    """
    N = int(rng.integers(10_000, 100_000))
    m_pool = [6, 10, 12, 14, 15, 18, 21, 22, 26, 33, 35, 39, 55, 77]
    while True:
        m1 = m_pool[int(rng.integers(0, len(m_pool)))]
        m2 = m_pool[int(rng.integers(0, len(m_pool)))]
        if m1 != m2 and math.gcd(m1, m2) == 1:
            break
    count = sum(
        1 for n in range(1, N + 1)
        if math.gcd(n, m1) == 1 and math.gcd(n, m2) == 1
    )
    prompt = (
        f"How many integers n with 1 ≤ n ≤ {N} satisfy BOTH "
        f"gcd(n, {m1}) = 1 AND gcd(n, {m2}) = 1? Return the integer count."
    )
    return prompt, str(count)


GENERATORS: list[tuple[float, Callable[[np.random.Generator], tuple[str, str]]]] = [
    (0.05, _gen_t0_arithmetic),
    (0.15, _gen_t1_algebra),
    (0.25, _gen_t2_modular),
    (0.35, _gen_t3_combo),
    (0.45, _gen_t4_sequence),
    (0.60, _gen_t5_word),
    (0.80, _gen_t6_olympiad),
    (0.92, _gen_t7_adversarial),
    (0.94, _gen_t8_linear_system3),
    (0.97, _gen_t9_diophantine),
    # T_extreme — target <15% on frontier (calibrated against 2025-2026 lit)
    (0.980, _gen_text_combinatorial_count),
    (0.985, _gen_text_mult_order),
    (0.988, _gen_text_quadratic_residue),
    (0.992, _gen_text_lattice_paths),
    (0.994, _gen_text_crt12),
    (0.996, _gen_text_pell),
    (0.998, _gen_text_discrete_log),
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

    def reference_score(self, task, response):
        return None  # mechanical only


FAMILY = MathFamily
