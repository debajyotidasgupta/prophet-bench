"""Code family — procedurally-generated programming problems with verifiable outputs.

Difficulty tiers:
  T0 (0.05) — predict stdout of a literal `print(expr)` statement (arithmetic).
  T1 (0.15) — predict the value of a one-line Python expression.
  T2 (0.25) — given a tiny function definition, return the value of `f(arg)`.
  T3 (0.40) — predict the output of a procgen 4–6 line Python snippet.
  T4 (0.55) — write/return a function body; verifier exec()s it on N=3 fixed test cases.
  T5 (0.70) — implement a harder algorithm (sort/dedup/search variant); same exec-style verifier.
  T6 (0.85) — bug-fix: a procgen buggy 1-line function; agent supplies the fixed body.
  T7 (0.95) — multi-step compose-two-functions; verify final output value.

Safety model
------------
For tiers T4..T7 the verifier needs to *execute* candidate code. We isolate
execution in a subprocess via ``subprocess.run([sys.executable, '-c', script], timeout=5)``
with no shell, no inherited environment, and a 5-second wall-clock cap.
The candidate code is wrapped inside a top-level module with a tightly
controlled namespace (no networking imports are required by the test harness;
candidates can still ``import math`` etc. but cannot affect the orchestrator).
The subprocess prints exactly one JSON line containing per-test pass/fail and
the verifier reads that.

For tiers that only ask the agent to *predict* an output (T0..T3, T7), the
verifier is a pure string compare and never executes anything.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import textwrap
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from prophet.engine.types import Task
from prophet.utils.seed import child_rng, child_seed

# -------------------------------------------------------------------------
# Verifier helpers
# -------------------------------------------------------------------------

def _normalize_str(s: str) -> str:
    """Strip ``\\boxed{…}``, code-fences, surrounding whitespace and trailing punctuation."""
    s = s.strip()
    s = re.sub(r"\\boxed\{([^}]*)\}", r"\1", s)
    s = re.sub(r"```(?:python|py)?\s*", "", s)
    s = s.replace("```", "")
    # Single-quote / double-quote bare strings → keep raw
    s = s.strip()
    s = s.rstrip(".,;:")
    return s


def _exact_or_numeric_match(expected: str) -> Callable[[str], bool]:
    """Compare answer to `expected`, accepting numeric equivalents and last-line answers."""
    expected_norm = _normalize_str(expected)
    try:
        expected_num: float | None = float(expected_norm)
    except Exception:
        expected_num = None

    def _verify(answer: str) -> bool:
        if not isinstance(answer, str):
            return False
        s = _normalize_str(answer)
        # If multiline, prefer the final non-empty line
        lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
        candidates: list[str] = []
        if lines:
            candidates.append(lines[-1])
            candidates.append(lines[0])
        candidates.append(s)
        for cand in candidates:
            cand = _normalize_str(cand)
            if cand == expected_norm:
                return True
            if expected_num is not None:
                m = re.search(r"-?\d+(?:\.\d+)?", cand)
                if m:
                    try:
                        if abs(float(m.group(0)) - expected_num) <= 1e-9 * max(1.0, abs(expected_num)):
                            return True
                    except Exception:
                        pass
        return False

    return _verify


def _extract_code_block(answer: str) -> str:
    """Extract the body of a Python code block; fall back to raw answer."""
    if not isinstance(answer, str):
        return ""
    m = re.search(r"```(?:python|py)?\s*(.+?)```", answer, flags=re.DOTALL)
    if m:
        return m.group(1)
    return answer


def _exec_candidate_in_subprocess(
    candidate_code: str,
    test_calls: list[tuple[str, str]],  # [(call_expr, expected_repr)]
    timeout_s: float = 5.0,
) -> bool:
    """Run candidate code in a subprocess and check expected outputs.

    The subprocess executes `candidate_code` (which should define the
    function under test), then evaluates each `call_expr` and prints whether
    its `repr` matches `expected_repr`. Returns True iff *every* test passed
    and the subprocess exited cleanly.
    """
    test_data = json.dumps(test_calls)
    runner = textwrap.dedent(
        f"""
        import json, sys
        CANDIDATE = {candidate_code!r}
        TESTS = json.loads({test_data!r})
        ns = {{}}
        try:
            exec(CANDIDATE, ns)
        except Exception as e:
            print(json.dumps({{"ok": False, "error": "exec: " + repr(e)}}))
            sys.exit(0)
        all_ok = True
        details = []
        for call, expected in TESTS:
            try:
                got = eval(call, ns)
                ok = repr(got) == expected
            except Exception as e:
                got = None
                ok = False
                details.append({{"call": call, "error": repr(e)}})
            details.append({{"call": call, "got": repr(got), "expected": expected, "ok": ok}})
            if not ok:
                all_ok = False
        print(json.dumps({{"ok": all_ok, "details": details}}))
        """
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", runner],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
            env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"},
        )
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    if proc.returncode != 0:
        return False
    try:
        # Use the last non-empty stdout line as the JSON result.
        lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
        if not lines:
            return False
        payload = json.loads(lines[-1])
    except Exception:
        return False
    return bool(payload.get("ok"))


def _exec_verifier(test_calls: list[tuple[str, str]]) -> Callable[[str], bool]:
    """Build a verifier that exec()s the candidate code on the given calls."""
    def _verify(answer: str) -> bool:
        code = _extract_code_block(answer)
        if not code.strip():
            return False
        return _exec_candidate_in_subprocess(code, test_calls)
    return _verify


# -------------------------------------------------------------------------
# Tier generators
# -------------------------------------------------------------------------

def _gen_t0_print(rng: np.random.Generator) -> tuple[str, str, Callable[[str], bool], dict]:
    a = int(rng.integers(2, 50))
    b = int(rng.integers(2, 50))
    op = rng.choice(["+", "-", "*"])
    if op == "+":
        ans = a + b
    elif op == "-":
        ans = a - b
    else:
        ans = a * b
    prompt = f"What does this Python statement print?\n\n    print({a} {op} {b})\n\nReturn the printed value (an integer)."
    return prompt, str(ans), _exact_or_numeric_match(str(ans)), {"a": a, "b": b, "op": op}


def _gen_t1_oneliner(rng: np.random.Generator) -> tuple[str, str, Callable[[str], bool], dict]:
    a = int(rng.integers(2, 30))
    b = int(rng.integers(2, 20))
    snippets = [
        (f"len('{ 'x' * a }')", a),
        (f"sum(range({a}))", sum(range(a))),
        (f"{a} ** 2", a * a),
        (f"abs({a} - {b})", abs(a - b)),
        (f"min({a}, {b}, {a + b})", min(a, b, a + b)),
        (f"max({a}, {b}, {a - b})", max(a, b, a - b)),
        (f"{a} // {b if b else 1}", a // (b if b else 1)),
        (f"{a} % {max(b, 2)}", a % max(b, 2)),
    ]
    expr, ans = snippets[int(rng.integers(0, len(snippets)))]
    prompt = (
        f"Evaluate this Python expression and return its integer value:\n\n    {expr}"
    )
    return prompt, str(ans), _exact_or_numeric_match(str(ans)), {"expr": expr}


def _gen_t2_call(rng: np.random.Generator) -> tuple[str, str, Callable[[str], bool], dict]:
    k = int(rng.integers(2, 10))
    arg = int(rng.integers(2, 15))
    variants = [
        ("def f(x): return x * x", arg * arg, f"f({arg})"),
        (f"def f(x): return x + {k}", arg + k, f"f({arg})"),
        (f"def f(x): return x * {k}", arg * k, f"f({arg})"),
        ("def f(x): return x ** 2 - x", arg * arg - arg, f"f({arg})"),
        (f"def f(x): return {k} - x", k - arg, f"f({arg})"),
    ]
    defn, ans, call = variants[int(rng.integers(0, len(variants)))]
    prompt = (
        f"Given the Python definition\n\n    {defn}\n\n"
        f"what is the value of `{call}`? Return an integer."
    )
    return prompt, str(ans), _exact_or_numeric_match(str(ans)), {"defn": defn, "call": call}


def _gen_t3_snippet(rng: np.random.Generator) -> tuple[str, str, Callable[[str], bool], dict]:
    a = int(rng.integers(2, 12))
    b = int(rng.integers(2, 12))
    k = int(rng.integers(1, 6))
    # 4-6 line snippet; we manually compute the ground truth.
    snippet = (
        f"a = {a}\n"
        f"b = {b}\n"
        f"total = 0\n"
        f"for i in range(1, {k} + 1):\n"
        f"    total += a * i - b\n"
        f"print(total)"
    )
    total = 0
    for i in range(1, k + 1):
        total += a * i - b
    prompt = (
        "What does the following Python snippet print?\n\n"
        + textwrap.indent(snippet, "    ")
        + "\n\nReturn the integer that is printed."
    )
    return prompt, str(total), _exact_or_numeric_match(str(total)), {"snippet": snippet}


def _gen_t4_implement_simple(rng: np.random.Generator) -> tuple[str, str, Callable[[str], bool], dict]:
    """Implement a tiny function; verifier exec()s candidate on fixed test cases."""
    k = int(rng.integers(2, 7))
    # Function: multiply by k and add 1
    name = "f"
    tests = [
        (f"{name}({int(rng.integers(0, 10))})", None),
        (f"{name}({int(rng.integers(10, 30))})", None),
        (f"{name}({int(rng.integers(30, 60))})", None),
    ]
    # Compute expected values
    eval_calls: list[tuple[str, str]] = []
    for call, _ in tests:
        x = int(re.search(r"\((-?\d+)\)", call).group(1))
        expected = x * k + 1
        eval_calls.append((call, repr(expected)))
    prompt = (
        f"Implement a Python function `{name}(x)` that returns `x * {k} + 1`. "
        f"Reply with a Python code block containing only the function definition, e.g.:\n\n"
        f"```python\ndef {name}(x):\n    return x * {k} + 1\n```"
    )
    reference = f"def {name}(x):\n    return x * {k} + 1"
    verifier = _exec_verifier(eval_calls)
    # Wrap reference in code-fence so OracleAgent's reply is accepted by _extract_code_block
    reference_block = f"```python\n{reference}\n```"
    return prompt, reference_block, verifier, {"k": k, "tests": eval_calls}


def _gen_t5_implement_algorithm(rng: np.random.Generator) -> tuple[str, str, Callable[[str], bool], dict]:
    """Implement an algorithmic function (dedup / reverse / sort variant)."""
    name = "g"
    variants = [
        (
            "Implement `g(lst)` that returns the sorted, deduplicated list (ascending).",
            "def g(lst):\n    return sorted(set(lst))",
            lambda lst: sorted(set(lst)),
        ),
        (
            "Implement `g(lst)` that returns the list reversed.",
            "def g(lst):\n    return list(reversed(lst))",
            lambda lst: list(reversed(lst)),
        ),
        (
            "Implement `g(lst)` that returns the sum of even numbers in `lst`.",
            "def g(lst):\n    return sum(x for x in lst if x % 2 == 0)",
            lambda lst: sum(x for x in lst if x % 2 == 0),
        ),
        (
            "Implement `g(lst)` that returns the index of the maximum element in `lst` "
            "(first occurrence; behavior undefined on empty).",
            "def g(lst):\n    return lst.index(max(lst))",
            lambda lst: lst.index(max(lst)),
        ),
    ]
    idx = int(rng.integers(0, len(variants)))
    description, body, fn = variants[idx]
    # 3 procgen test cases
    test_calls: list[tuple[str, str]] = []
    for _ in range(3):
        n = int(rng.integers(3, 8))
        lst = [int(rng.integers(-10, 30)) for _ in range(n)]
        try:
            expected = fn(lst)
        except Exception:
            expected = None
        test_calls.append((f"{name}({lst!r})", repr(expected)))
    prompt = (
        f"{description}\n\nReply with a Python code block containing only the function "
        f"definition. Example format:\n\n```python\n{body}\n```"
    )
    reference_block = f"```python\n{body}\n```"
    verifier = _exec_verifier(test_calls)
    return prompt, reference_block, verifier, {"variant": idx, "tests": test_calls}


def _gen_t6_bugfix(rng: np.random.Generator) -> tuple[str, str, Callable[[str], bool], dict]:
    """Bug-fix: provide a buggy one-line function; agent supplies the fixed body."""
    a = int(rng.integers(2, 20))
    name = "h"
    buggy_variants = [
        (
            f"def {name}(x):\n    return x * x + {a}    # BUG: should be x*x - {a}",
            f"def {name}(x):\n    return x * x - {a}",
            lambda x: x * x - a,
        ),
        (
            f"def {name}(x):\n    return x + {a}    # BUG: should be x - {a}",
            f"def {name}(x):\n    return x - {a}",
            lambda x: x - a,
        ),
        (
            f"def {name}(x):\n    return x // {a}    # BUG: should be x % {a}",
            f"def {name}(x):\n    return x % {a}",
            lambda x: x % a,
        ),
    ]
    idx = int(rng.integers(0, len(buggy_variants)))
    buggy, fixed, fn = buggy_variants[idx]
    # 3 test calls
    test_calls: list[tuple[str, str]] = []
    for _ in range(3):
        x = int(rng.integers(2, 50))
        test_calls.append((f"{name}({x})", repr(fn(x))))
    prompt = (
        f"The following Python function has a one-line bug:\n\n"
        f"```python\n{buggy}\n```\n\n"
        f"Reply with the corrected function (just the def block, in a python code fence). "
        f"Three calls of the form `{name}(x)` will be checked."
    )
    reference_block = f"```python\n{fixed}\n```"
    verifier = _exec_verifier(test_calls)
    return prompt, reference_block, verifier, {"variant": idx, "tests": test_calls}


def _gen_t7_compose(rng: np.random.Generator) -> tuple[str, str, Callable[[str], bool], dict]:
    """Predict the output of composing two given functions on a specific input."""
    a = int(rng.integers(2, 8))
    b = int(rng.integers(1, 8))
    x = int(rng.integers(3, 12))
    # f(x) = x*a + b ; g(x) = x*x ; want g(f(x))
    fx = x * a + b
    gx = fx * fx
    prompt = (
        "Consider two Python functions:\n\n"
        f"    def f(x): return x * {a} + {b}\n"
        f"    def g(x): return x * x\n\n"
        f"What is the integer value of `g(f({x}))`?"
    )
    return prompt, str(gx), _exact_or_numeric_match(str(gx)), {"f_a": a, "f_b": b, "x": x}


# -------------------------------------------------------------------------
# T_extreme (d ≥ 0.97) — designed to be HARD for frontier models in 2026.
# Pure prediction tasks (no subprocess execution): subset-sum-count,
# modular knapsack, and DAG path counting. All return single integers.
# -------------------------------------------------------------------------


def _gen_text_dp_count(rng: np.random.Generator) -> tuple[str, str, Callable[[str], bool], dict]:
    """Subset-sum COUNT: number of distinct subsets summing to target T.

    Frontier LLMs often confuse "count subsets" with "exists" or "min subset",
    and DP-counting requires careful state-tracking they tend to skip.
    """
    n = int(rng.integers(8, 13))  # 8..12 items
    values = [int(rng.integers(1, 16)) for _ in range(n)]  # 1..15
    # Pick target so at least one subset hits it: build from a random sub-subset
    # then optionally perturb so the count is non-trivial (>=2 typical).
    indicator = [int(rng.integers(0, 2)) for _ in range(n)]
    target = sum(v for v, b in zip(values, indicator) if b)
    # Avoid degenerate target = 0 (counts all subsets equal to empty set only
    # for non-zero values, but if any value is 0 we get more — keep it safe).
    if target == 0:
        target = int(values[0])
    # DP count
    dp = [0] * (target + 1)
    dp[0] = 1
    for v in values:
        if v > target:
            continue
        for s in range(target, v - 1, -1):
            dp[s] += dp[s - v]
    count = dp[target]
    # If somehow count is 0 (shouldn't happen — indicator yields at least 1),
    # fall back to 1.
    if count <= 0:
        count = 1
    prompt = (
        "Count the number of distinct subsets of the following list of "
        "non-negative integers whose elements sum to exactly the given target. "
        "Two subsets are distinct if their index sets differ (the empty subset "
        "counts only if the target is 0). Return a single integer.\n\n"
        f"List: {values}\n"
        f"Target: {target}\n\n"
        "Return only the integer count."
    )
    return (
        prompt,
        str(count),
        _exact_or_numeric_match(str(count)),
        {"values": values, "target": target, "count": count},
    )


def _gen_text_knapsack_mod(rng: np.random.Generator) -> tuple[str, str, Callable[[str], bool], dict]:
    """Modular knapsack: max sum-of-values with sum-of-weights ≡ k (mod m).

    Each item has a (weight, value). Use any subset (0/1 knapsack — no
    multiplicities) and maximize total value subject to the modular weight
    constraint. The "no weight budget" twist plus modular constraint trips up
    frontier LLMs that default to vanilla knapsack reasoning. We re-sample
    parameters until the constraint is satisfiable by some non-empty subset
    (or by the empty set when k=0), guaranteeing a well-defined positive
    or zero answer.
    """
    NEG = -10**9
    for _ in range(200):
        n = int(rng.integers(11, 16))  # 11..15 items (was 6..8)
        weights = [int(rng.integers(1, 31)) for _ in range(n)]  # 1..30 (was 1..20)
        values = [int(rng.integers(1, 71)) for _ in range(n)]   # 1..70 (was 1..50)
        m = int(rng.integers(5, 12))  # 5..11 (was 3..7)
        k = int(rng.integers(0, m))
        # DP: best[r] = max value such that subset weight ≡ r (mod m).
        best = [NEG] * m
        best[0] = 0
        for w, v in zip(weights, values):
            new = best[:]
            for r in range(m):
                if best[r] == NEG:
                    continue
                nr = (r + w) % m
                if best[r] + v > new[nr]:
                    new[nr] = best[r] + v
            best = new
        # Require satisfiable AND a non-trivial answer (avoid the trivial
        # k=0/empty-subset case where answer = 0 + any sum is meaningful).
        if best[k] != NEG and best[k] > 0:
            break
    else:
        # Fallback (vanishingly rare): use a fixed example.
        weights = [3, 5, 7, 2, 4]
        values = [10, 20, 15, 5, 8]
        m, k = 4, 1
        best = [NEG] * m
        best[0] = 0
        for w, v in zip(weights, values):
            new = best[:]
            for r in range(m):
                if best[r] == NEG:
                    continue
                nr = (r + w) % m
                if best[r] + v > new[nr]:
                    new[nr] = best[r] + v
            best = new
    optimal = best[k]
    items_str = ", ".join(f"(w={w}, v={v})" for w, v in zip(weights, values))
    prompt = (
        "You are given a list of items, each with a weight w and a value v. "
        "Choose any subset of the items (each item at most once; no overall "
        "weight budget) so that the SUM OF WEIGHTS is congruent to "
        f"{k} modulo {m}. Among all such subsets, maximise the SUM OF VALUES. "
        "The empty subset (weight sum 0, value 0) is allowed; assume at least "
        "one valid subset exists.\n\n"
        f"Items: {items_str}\n"
        f"Constraint: total weight ≡ {k} (mod {m})\n\n"
        "Return only the integer optimal total value."
    )
    return (
        prompt,
        str(optimal),
        _exact_or_numeric_match(str(optimal)),
        {"weights": weights, "values": values, "m": m, "k": k, "optimal": optimal},
    )


def _gen_text_dag_paths(rng: np.random.Generator) -> tuple[str, str, Callable[[str], bool], dict]:
    """DAG path counting: number of distinct paths from node 0 to node n-1.

    Build a topo-sorted random DAG (edges only point from lower → higher index)
    so acyclicity is guaranteed by construction. Frontier LLMs often miscount
    when fan-in/fan-out compound across multiple intermediate vertices.
    """
    n = int(rng.integers(12, 17))  # 12..16 nodes (was 6..8)
    edges: list[tuple[int, int]] = []
    # Ensure 0 → n-1 is reachable: include the chain 0→1→…→n-1 with some
    # probability for each link, then add extra random forward edges.
    # Simpler: always add edge (i, i+1) for i in 0..n-2 to guarantee
    # at least one path, then sprinkle additional forward edges.
    for i in range(n - 1):
        edges.append((i, i + 1))
    # Add ~n..2n random extra forward edges (no duplicates) — denser graph
    # produces larger path counts and more error-prone counting.
    edge_set = set(edges)
    extra = int(rng.integers(n, 2 * n + 1))
    attempts = 0
    while extra > 0 and attempts < 200:
        attempts += 1
        u = int(rng.integers(0, n - 1))
        v = int(rng.integers(u + 1, n))
        if (u, v) not in edge_set:
            edge_set.add((u, v))
            edges.append((u, v))
            extra -= 1
    edges.sort()
    # Count paths via DP on topo order (already 0..n-1).
    dp = [0] * n
    dp[0] = 1
    adj: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
    for u in range(n):
        if dp[u] == 0:
            continue
        for v in adj[u]:
            dp[v] += dp[u]
    count = dp[n - 1]
    prompt = (
        f"Consider a directed acyclic graph (DAG) on the {n} nodes "
        f"{{0, 1, …, {n - 1}}} with the following edge list (each edge "
        "u→v points from a lower-numbered node to a higher-numbered node):\n\n"
        f"  edges = {edges}\n\n"
        f"How many distinct directed paths are there from node 0 to node "
        f"{n - 1}? (Each path is a sequence of edges that begins at 0 and "
        "ends at n-1.) Return only the integer count."
    )
    return (
        prompt,
        str(count),
        _exact_or_numeric_match(str(count)),
        {"n": n, "edges": edges, "count": count},
    )


# -------------------------------------------------------------------------
# Tier table
# -------------------------------------------------------------------------

GENERATORS: list[tuple[float, Callable[[np.random.Generator], tuple[str, str, Callable[[str], bool], dict]]]] = [
    (0.05, _gen_t0_print),
    (0.15, _gen_t1_oneliner),
    (0.25, _gen_t2_call),
    (0.40, _gen_t3_snippet),
    (0.55, _gen_t4_implement_simple),
    (0.70, _gen_t5_implement_algorithm),
    (0.85, _gen_t6_bugfix),
    (0.95, _gen_t7_compose),
    # T_extreme — target <15% on frontier
    (0.975, _gen_text_dp_count),
    (0.985, _gen_text_knapsack_mod),
    (0.995, _gen_text_dag_paths),
]


@dataclass(slots=True)
class CodeFamily:
    name: str = "code"
    description: str = (
        "Programming problems with verifiable outputs: prediction tasks plus "
        "implement/bug-fix tasks executed in a sandboxed subprocess."
    )

    def generate(
        self,
        n: int,
        seed: int,
        difficulty_range: tuple[float, float] = (0.0, 1.0),
    ) -> list[Task]:
        rng = child_rng(seed, "code.generate")
        lo, hi = difficulty_range
        valid = [(d, g) for d, g in GENERATORS if lo <= d <= hi]
        if not valid:
            valid = GENERATORS
        tasks: list[Task] = []
        for i in range(n):
            d, gen = valid[rng.integers(0, len(valid))]
            sub_rng = np.random.default_rng(child_seed(seed, f"code.{i}"))
            prompt, expected, verifier, meta = gen(sub_rng)
            tasks.append(
                Task(
                    task_id=f"code-{seed}-{i:04d}",
                    family="code",
                    difficulty=float(d),
                    prompt=prompt,
                    verifier=verifier,
                    reference_answer=expected,
                    metadata={"generator": gen.__name__, "tier": d, **meta},
                    estimated_seconds=15.0 + 90.0 * d,
                )
            )
        return tasks

    def reference_score(self, task, response):
        return None  # mechanical only


FAMILY = CodeFamily
