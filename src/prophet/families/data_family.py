"""Data family — data analysis on procedurally-generated CSV tables.

Each task embeds one or two small CSV strings directly in the prompt
(≤ 30 rows). The agent must read the CSV mentally and return a single
numeric or short-string answer. The mechanical verifier compares with
small relative tolerance for floating-point answers and exact match for
integer / categorical answers.

Difficulty tiers:
  T0 (0.10) — single-cell lookup: "what is column X in row N?"
  T1 (0.25) — sum a numeric column.
  T2 (0.40) — mean / median / max of a column.
  T3 (0.55) — filter + count rows.
  T4 (0.70) — group-by + aggregate.
  T5 (0.85) — join two CSVs and compute an aggregate.
  T6 (0.95) — detect a simple statistical anomaly (outlier or monotonic
              trend break) — return either the index or a yes/no flag.
"""

from __future__ import annotations

import io
import re
import statistics
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from prophet.engine.types import Task
from prophet.utils.seed import child_rng, child_seed


# ---------------------------------------------------------------------------
# Verifier factories
# ---------------------------------------------------------------------------


def _make_numeric_verifier(expected: float, tol: float = 1e-3) -> Callable[[str], bool]:
    """Tolerant numeric match.

    Accepts the answer wrapped with prose. Extracts the last numeric token
    and compares with `tol` either absolute (for small numbers) or relative.
    """
    exp = float(expected)

    def _verify(answer: str) -> bool:
        if not isinstance(answer, str):
            return False
        s = answer.strip().replace(",", "")
        # Search last numeric token (with optional sign / decimal point)
        nums = re.findall(r"-?\d+(?:\.\d+)?", s)
        if not nums:
            return False
        try:
            got = float(nums[-1])
        except Exception:
            return False
        if abs(exp) < 1e-9:
            return abs(got) < tol
        return abs(got - exp) <= max(tol, tol * abs(exp))

    return _verify


def _make_int_verifier(expected: int) -> Callable[[str], bool]:
    exp = int(expected)

    def _verify(answer: str) -> bool:
        if not isinstance(answer, str):
            return False
        s = answer.strip().replace(",", "")
        nums = re.findall(r"-?\d+", s)
        if not nums:
            return False
        try:
            return int(nums[-1]) == exp
        except Exception:
            return False

    return _verify


def _make_string_verifier(expected: str) -> Callable[[str], bool]:
    exp = expected.strip().lower()

    def _verify(answer: str) -> bool:
        if not isinstance(answer, str):
            return False
        s = answer.strip().lower()
        s = re.sub(r"[*`\"'`]", "", s)
        s = s.rstrip(".,;:!?")
        lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
        if lines:
            s = lines[-1]
        if s == exp:
            return True
        toks = re.findall(r"[a-z0-9_\-]+", s)
        return bool(toks) and toks[-1] == exp

    return _verify


# ---------------------------------------------------------------------------
# CSV helpers (avoid pandas-at-import-time cost; we still keep stdlib only).
# ---------------------------------------------------------------------------


def _csv_dump(rows: list[dict[str, Any]], cols: list[str]) -> str:
    buf = io.StringIO()
    buf.write(",".join(cols) + "\n")
    for r in rows:
        buf.write(",".join(str(r[c]) for c in cols) + "\n")
    return buf.getvalue().rstrip("\n")


_CITIES = ["Paris", "Tokyo", "London", "Lisbon", "Madrid", "Berlin", "Rome", "Cairo"]
_DEPTS = ["eng", "ops", "sales", "hr", "marketing", "finance"]
_NAMES = ["alice", "bob", "carol", "dave", "eve", "frank", "grace", "heidi",
          "ivan", "judy", "karl", "linda", "mike", "nora", "olive", "pete"]


def _people(rng: np.random.Generator, n: int) -> list[dict[str, Any]]:
    if n <= len(_NAMES):
        names = list(rng.choice(_NAMES, size=n, replace=False))
    else:
        # name pool too small — append numeric suffixes to keep names unique
        base = list(_NAMES)
        rng.shuffle(base)
        extras = [f"{base[i % len(base)]}{i // len(base) + 2}" for i in range(n - len(base))]
        names = base + extras
    return [
        {
            "id": i + 1,
            "name": names[i],
            "age": int(rng.integers(20, 65)),
            "city": str(rng.choice(_CITIES)),
            "dept": str(rng.choice(_DEPTS)),
            "salary": int(rng.integers(40, 150)) * 1000,
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Tier generators
# ---------------------------------------------------------------------------


def _gen_t0(rng: np.random.Generator) -> dict[str, Any]:
    n = int(rng.integers(6, 12))
    rows = _people(rng, n)
    cols = ["id", "name", "age", "city", "dept", "salary"]
    target_row = int(rng.integers(0, n))
    target_col = str(rng.choice(["age", "city", "dept", "salary"]))
    expected = rows[target_row][target_col]
    desc = (
        f"In the CSV below, what is the value of column {target_col!r} in row "
        f"{target_row} (0-indexed, header excluded)? Return only the value."
    )
    return {
        "prompt": _wrap(desc, _csv_dump(rows, cols)),
        "expected": str(expected),
        "kind": "string" if target_col in {"city", "dept"} else "int",
        "tier": 0.10,
    }


def _gen_t1(rng: np.random.Generator) -> dict[str, Any]:
    n = int(rng.integers(8, 16))
    rows = _people(rng, n)
    cols = ["id", "name", "age", "city", "dept", "salary"]
    target = str(rng.choice(["age", "salary"]))
    expected = sum(r[target] for r in rows)
    desc = f"Sum the values in the {target!r} column. Return the integer total."
    return {
        "prompt": _wrap(desc, _csv_dump(rows, cols)),
        "expected": str(expected),
        "kind": "int",
        "tier": 0.25,
    }


def _gen_t2(rng: np.random.Generator) -> dict[str, Any]:
    n = int(rng.integers(8, 18))
    rows = _people(rng, n)
    cols = ["id", "name", "age", "city", "dept", "salary"]
    op = str(rng.choice(["mean", "median", "max"]))
    target = str(rng.choice(["age", "salary"]))
    vals = [r[target] for r in rows]
    if op == "mean":
        expected = sum(vals) / len(vals)
        kind = "numeric"
    elif op == "median":
        expected = float(statistics.median(vals))
        kind = "numeric"
    else:
        expected = max(vals)
        kind = "int"
    desc = (
        f"Compute the {op} of the {target!r} column. "
        f"Return the {'numeric value (any precision)' if kind=='numeric' else 'integer'}."
    )
    return {
        "prompt": _wrap(desc, _csv_dump(rows, cols)),
        "expected": str(expected),
        "kind": kind,
        "tier": 0.40,
    }


def _gen_t3(rng: np.random.Generator) -> dict[str, Any]:
    n = int(rng.integers(12, 25))
    rows = _people(rng, n)
    cols = ["id", "name", "age", "city", "dept", "salary"]
    age_thr = int(rng.integers(25, 50))
    target_city = str(rng.choice(_CITIES))
    matched = [r for r in rows if r["age"] > age_thr and r["city"] == target_city]
    expected = len(matched)
    desc = (
        f"How many rows have age > {age_thr} AND city = {target_city!r}? "
        f"Return the integer count."
    )
    return {
        "prompt": _wrap(desc, _csv_dump(rows, cols)),
        "expected": str(expected),
        "kind": "int",
        "tier": 0.55,
    }


def _gen_t4(rng: np.random.Generator) -> dict[str, Any]:
    n = int(rng.integers(12, 22))
    rows = _people(rng, n)
    cols = ["id", "name", "age", "city", "dept", "salary"]
    group_col = str(rng.choice(["dept", "city"]))
    metric = str(rng.choice(["salary", "age"]))
    op = str(rng.choice(["mean", "max"]))
    groups: dict[str, list[float]] = {}
    for r in rows:
        groups.setdefault(r[group_col], []).append(float(r[metric]))
    if op == "mean":
        agg = {k: sum(v) / len(v) for k, v in groups.items()}
    else:
        agg = {k: max(v) for k, v in groups.items()}
    # The answer is the group with the largest aggregate.
    winner = max(agg.items(), key=lambda kv: kv[1])[0]
    expected = winner
    desc = (
        f"Group the rows by {group_col!r} and compute the {op} of {metric!r} "
        f"per group. Return the group name with the largest {op}."
    )
    return {
        "prompt": _wrap(desc, _csv_dump(rows, cols)),
        "expected": str(expected),
        "kind": "string",
        "tier": 0.70,
    }


def _gen_t5(rng: np.random.Generator) -> dict[str, Any]:
    # Two CSVs: employees and salaries (by dept). Join + sum.
    n = int(rng.integers(8, 14))
    rows = _people(rng, n)
    cols = ["id", "name", "age", "city", "dept", "salary"]
    # bonus table per dept
    depts = sorted({r["dept"] for r in rows})
    bonus = {d: int(rng.integers(1, 10)) * 1000 for d in depts}
    bonus_rows = [{"dept": d, "bonus": bonus[d]} for d in depts]
    bonus_csv = _csv_dump(bonus_rows, ["dept", "bonus"])
    total = sum(r["salary"] + bonus[r["dept"]] for r in rows)
    expected = total
    desc = (
        "You are given two CSVs. Join them on the 'dept' column and compute "
        "the sum of (salary + bonus) across all employees. Return the integer."
    )
    body = (
        f"=== employees ===\n{_csv_dump(rows, cols)}\n\n"
        f"=== bonus_by_dept ===\n{bonus_csv}"
    )
    return {
        "prompt": _wrap(desc, body),
        "expected": str(expected),
        "kind": "int",
        "tier": 0.85,
    }


def _gen_t6(rng: np.random.Generator) -> dict[str, Any]:
    # Time-series with a single outlier; agent must return the index of the
    # outlier (or 'none' if the sequence is monotonic). We force an outlier.
    n = int(rng.integers(8, 16))
    base = int(rng.integers(50, 200))
    step = int(rng.integers(1, 10))
    series = [base + step * i for i in range(n)]
    # Inject a single outlier of magnitude ~ step * 20 at a random index.
    outlier_idx = int(rng.integers(1, n - 1))
    sign = 1 if rng.random() < 0.5 else -1
    series[outlier_idx] += sign * step * 20
    rows = [{"t": i, "value": v} for i, v in enumerate(series)]
    cols = ["t", "value"]
    expected = outlier_idx
    desc = (
        "The CSV below is a time series that is supposed to follow a roughly "
        "linear trend. Exactly one data point is an outlier. Return the index "
        "(value of column 't') of the outlier."
    )
    return {
        "prompt": _wrap(desc, _csv_dump(rows, cols)),
        "expected": str(expected),
        "kind": "int",
        "tier": 0.95,
    }


def _wrap(description: str, csv_block: str) -> str:
    return (
        "You are a data-analysis agent. Below is one or more CSV table(s). "
        "Read the data carefully and answer the task. Return only the answer "
        "value — no prose, no commentary.\n\n"
        f"```csv\n{csv_block}\n```\n\nTask: {description}"
    )


GENERATORS: list[tuple[float, Callable[[np.random.Generator], dict[str, Any]]]] = [
    (0.10, _gen_t0),
    (0.25, _gen_t1),
    (0.40, _gen_t2),
    (0.55, _gen_t3),
    (0.70, _gen_t4),
    (0.85, _gen_t5),
    (0.95, _gen_t6),
]


def _verifier_for(spec: dict[str, Any]) -> Callable[[str], bool]:
    kind = spec.get("kind", "int")
    expected = spec["expected"]
    if kind == "int":
        return _make_int_verifier(int(float(expected)))
    if kind == "numeric":
        return _make_numeric_verifier(float(expected), tol=1e-2)
    return _make_string_verifier(str(expected))


@dataclass(slots=True)
class DataFamily:
    name: str = "data"
    description: str = (
        "Data-analysis on procedurally-generated small CSV tables. Tiers cover "
        "lookup, aggregation, filtering, group-by, join, and outlier detection."
    )

    def generate(
        self,
        n: int,
        seed: int,
        difficulty_range: tuple[float, float] = (0.0, 1.0),
    ) -> list[Task]:
        rng = child_rng(seed, "data.generate")
        lo, hi = difficulty_range
        valid = [(d, g) for d, g in GENERATORS if lo <= d <= hi]
        if not valid:
            valid = GENERATORS
        tasks: list[Task] = []
        for i in range(n):
            d, gen = valid[rng.integers(0, len(valid))]
            sub = np.random.default_rng(child_seed(seed, f"data.{i}"))
            spec = gen(sub)
            tasks.append(
                Task(
                    task_id=f"data-{seed}-{i:04d}",
                    family="data",
                    difficulty=float(d),
                    prompt=spec["prompt"],
                    verifier=_verifier_for(spec),
                    reference_answer=str(spec["expected"]),
                    metadata={
                        "generator": gen.__name__,
                        "tier": d,
                        "kind": spec.get("kind", "int"),
                    },
                    estimated_seconds=25.0 + 60.0 * d,
                )
            )
        return tasks

    def reference_score(self, task, response):  # noqa: ANN001
        return None


FAMILY = DataFamily
