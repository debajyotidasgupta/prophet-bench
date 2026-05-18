"""Scientific family — physics / chemistry / biology problems with closed-form numeric answers.

Procedurally generated, every problem has a single numeric ground truth checked
with a relative-tolerance verifier (default 1e-3). This keeps grading entirely
mechanical (no LLM-judge) while exercising domain knowledge that is genuinely
different from "math word problem" pattern matching.

Difficulty tiers:
  T0 (0.10) — unit conversion: joules ↔ calories.
  T1 (0.20) — stoichiometry: mol ratios in a (pre-balanced) equation.
  T2 (0.30) — kinematics: v = u + at.
  T3 (0.45) — simple genetics: P(homozygous recessive) from a Punnett square.
  T4 (0.60) — acid-base titration: moles + dilution to find concentration.
  T5 (0.75) — thermodynamics: Q = m·c·ΔT.
  T6 (0.90) — chemical equilibrium: Kc from equilibrium concentrations.

Notes:
  * The verifier parses the *last* numeric token in the response and compares it
    to the expected ground-truth float with relative tolerance.
  * `reference_answer` is the canonical numeric string ("12.34"); useful for the
    OracleAgent which reads it directly.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from prophet.engine.types import Task
from prophet.utils.seed import child_rng, child_seed

# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

# Match (signed) decimal numbers, optionally with scientific notation.
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")


def _last_number(s: str) -> float | None:
    """Extract the last numeric token from a free-form answer string."""
    if not isinstance(s, str):
        return None
    cleaned = s.replace(",", "")
    # Drop common wrappers.
    cleaned = cleaned.replace("$", "").replace("**", "")
    cleaned = re.sub(r"\\boxed\{([^}]*)\}", r"\1", cleaned)
    matches = _NUM_RE.findall(cleaned)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def _numeric_verifier(expected: float, rel_tol: float = 1e-3, abs_tol: float = 1e-6) -> Callable[[str], bool]:
    """Build a relative-tolerance verifier for numeric answers."""

    def _verify(answer: str) -> bool:
        got = _last_number(answer)
        if got is None:
            return False
        # Use math.isclose with relative + absolute fallback for near-zero cases.
        return math.isclose(got, expected, rel_tol=rel_tol, abs_tol=abs_tol)

    return _verify


def _fmt(x: float) -> str:
    """Format a float so the OracleAgent emits a value the verifier will accept."""
    # Use repr-like format with enough precision but keep it readable.
    if abs(x) >= 1e6 or (x != 0 and abs(x) < 1e-3):
        return f"{x:.6g}"
    return f"{x:.4f}".rstrip("0").rstrip(".") if "." in f"{x:.4f}" else f"{x:.4f}"


# ---------------------------------------------------------------------------
# Tier generators — each returns (prompt, expected_float)
# ---------------------------------------------------------------------------

# 1 calorie = 4.184 joules (thermochemical: 4.184 exactly).
_CAL_TO_J = 4.184


def _gen_t0_unit_conversion(rng: np.random.Generator) -> tuple[str, float]:
    """Convert between joules and calories."""
    direction = rng.choice(["j_to_cal", "cal_to_j"])
    if direction == "j_to_cal":
        joules = float(rng.integers(50, 5000))
        ans = joules / _CAL_TO_J
        prompt = (
            f"Convert {joules:.0f} joules to calories. Use 1 calorie = 4.184 joules. "
            f"Give a numeric answer in calories."
        )
        return prompt, ans
    cal = float(rng.integers(10, 1000))
    ans = cal * _CAL_TO_J
    prompt = (
        f"Convert {cal:.0f} calories to joules. Use 1 calorie = 4.184 joules. "
        f"Give a numeric answer in joules."
    )
    return prompt, ans


def _gen_t1_stoichiometry(rng: np.random.Generator) -> tuple[str, float]:
    """Mol-ratio: how many mol of B form from M mol of A, given balanced eq."""
    # Choose from a small set of canonical balanced equations.
    eqs = [
        # (A formula, A coef, B formula, B coef, full equation string)
        ("H2", 2, "H2O", 2, "2 H2 + O2 -> 2 H2O"),
        ("N2", 1, "NH3", 2, "N2 + 3 H2 -> 2 NH3"),
        ("C", 1, "CO2", 1, "C + O2 -> CO2"),
        ("CH4", 1, "CO2", 1, "CH4 + 2 O2 -> CO2 + 2 H2O"),
        ("Fe", 4, "Fe2O3", 2, "4 Fe + 3 O2 -> 2 Fe2O3"),
    ]
    idx = int(rng.integers(0, len(eqs)))
    a, ca, b, cb, eq = eqs[idx]
    mol_a = float(rng.integers(1, 12))
    ans = mol_a * (cb / ca)
    prompt = (
        f"Given the balanced reaction {eq}, how many moles of {b} are produced "
        f"from {mol_a:.0f} moles of {a}? Give a numeric answer in moles."
    )
    return prompt, ans


def _gen_t2_kinematics(rng: np.random.Generator) -> tuple[str, float]:
    """v = u + a*t"""
    u = float(rng.integers(0, 30))  # m/s
    a = float(rng.integers(1, 15))  # m/s^2
    t = float(rng.integers(2, 20))  # s
    v = u + a * t
    prompt = (
        f"An object starts with velocity u = {u:.0f} m/s and accelerates at "
        f"a = {a:.0f} m/s^2 for t = {t:.0f} seconds. Using v = u + a*t, what is "
        f"its final velocity in m/s? Give a numeric answer."
    )
    return prompt, v


def _gen_t3_genetics(rng: np.random.Generator) -> tuple[str, float]:
    """Punnett square: probability of homozygous recessive (or related)."""
    # Two heterozygous parents (Aa x Aa) → 1/4 = 0.25 homozygous recessive.
    # We vary the question slightly: AA x Aa, Aa x Aa, Aa x aa.
    scenarios = [
        # (parent1, parent2, p_homozygous_recessive)
        ("Aa", "Aa", 0.25),
        ("Aa", "aa", 0.50),
        ("AA", "Aa", 0.00),
        ("aa", "Aa", 0.50),
        ("Aa", "Aa", 0.25),
    ]
    idx = int(rng.integers(0, len(scenarios)))
    p1, p2, ans = scenarios[idx]
    prompt = (
        f"Two parents have genotypes {p1} and {p2} for a single gene. "
        f"What is the probability (as a decimal fraction) that a randomly chosen "
        f"offspring is homozygous recessive (aa)? Give a numeric answer between 0 and 1."
    )
    return prompt, ans


def _gen_t4_titration(rng: np.random.Generator) -> tuple[str, float]:
    """Acid-base titration:
       Given V_acid mL of c_acid M acid neutralized by V_base mL of base,
       what is the concentration of the base (assume 1:1 stoichiometry)?
       moles_acid = c_acid * V_acid/1000
       c_base = moles_acid / (V_base/1000)
    """
    c_acid = float(rng.integers(1, 10)) * 0.1  # 0.1 - 0.9 M
    v_acid = float(rng.integers(10, 50))  # mL
    v_base = float(rng.integers(10, 80))  # mL
    moles_acid = c_acid * v_acid / 1000.0
    c_base = moles_acid / (v_base / 1000.0)
    prompt = (
        f"In a 1:1 acid-base titration, {v_acid:.0f} mL of {c_acid:.2f} M HCl is "
        f"fully neutralized by {v_base:.0f} mL of NaOH. What is the concentration "
        f"of the NaOH solution in mol/L? Give a numeric answer."
    )
    return prompt, c_base


def _gen_t5_thermo(rng: np.random.Generator) -> tuple[str, float]:
    """Q = m * c * dT
       Compute Q (J) given m (g), c (J/g/K), and dT (K)."""
    # specific heats (J/g/K)
    materials = [
        ("water", 4.184),
        ("aluminum", 0.897),
        ("copper", 0.385),
        ("iron", 0.449),
        ("glass", 0.840),
    ]
    name, c = materials[int(rng.integers(0, len(materials)))]
    m = float(rng.integers(20, 500))
    dT = float(rng.integers(5, 80))
    Q = m * c * dT
    prompt = (
        f"How much heat (in joules) is required to raise the temperature of "
        f"{m:.0f} g of {name} (specific heat capacity c = {c} J/g/K) by "
        f"{dT:.0f} K? Use Q = m*c*deltaT. Give a numeric answer in joules."
    )
    return prompt, Q


def _gen_t6_equilibrium(rng: np.random.Generator) -> tuple[str, float]:
    """Kc for a reaction A + B <-> C + D from equilibrium concentrations.
       Kc = [C][D] / ([A][B])"""
    # Concentrations all in mol/L
    a = float(rng.integers(1, 9)) * 0.1
    b = float(rng.integers(1, 9)) * 0.1
    c = float(rng.integers(1, 9)) * 0.1
    d = float(rng.integers(1, 9)) * 0.1
    Kc = (c * d) / (a * b)
    prompt = (
        f"For the equilibrium reaction A + B <-> C + D, the equilibrium "
        f"concentrations are [A] = {a:.2f} M, [B] = {b:.2f} M, "
        f"[C] = {c:.2f} M, [D] = {d:.2f} M. Compute the equilibrium constant "
        f"Kc. Give a numeric answer."
    )
    return prompt, Kc


GENERATORS: list[tuple[float, Callable[[np.random.Generator], tuple[str, float]]]] = [
    (0.10, _gen_t0_unit_conversion),
    (0.20, _gen_t1_stoichiometry),
    (0.30, _gen_t2_kinematics),
    (0.45, _gen_t3_genetics),
    (0.60, _gen_t4_titration),
    (0.75, _gen_t5_thermo),
    (0.90, _gen_t6_equilibrium),
]


@dataclass(slots=True)
class ScientificFamily:
    name: str = "scientific"
    description: str = (
        "Procedurally generated science problems (physics / chemistry / biology) "
        "with closed-form numeric answers and relative-tolerance verifiers."
    )

    def generate(
        self,
        n: int,
        seed: int,
        difficulty_range: tuple[float, float] = (0.0, 1.0),
    ) -> list[Task]:
        rng = child_rng(seed, "scientific.generate")
        lo, hi = difficulty_range
        valid = [(d, g) for d, g in GENERATORS if lo <= d <= hi]
        if not valid:
            valid = GENERATORS
        tasks: list[Task] = []
        for i in range(n):
            d, gen = valid[int(rng.integers(0, len(valid)))]
            sub_rng = np.random.default_rng(child_seed(seed, f"scientific.{i}"))
            prompt, expected = gen(sub_rng)
            tasks.append(
                Task(
                    task_id=f"scientific-{seed}-{i:04d}",
                    family="scientific",
                    difficulty=float(d),
                    prompt=prompt,
                    verifier=_numeric_verifier(expected),
                    reference_answer=_fmt(expected),
                    metadata={"generator": gen.__name__, "tier": d, "expected_value": expected},
                    estimated_seconds=25.0 + 50.0 * d,
                )
            )
        return tasks

    def reference_score(self, task, response):
        return None


FAMILY = ScientificFamily
