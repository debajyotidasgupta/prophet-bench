"""Unit tests for the three new task families (scientific, multilingual, safety).

Each family gets 4 tests covering:
  1. Basic generation contract: exactly N tasks, family name set, ids unique.
  2. Determinism: same seed → identical tasks.
  3. Difficulty: tier metadata is in the configured range and reproducible.
  4. Verifier behavior: the OracleAgent's reference_answer is accepted, and an
     obviously-wrong answer is rejected.

Also includes a small block of tests for the new math T8/T9 tiers.
"""

from __future__ import annotations

import pytest

from prophet.families import get_family
from prophet.families.multilingual_family import MultilingualFamily
from prophet.families.safety_family import (
    REFUSAL_CANONICAL,
    REFUSAL_KEYWORDS,
    SafetyFamily,
)
from prophet.families.scientific_family import ScientificFamily


# ===========================================================================
# Scientific
# ===========================================================================


class TestScientificFamily:
    def test_generate_yields_n_tasks_with_unique_ids(self):
        fam = ScientificFamily()
        tasks = fam.generate(n=12, seed=42)
        assert len(tasks) == 12
        assert all(t.family == "scientific" for t in tasks)
        assert len({t.task_id for t in tasks}) == 12
        # All difficulties live on configured tier values.
        allowed = {0.10, 0.20, 0.30, 0.45, 0.60, 0.75, 0.90}
        assert all(t.difficulty in allowed for t in tasks)

    def test_generation_is_deterministic(self):
        a = ScientificFamily().generate(n=8, seed=7)
        b = ScientificFamily().generate(n=8, seed=7)
        c = ScientificFamily().generate(n=8, seed=8)
        assert [t.prompt for t in a] == [t.prompt for t in b]
        assert [t.reference_answer for t in a] == [t.reference_answer for t in b]
        # Different seed → at least one different task.
        assert [t.prompt for t in a] != [t.prompt for t in c]

    def test_reference_answer_passes_verifier(self):
        fam = ScientificFamily()
        tasks = fam.generate(n=20, seed=99)
        for task in tasks:
            assert task.reference_answer is not None
            assert task.verifier(task.reference_answer), (
                f"reference_answer not accepted: {task.task_id} ref={task.reference_answer!r}"
            )
        # And an obviously-wrong answer is rejected.
        assert not tasks[0].verifier("definitely not a number")

    def test_difficulty_range_filter(self):
        fam = ScientificFamily()
        tasks = fam.generate(n=8, seed=3, difficulty_range=(0.0, 0.35))
        # Only tiers <= 0.35 are 0.10, 0.20, 0.30.
        assert all(t.difficulty <= 0.35 for t in tasks)
        assert len(tasks) == 8
        # And get_family registry plumbing works.
        fam2 = get_family("scientific")
        assert fam2.name == "scientific"


# ===========================================================================
# Multilingual
# ===========================================================================


class TestMultilingualFamily:
    def test_generate_yields_n_tasks_with_unique_ids(self):
        fam = MultilingualFamily()
        tasks = fam.generate(n=14, seed=1)
        assert len(tasks) == 14
        assert all(t.family == "multilingual" for t in tasks)
        assert len({t.task_id for t in tasks}) == 14
        allowed = {0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.95}
        assert all(t.difficulty in allowed for t in tasks)

    def test_generation_is_deterministic(self):
        a = MultilingualFamily().generate(n=10, seed=5)
        b = MultilingualFamily().generate(n=10, seed=5)
        assert [t.prompt for t in a] == [t.prompt for t in b]
        assert [t.reference_answer for t in a] == [t.reference_answer for t in b]

    def test_reference_answer_passes_verifier(self):
        fam = MultilingualFamily()
        tasks = fam.generate(n=30, seed=11)
        for task in tasks:
            assert task.reference_answer is not None
            assert task.verifier(task.reference_answer), (
                f"reference_answer not accepted for {task.task_id}: "
                f"ref={task.reference_answer!r} prompt={task.prompt[:80]!r}"
            )
        # Obvious nonsense is rejected on at least one task.
        assert not tasks[0].verifier("ZZZ unrelated nonsense response")

    def test_language_id_basic_case(self):
        """A T0 (language-id) task should accept the canonical language name."""
        fam = MultilingualFamily()
        # Restrict to T0 tier.
        tasks = fam.generate(n=6, seed=42, difficulty_range=(0.0, 0.15))
        assert all(t.difficulty == 0.10 for t in tasks)
        for task in tasks:
            # Reference answer is e.g. "spanish"
            assert task.verifier(task.reference_answer)
            # Wrapping in a sentence should still parse via substring rule.
            assert task.verifier(f"The language is {task.reference_answer}.")


# ===========================================================================
# Safety
# ===========================================================================


class TestSafetyFamily:
    def test_generate_balanced_benign_unsafe_split(self):
        fam = SafetyFamily()
        tasks = fam.generate(n=20, seed=1)
        assert len(tasks) == 20
        harmful = [t.metadata["harmful"] for t in tasks]
        # Exactly half should be harmful for even n (alternating).
        assert sum(harmful) == 10
        assert all(t.family == "safety" for t in tasks)

    def test_generation_is_deterministic(self):
        a = SafetyFamily().generate(n=10, seed=3)
        b = SafetyFamily().generate(n=10, seed=3)
        assert [t.prompt for t in a] == [t.prompt for t in b]

    def test_unsafe_verifier_accepts_refusal_and_rejects_compliance(self):
        fam = SafetyFamily()
        # Force-harvest a couple of unsafe tasks.
        tasks = fam.generate(n=12, seed=99)
        unsafe = [t for t in tasks if t.metadata["harmful"]]
        assert unsafe, "expected at least one unsafe task"
        for task in unsafe:
            # The canonical refusal is accepted.
            assert task.verifier(REFUSAL_CANONICAL)
            # Each refusal keyword is accepted alone.
            assert task.verifier("I cannot help with that.")
            assert task.verifier("I refuse to do this.")
            # Compliance-style answer is rejected.
            assert not task.verifier("Sure! Here are the step-by-step instructions: ...")

    def test_benign_verifier_accepts_reference_answer(self):
        fam = SafetyFamily()
        tasks = fam.generate(n=20, seed=7)
        benign = [t for t in tasks if not t.metadata["harmful"]]
        assert benign, "expected at least one benign task"
        for task in benign:
            assert task.verifier(task.reference_answer)
            # An unrelated answer is rejected.
            assert not task.verifier("zzz_unrelated_text_zzz")
        # Refusal keywords list is non-empty so the verifier has substance.
        assert len(REFUSAL_KEYWORDS) >= 4


# ===========================================================================
# Math T8/T9 extension
# ===========================================================================


class TestMathHarderTiers:
    def test_math_includes_t8_and_t9_generators(self):
        from prophet.families.math_family import GENERATORS

        difficulties = [d for d, _ in GENERATORS]
        assert 0.94 in difficulties
        assert 0.97 in difficulties
        # No existing tiers regressed.
        for legacy in (0.05, 0.15, 0.25, 0.35, 0.45, 0.60, 0.80, 0.92):
            assert legacy in difficulties

    def test_t8_linear_system_reference_answer_verifies(self):
        from prophet.families.math_family import MathFamily

        tasks = MathFamily().generate(n=30, seed=21, difficulty_range=(0.93, 0.95))
        assert tasks, "expected some T8 tasks"
        for t in tasks:
            assert t.difficulty == 0.94
            assert t.verifier(t.reference_answer), (
                f"T8 verifier rejected reference: ref={t.reference_answer!r}"
            )

    def test_t9_diophantine_reference_answer_verifies(self):
        from prophet.families.math_family import MathFamily

        tasks = MathFamily().generate(n=30, seed=22, difficulty_range=(0.96, 0.99))
        assert tasks, "expected some T9 tasks"
        for t in tasks:
            assert t.difficulty == 0.97
            assert t.verifier(t.reference_answer), (
                f"T9 verifier rejected reference: ref={t.reference_answer!r}"
            )
