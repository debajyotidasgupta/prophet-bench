"""Unit tests for the four newly-implemented task families.

Each family gets four tests:
  1. generate(n, seed) returns exactly n tasks with the requested family tag.
  2. Generation is deterministic — same seed → identical reference_answer.
  3. Reference-answer round-trip: the verifier accepts the reference answer.
  4. Verifier robustness: rejects obviously-wrong outputs.

These are *mechanical* tests — no LLM calls, no network access.
"""

from __future__ import annotations

import json

import pytest

from prophet.families import get_family


FAMILY_NAMES = ["tools", "browser", "multimodal", "data"]


# ---------- shared helpers --------------------------------------------------


def _gen(name: str, n: int = 14, seed: int = 12345):
    return get_family(name).generate(n=n, seed=seed)


# =============================================================================
# tools_family
# =============================================================================


def test_tools_generate_n_and_family_tag() -> None:
    tasks = _gen("tools", n=12, seed=11)
    assert len(tasks) == 12
    assert all(t.family == "tools" for t in tasks)
    # All tier difficulties land in [0.05, 0.95]
    assert all(0.05 <= t.difficulty <= 0.95 for t in tasks)
    # Each prompt should mention "tool" so a model sees the framing.
    assert all("tool" in t.prompt.lower() for t in tasks)


def test_tools_generation_is_deterministic() -> None:
    a = _gen("tools", n=10, seed=7)
    b = _gen("tools", n=10, seed=7)
    assert [t.reference_answer for t in a] == [t.reference_answer for t in b]
    assert [t.task_id for t in a] == [t.task_id for t in b]


def test_tools_reference_answer_verifies() -> None:
    tasks = _gen("tools", n=30, seed=3)
    ok = sum(1 for t in tasks if t.verifier(t.reference_answer))
    assert ok == len(tasks)


def test_tools_verifier_rejects_garbage() -> None:
    tasks = _gen("tools", n=8, seed=9)
    for t in tasks:
        assert not t.verifier("not json at all")
        assert not t.verifier("[]")
        assert not t.verifier('[{"tool":"add","args":{"a":"oops"}}]')


# =============================================================================
# browser_family
# =============================================================================


def test_browser_generate_n_and_family_tag() -> None:
    tasks = _gen("browser", n=15, seed=22)
    assert len(tasks) == 15
    assert all(t.family == "browser" for t in tasks)
    # Every prompt must contain the simulated website marker.
    assert all("SIMULATED WEBSITE" in t.prompt for t in tasks)


def test_browser_generation_is_deterministic() -> None:
    a = _gen("browser", n=8, seed=42)
    b = _gen("browser", n=8, seed=42)
    assert [t.reference_answer for t in a] == [t.reference_answer for t in b]


def test_browser_reference_answer_verifies() -> None:
    tasks = _gen("browser", n=25, seed=5)
    ok = sum(1 for t in tasks if t.verifier(t.reference_answer))
    assert ok == len(tasks)


def test_browser_verifier_rejects_wrong_answer() -> None:
    tasks = _gen("browser", n=6, seed=8)
    for t in tasks:
        # an empty action list cannot answer
        assert not t.verifier("[]")
        # plausible JSON with a clearly wrong final answer
        bad = '[{"action":"answer","value":"definitely-wrong-xyz"}]'
        assert not t.verifier(bad) or t.metadata.get("expected_value") == "definitely-wrong-xyz"


# =============================================================================
# multimodal_family
# =============================================================================


def test_multimodal_generate_n_and_family_tag() -> None:
    tasks = _gen("multimodal", n=14, seed=15)
    assert len(tasks) == 14
    assert all(t.family == "multimodal" for t in tasks)
    # Reference answers should be short (single token).
    assert all(len(t.reference_answer) <= 12 for t in tasks)


def test_multimodal_generation_is_deterministic() -> None:
    a = _gen("multimodal", n=10, seed=88)
    b = _gen("multimodal", n=10, seed=88)
    assert [t.reference_answer for t in a] == [t.reference_answer for t in b]
    assert [t.prompt for t in a] == [t.prompt for t in b]


def test_multimodal_reference_answer_verifies() -> None:
    tasks = _gen("multimodal", n=40, seed=21)
    ok = sum(1 for t in tasks if t.verifier(t.reference_answer))
    assert ok == len(tasks)


def test_multimodal_verifier_rejects_unrelated_answer() -> None:
    tasks = _gen("multimodal", n=10, seed=2)
    for t in tasks:
        assert not t.verifier("zzzz-not-a-valid-answer")


# =============================================================================
# data_family
# =============================================================================


def test_data_generate_n_and_family_tag() -> None:
    tasks = _gen("data", n=12, seed=14)
    assert len(tasks) == 12
    assert all(t.family == "data" for t in tasks)
    # Each prompt must embed a CSV code block.
    assert all("```csv" in t.prompt for t in tasks)


def test_data_generation_is_deterministic() -> None:
    a = _gen("data", n=10, seed=64)
    b = _gen("data", n=10, seed=64)
    assert [t.reference_answer for t in a] == [t.reference_answer for t in b]


def test_data_reference_answer_verifies() -> None:
    tasks = _gen("data", n=40, seed=17)
    ok = sum(1 for t in tasks if t.verifier(t.reference_answer))
    assert ok == len(tasks)


def test_data_verifier_rejects_wrong_number() -> None:
    tasks = _gen("data", n=10, seed=6)
    for t in tasks:
        # Off-by-a-huge-amount answers must be rejected.
        bogus = "9999999999"
        # int / numeric verifiers should reject this unless the true answer happens to match
        if t.reference_answer == bogus:
            continue
        # for string verifier, this is just a different word
        if t.metadata.get("kind") == "string":
            assert not t.verifier(bogus)
        else:
            assert not t.verifier(bogus)


# =============================================================================
# Cross-family sanity
# =============================================================================


@pytest.mark.parametrize("name", FAMILY_NAMES)
def test_family_supports_oracle_via_reference_answer(name: str) -> None:
    """Oracle agents return task.reference_answer; verifier must accept it.

    This is the same property tested above per-family, but parameterized to
    catch regressions across all four at once.
    """
    tasks = get_family(name).generate(n=10, seed=2024)
    for t in tasks:
        assert t.reference_answer is not None
        assert t.verifier(t.reference_answer), f"{name}/{t.task_id} did not self-verify"
