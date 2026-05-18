"""Stub generators used by family modules during scaffolding.

Each real family overrides these. Keeping a minimal stub means
``from prophet.families import get_family`` works for any registered name
even before its full generator is implemented; tests for that family will
just see deterministic, easy tasks.
"""

from __future__ import annotations

from dataclasses import dataclass

from prophet.engine.types import Task


def _yes_verifier(answer: str) -> bool:
    return isinstance(answer, str) and answer.strip().lower() in {"yes", "y", "true", "1"}


@dataclass(slots=True)
class StubFamily:
    name: str = "stub"
    description: str = "Placeholder family — real implementation pending."

    def generate(
        self,
        n: int,
        seed: int,
        difficulty_range: tuple[float, float] = (0.0, 1.0),
    ) -> list[Task]:
        return [
            Task(
                task_id=f"{self.name}-stub-{seed}-{i:04d}",
                family=self.name,
                difficulty=0.5,
                prompt=f"Is 2 + 2 = 4? Answer 'yes' or 'no'. (Stub family {self.name} task {i})",
                verifier=_yes_verifier,
                reference_answer="yes",
                metadata={"stub": True},
                estimated_seconds=5.0,
            )
            for i in range(n)
        ]

    def reference_score(self, task, response):
        return None


def make_stub_family(name: str) -> StubFamily:
    return StubFamily(name=name, description=f"Placeholder {name} family")
