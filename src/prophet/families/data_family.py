"""Placeholder data family — to be replaced with full implementation."""
from __future__ import annotations
from prophet.families._stub import make_stub_family
class _Family:
    name = "data"
    description = "Placeholder data family."
    def __init__(self):
        self._stub = make_stub_family("data")
    def generate(self, n, seed, difficulty_range=(0.0, 1.0)):
        return self._stub.generate(n, seed, difficulty_range)
    def reference_score(self, task, response):
        return None
FAMILY = _Family
