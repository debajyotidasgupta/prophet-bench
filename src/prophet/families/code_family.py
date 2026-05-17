"""Placeholder code family — to be replaced with full implementation."""
from __future__ import annotations
from prophet.families._stub import make_stub_family
class _Family:
    name = "code"
    description = "Placeholder code family."
    def __init__(self):
        self._stub = make_stub_family("code")
    def generate(self, n, seed, difficulty_range=(0.0, 1.0)):
        return self._stub.generate(n, seed, difficulty_range)
    def reference_score(self, task, response):
        return None
FAMILY = _Family
