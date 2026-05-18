"""Task families.

Each family lives in its own module and is registered here. Families are
expected to:
  * be pure-Python with no heavy dependencies at import time;
  * generate exactly N tasks for a given seed;
  * provide a mechanical verifier (or judge fallback);
  * report reference difficulty per task.

Public entry point: `get_family(name)` returns an instance.
"""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prophet.engine.types import TaskFamily

log = logging.getLogger("prophet.families")

# name -> dotted module path
REGISTRY: dict[str, str] = {
    "math": "prophet.families.math_family",
    "code": "prophet.families.code_family",
    "knowledge": "prophet.families.knowledge_family",
    "reasoning": "prophet.families.reasoning_family",
    "writing": "prophet.families.writing_family",
    "tools": "prophet.families.tools_family",
    "browser": "prophet.families.browser_family",
    "multimodal": "prophet.families.multimodal_family",
    "data": "prophet.families.data_family",
    "scientific": "prophet.families.scientific_family",
    "multilingual": "prophet.families.multilingual_family",
    "safety": "prophet.families.safety_family",
}


def get_family(name: str) -> TaskFamily:
    name = name.strip().lower()
    if name not in REGISTRY:
        raise KeyError(f"Unknown family {name!r}. Known: {list(REGISTRY)}")
    mod = importlib.import_module(REGISTRY[name])
    cls = mod.FAMILY
    return cls()


def list_families() -> list[str]:
    return list(REGISTRY.keys())
