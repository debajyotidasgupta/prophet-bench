"""Seeded RNG helpers — keep entire runs reproducible.

Every family + procgen path should accept a `seed` and derive a child
generator using `child_rng(seed, label)`. The labels make collisions
impossible across families even with the same top-level cycle seed.
"""

from __future__ import annotations

import hashlib
import os
import random

import numpy as np


def child_rng(parent_seed: int, label: str) -> np.random.Generator:
    """Deterministic child PRNG: parent_seed × label → seed → Generator."""
    h = hashlib.sha256(f"{parent_seed}|{label}".encode()).digest()
    child = int.from_bytes(h[:8], "big", signed=False) % (2**32 - 1)
    return np.random.default_rng(child)


def child_seed(parent_seed: int, label: str) -> int:
    h = hashlib.sha256(f"{parent_seed}|{label}".encode()).digest()
    return int.from_bytes(h[:8], "big", signed=False) % (2**32 - 1)


def seed_everything(seed: int) -> None:
    """Best-effort seeding of common libraries."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
