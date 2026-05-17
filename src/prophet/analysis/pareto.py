"""Pareto frontier construction over agent metrics.

The PROPHET leaderboard is a frontier, not a scalar. Frontiers are computed
over (cost, payoff), (ECE, payoff), (cost, accuracy) — minimising what's
"better-when-smaller" (cost, ECE) and maximising the rest.

We also support 3-D frontiers via brute filtering — for ≤ 50 agents this is
fine; for more we'd switch to a proper non-dominated sort.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(slots=True)
class PointRecord:
    name: str
    payoff: float
    ece: float
    brier: float
    accuracy: float
    cost_usd: float
    extra: dict | None = None


def _dominates(a: dict[str, float], b: dict[str, float], maximise: set[str]) -> bool:
    """Return True if a Pareto-dominates b on the chosen axes.

    a dominates b iff a is no-worse on every axis and strictly better on ≥ 1.
    """
    strict = False
    for k in a:
        ka, kb = a[k], b[k]
        better = ka > kb if k in maximise else ka < kb
        worse = ka < kb if k in maximise else ka > kb
        if worse:
            return False
        if better:
            strict = True
    return strict


def pareto_front(
    records: Sequence[PointRecord],
    axes: tuple[str, ...] = ("payoff", "ece"),
    maximise: set[str] | None = None,
) -> list[PointRecord]:
    """Return Pareto-optimal records over the named axes.

    `maximise` is the set of axes where larger = better. Defaults:
      payoff / accuracy → maximise
      ece / brier / logloss / cost_usd → minimise
    """
    if maximise is None:
        maximise = {"payoff", "accuracy"}
    proj = []
    for r in records:
        proj.append({k: getattr(r, k) for k in axes})
    front: list[PointRecord] = []
    for i, ri in enumerate(records):
        ai = proj[i]
        dominated = False
        for j, rj in enumerate(records):
            if i == j:
                continue
            aj = proj[j]
            if _dominates(aj, ai, maximise):
                dominated = True
                break
        if not dominated:
            front.append(ri)
    return front


def hypervolume_2d(
    records: Sequence[PointRecord],
    axes: tuple[str, str] = ("payoff", "ece"),
    maximise: set[str] | None = None,
    reference: tuple[float, float] | None = None,
) -> float:
    """Simple 2-D hypervolume against a reference point.

    Larger = better leaderboard "area dominated" — a useful summary scalar
    over a frontier, but never the *headline* number.
    """
    if maximise is None:
        maximise = {"payoff", "accuracy"}
    front = pareto_front(records, axes=axes, maximise=maximise)
    if not front:
        return 0.0
    # Convert all axes to "maximise" sign convention
    pts = []
    for r in front:
        vx, vy = getattr(r, axes[0]), getattr(r, axes[1])
        if axes[0] not in maximise:
            vx = -vx
        if axes[1] not in maximise:
            vy = -vy
        pts.append((vx, vy))
    pts.sort()
    if reference is None:
        rx = min(p[0] for p in pts) - 1.0
        ry = min(p[1] for p in pts) - 1.0
    else:
        rx, ry = reference
        if axes[0] not in maximise:
            rx = -rx
        if axes[1] not in maximise:
            ry = -ry
    # Sweep-line hypervolume
    hv = 0.0
    y_prev = ry
    for x, y in sorted(pts, key=lambda p: -p[1]):  # descending y
        hv += (x - rx) * (y - y_prev)
        y_prev = y
    return float(max(0.0, hv))
