"""Scoring: per-task payoff math and aggregate calibration metrics.

Primary metrics are formula-based (no LLM judge). The single exception is
the open-ended writing family's quality score, which contributes only to the
attempt-outcome boolean — confidence calibration is still measured against
that mechanically-thresholded boolean.

All aggregate metrics support bootstrap confidence intervals.

Key references (anchored in our literature mining):
  - Brier (1950); Gneiting & Raftery (2007) — proper scoring rules.
  - Guo et al. (2017) — temperature scaling, ECE.
  - Beyond pass@1 (arxiv:2603.29231) — Reliability Decay Curve, MOP.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from prophet.engine.types import (
    AgentResponse,
    DecisionMode,
    MarketOffer,
    Outcome,
)

EPS = 1e-9


# -------------------------------------------------------------------------
# Per-task payoff
# -------------------------------------------------------------------------

def compute_payoff(
    response: AgentResponse,
    success: bool | None,
    offer: MarketOffer,
) -> tuple[float, float, float]:
    """Return (payoff_attempt, payoff_calib, payoff_total) for one task.

    Rules:
      PASS     → attempt = 0,        calib = 0,                       total = -delta_pass
      TAKE     → attempt = V·y - C·(1-y),  calib = 0,                  total = attempt
      QUOTE    → attempt = V·y - C·(1-y),  calib = κ·(1 - 4·(p-y)²),   total = attempt + calib

    `success` must be None iff mode == PASS.
    """
    mode = response.mode
    if mode == DecisionMode.PASS:
        if success is not None:
            raise ValueError("PASS implies no graded outcome; success must be None.")
        return 0.0, 0.0, -float(offer.delta_pass)
    if success is None:
        raise ValueError(f"Mode {mode} requires success ∈ {{True, False}}.")
    y = 1.0 if success else 0.0
    attempt = offer.V_success * y - offer.C_failure * (1.0 - y)
    if mode == DecisionMode.TAKE:
        return attempt, 0.0, attempt
    # QUOTE
    p = float(np.clip(response.confidence, 0.0, 1.0))
    brier = (p - y) ** 2
    calib = offer.kappa_calib * (1.0 - 4.0 * brier)
    return attempt, calib, attempt + calib


# -------------------------------------------------------------------------
# Aggregate calibration metrics
# -------------------------------------------------------------------------

def brier_score(confidences: Sequence[float], outcomes: Sequence[int]) -> float:
    """Mean squared error between confidence and outcome."""
    c = np.asarray(confidences, dtype=float)
    o = np.asarray(outcomes, dtype=float)
    return float(np.mean((c - o) ** 2))


def log_score(confidences: Sequence[float], outcomes: Sequence[int]) -> float:
    """Negative log-likelihood (lower is better). Clipped for numerical stability."""
    c = np.clip(np.asarray(confidences, dtype=float), EPS, 1 - EPS)
    o = np.asarray(outcomes, dtype=float)
    return float(-np.mean(o * np.log(c) + (1 - o) * np.log(1 - c)))


def ece(
    confidences: Sequence[float],
    outcomes: Sequence[int],
    n_bins: int = 10,
    norm: str = "l1",
) -> float:
    """Expected Calibration Error (equal-width bins, weighted by support).

    norm: 'l1' (default) or 'l2'.
    """
    c = np.asarray(confidences, dtype=float)
    o = np.asarray(outcomes, dtype=float)
    n = len(c)
    if n == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    err = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (c >= lo) & (c < hi if i < n_bins - 1 else c <= hi)
        if mask.sum() == 0:
            continue
        bc = float(c[mask].mean())
        ba = float(o[mask].mean())
        gap = abs(bc - ba) if norm == "l1" else (bc - ba) ** 2
        err += gap * mask.sum() / n
    return float(err)


def adaptive_ece(
    confidences: Sequence[float],
    outcomes: Sequence[int],
    n_bins: int = 15,
) -> float:
    """Equal-mass (quantile) binned ECE; more robust than equal-width on skewed P̂."""
    c = np.asarray(confidences, dtype=float)
    o = np.asarray(outcomes, dtype=float)
    n = len(c)
    if n == 0:
        return float("nan")
    order = np.argsort(c)
    c_sorted = c[order]
    o_sorted = o[order]
    bin_size = max(1, n // n_bins)
    err = 0.0
    used = 0
    for start in range(0, n, bin_size):
        end = min(start + bin_size, n)
        if end <= start:
            break
        bc = float(c_sorted[start:end].mean())
        ba = float(o_sorted[start:end].mean())
        err += abs(bc - ba) * (end - start) / n
        used += end - start
        if used >= n:
            break
    return float(err)


def calibration_curve(
    confidences: Sequence[float],
    outcomes: Sequence[int],
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (bin_centers, bin_accuracy, bin_support_count)."""
    c = np.asarray(confidences, dtype=float)
    o = np.asarray(outcomes, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2.0
    accs = np.full(n_bins, np.nan, dtype=float)
    counts = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (c >= lo) & (c < hi if i < n_bins - 1 else c <= hi)
        counts[i] = int(mask.sum())
        if counts[i] > 0:
            accs[i] = float(o[mask].mean())
    return centers, accs, counts


def reliability_diagram_data(
    confidences: Sequence[float],
    outcomes: Sequence[int],
    n_bins: int = 10,
) -> dict[str, np.ndarray]:
    """Convenience wrapper for plotting."""
    centers, accs, counts = calibration_curve(confidences, outcomes, n_bins)
    return {"bin_centers": centers, "bin_accuracy": accs, "bin_support": counts}


@dataclass(slots=True)
class MOPResult:
    """Model Overreach Point: difficulty bin where stated P̂ ≫ empirical reliability."""

    mop_difficulty: float | None  # midpoint of the offending bin, None if calibrated everywhere
    max_overreach: float  # max(P̂_bar - acc_bar) over bins
    bin_overreach: np.ndarray  # per-bin (P̂_bar - acc_bar)
    bin_centers: np.ndarray
    bin_support: np.ndarray


def model_overreach_point(
    confidences: Sequence[float],
    outcomes: Sequence[int],
    difficulties: Sequence[float],
    n_bins: int = 8,
    sigma_threshold: float = 2.0,
) -> MOPResult:
    """Find the *difficulty* bin where the agent over-promises by ≥ sigma_threshold·σ.

    Sigma is the binomial std-error of mean accuracy in that bin.
    """
    c = np.asarray(confidences, dtype=float)
    o = np.asarray(outcomes, dtype=float)
    d = np.asarray(difficulties, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2.0
    overreach = np.full(n_bins, np.nan, dtype=float)
    support = np.zeros(n_bins, dtype=int)
    mop = None
    max_o = -np.inf
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (d >= lo) & (d < hi if i < n_bins - 1 else d <= hi)
        n_i = int(mask.sum())
        support[i] = n_i
        if n_i < 5:
            continue
        bc = float(c[mask].mean())
        ba = float(o[mask].mean())
        sig = float(np.sqrt(max(ba * (1 - ba), 1e-9) / n_i))
        gap = bc - ba
        overreach[i] = gap
        if gap > max_o:
            max_o = gap
        if gap > sigma_threshold * sig and mop is None:
            mop = float(centers[i])
    return MOPResult(
        mop_difficulty=mop,
        max_overreach=float(max_o if np.isfinite(max_o) else 0.0),
        bin_overreach=overreach,
        bin_centers=centers,
        bin_support=support,
    )


# -------------------------------------------------------------------------
# Aggregate payoff and cost
# -------------------------------------------------------------------------

def total_payoff(outcomes: Iterable[Outcome]) -> float:
    return float(sum(o.payoff_total for o in outcomes))


def expected_payoff_per_dollar(outcomes: Iterable[Outcome]) -> float:
    """Headline economic ratio: net PROPHET-cents earned per real-dollar spent on inference.

    If total cost is zero (e.g. local CPU), return raw total payoff.
    """
    outs = list(outcomes)
    cost = sum(o.response.cost_usd for o in outs)
    payoff = total_payoff(outs)
    return payoff / cost if cost > 1e-9 else payoff


def abstention_precision(outcomes: Iterable[Outcome]) -> tuple[float, int, int]:
    """Of all PASSed outcomes, what fraction would have failed if attempted?

    Returns (precision, n_passed, n_would_fail). For PASS outcomes we cannot
    observe success directly. The benchmark therefore runs a *shadow attempt*
    for a held-out subsample of PASS instances using a deterministic forced
    re-attempt, OR estimates from family-average reliability at matched
    difficulty. The orchestrator populates `metadata["shadow_success"]` on
    PASS outcomes when available; this function uses that when present.
    """
    n_p = 0
    n_would_fail = 0
    for o in outcomes:
        if o.response.mode != DecisionMode.PASS:
            continue
        n_p += 1
        shadow = o.metadata.get("shadow_success") if o.metadata else None
        if shadow is None:
            shadow = o.response.metadata.get("shadow_success") if o.response.metadata else None
        if shadow is False:
            n_would_fail += 1
    if n_p == 0:
        return float("nan"), 0, 0
    return n_would_fail / n_p, n_p, n_would_fail


# -------------------------------------------------------------------------
# Bootstrap CI utility (used across analysis)
# -------------------------------------------------------------------------

def bootstrap_ci(
    fn,
    samples: Sequence,
    n_boot: int = 1000,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Bootstrap (mean, lower, upper) for a scalar function of samples.

    `fn(subsample)` should return a scalar. `samples` is any sequence of items
    sampleable by index.
    """
    rng = rng or np.random.default_rng(0)
    arr = np.asarray(samples, dtype=object)
    n = len(arr)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(fn(arr.tolist()))
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = float(fn(arr[idx].tolist()))
    lo = float(np.quantile(boots, (1 - ci) / 2))
    hi = float(np.quantile(boots, 1 - (1 - ci) / 2))
    return point, lo, hi
