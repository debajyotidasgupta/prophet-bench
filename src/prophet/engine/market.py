"""Market-maker: turns a task into a MarketOffer.

The market is deterministic given (task_id, market_seed). In production the
market_seed rotates daily so agents cannot pre-train against a fixed pricing
schedule; for reproducibility we expose the seed and freeze it per
benchmark cycle.

We use *Brier-based* proper scoring for the calibration term:

    payoff_calib = κ · (1 − 4·(P̂ − y)²)

Properties:
  - At P̂ = y (perfect), payoff_calib = +κ.
  - At P̂ = 0.5 (uniform) and y∈{0,1}, payoff_calib = 0  — no free κ for hedging.
  - At P̂ = 1, y = 0 (confidently wrong), payoff_calib = −3·κ.
  - Maximises in expectation iff agent reports its true belief E[y | task].

For the attempt term we use simple linear pricing:

    payoff_attempt = V · y − C · (1 − y)

Per-family (V, C, κ, δ) are loaded from `configs/market.yaml`. Defaults below
are calibrated so that an oracle-rate agent on a balanced workload earns ≈ 0
net payoff; over-performers earn positive expected value.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from prophet.engine.types import MarketOffer, Task

DEFAULT_FAMILY_CONFIG: dict[str, dict[str, float]] = {
    # baseline pricing (USD-cents-style "PROPHET-cents")
    # V_success: reward, C_failure: fine, kappa: calibration coefficient,
    # delta_pass: small research cost.
    "default": {"V_success": 100.0, "C_failure": 100.0, "kappa": 50.0, "delta_pass": 2.0},
    "math": {"V_success": 100.0, "C_failure": 100.0, "kappa": 50.0, "delta_pass": 2.0},
    "code": {"V_success": 120.0, "C_failure": 120.0, "kappa": 60.0, "delta_pass": 3.0},
    "knowledge": {"V_success": 80.0, "C_failure": 80.0, "kappa": 40.0, "delta_pass": 2.0},
    "reasoning": {"V_success": 110.0, "C_failure": 110.0, "kappa": 55.0, "delta_pass": 2.0},
    "writing": {"V_success": 90.0, "C_failure": 60.0, "kappa": 45.0, "delta_pass": 2.0},
    "browser": {"V_success": 130.0, "C_failure": 100.0, "kappa": 60.0, "delta_pass": 4.0},
    "tools": {"V_success": 130.0, "C_failure": 100.0, "kappa": 60.0, "delta_pass": 4.0},
    "multimodal": {"V_success": 110.0, "C_failure": 90.0, "kappa": 55.0, "delta_pass": 3.0},
    "data": {"V_success": 110.0, "C_failure": 90.0, "kappa": 55.0, "delta_pass": 3.0},
    "scientific": {"V_success": 120.0, "C_failure": 100.0, "kappa": 60.0, "delta_pass": 3.0},
    "multilingual": {"V_success": 90.0, "C_failure": 70.0, "kappa": 45.0, "delta_pass": 2.0},
    "safety": {"V_success": 100.0, "C_failure": 200.0, "kappa": 60.0, "delta_pass": 2.0},
}


def _deterministic_noise(market_seed: int, task_id: str, key: str, scale: float) -> float:
    """Tiny deterministic noise in (-scale, +scale) per (seed, task, key).

    Keeps the market non-cherry-pickable while staying replayable.
    """
    h = hashlib.sha256(f"{market_seed}|{task_id}|{key}".encode()).digest()
    # Use first 8 bytes as int in [0, 2^64-1], scale to (-1, 1)
    raw = int.from_bytes(h[:8], "big", signed=False)
    unit = (raw / (2**64 - 1)) * 2.0 - 1.0
    return scale * unit


@dataclass(slots=True)
class MarketMaker:
    """Deterministic market maker.

    Args:
      family_configs: per-family base prices. Falls back to "default".
      market_seed: integer seed rotated per benchmark cycle.
      noise_scale: relative magnitude of pricing noise (0 = exact).
    """

    family_configs: dict[str, dict[str, float]] = field(default_factory=lambda: dict(DEFAULT_FAMILY_CONFIG))
    market_seed: int = 0
    noise_scale: float = 0.02

    def offer(self, task: Task, override: dict[str, Any] | None = None) -> MarketOffer:
        cfg = dict(self.family_configs.get(task.family) or self.family_configs["default"])
        if override:
            cfg.update(override)
        # tiny per-task noise (gaming-resistant)
        v_n = _deterministic_noise(self.market_seed, task.task_id, "V", self.noise_scale * cfg["V_success"])
        c_n = _deterministic_noise(self.market_seed, task.task_id, "C", self.noise_scale * cfg["C_failure"])
        k_n = _deterministic_noise(self.market_seed, task.task_id, "K", self.noise_scale * cfg["kappa"])
        return MarketOffer(
            task_id=task.task_id,
            family=task.family,
            V_success=cfg["V_success"] + v_n,
            C_failure=cfg["C_failure"] + c_n,
            kappa_calib=cfg["kappa"] + k_n,
            delta_pass=cfg["delta_pass"],
            metadata={"market_seed": self.market_seed, "noise_scale": self.noise_scale},
        )
