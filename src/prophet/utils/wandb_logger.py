"""Thin W&B logger wrapper.

Use:
  with WandbRun(project="prophet", config={"agent": "openai:gpt-5-mini"}) as run:
      run.log_summary({"net_payoff": 123.4, "ece": 0.12})
      run.log_outcomes(outcomes)
"""

from __future__ import annotations

import logging
import os
from contextlib import AbstractContextManager
from typing import Any

log = logging.getLogger("prophet.wandb")


class WandbRun(AbstractContextManager):
    def __init__(
        self,
        project: str | None = None,
        entity: str | None = None,
        config: dict[str, Any] | None = None,
        name: str | None = None,
        tags: list[str] | None = None,
        offline: bool = False,
    ) -> None:
        self.project = project or os.environ.get("WANDB_PROJECT", "prophet")
        self.entity = entity or os.environ.get("WANDB_ENTITY")
        self.config = config or {}
        self.name = name
        self.tags = tags or []
        self.offline = offline
        self._run = None

    def __enter__(self):
        try:
            import wandb
        except Exception as e:  # noqa: BLE001
            log.warning("wandb unavailable, logging disabled: %s", e)
            return self
        try:
            mode = "offline" if self.offline else "online"
            self._run = wandb.init(
                project=self.project,
                entity=self.entity,
                config=self.config,
                name=self.name,
                tags=self.tags,
                mode=mode,
                reinit=True,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("wandb init failed (continuing without): %s", e)
            self._run = None
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._run is not None:
            try:
                self._run.finish()
            except Exception:
                pass
        return False

    def log_summary(self, summary: dict[str, Any]) -> None:
        if self._run is None:
            return
        try:
            for k, v in summary.items():
                self._run.summary[k] = v
        except Exception as e:
            log.warning("wandb log_summary failed: %s", e)

    def log_step(self, step: int, payload: dict[str, Any]) -> None:
        if self._run is None:
            return
        try:
            self._run.log(payload, step=step)
        except Exception as e:
            log.warning("wandb log_step failed: %s", e)

    def log_outcomes(self, outcomes: list) -> None:
        """Log a Table of per-task outcomes."""
        if self._run is None:
            return
        try:
            import wandb

            cols = [
                "task_id",
                "family",
                "difficulty",
                "mode",
                "confidence",
                "success",
                "payoff_total",
                "cost_usd",
            ]
            rows = []
            for o in outcomes:
                rows.append(
                    [
                        getattr(o, "task_id", None),
                        getattr(o, "family", None),
                        getattr(o, "difficulty", None),
                        o.response.mode.value if hasattr(o.response.mode, "value") else o.response.mode,
                        o.response.confidence,
                        o.success,
                        o.payoff_total,
                        o.response.cost_usd,
                    ]
                )
            table = wandb.Table(columns=cols, data=rows)
            self._run.log({"outcomes": table})
        except Exception as e:
            log.warning("wandb log_outcomes failed: %s", e)
