"""Orchestrator: drives Agent through tasks, applies market pricing, scores.

Runs are deterministic given (cycle_seed, market_seed). Cost guardrails are
enforced both per-run (PROPHET_MAX_RUN_COST_USD) and per-call.

The orchestrator is single-threaded by default; concurrency for many small
API calls is handled by the agent adapters themselves (anyio + aiolimiter).
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from prophet.engine.market import MarketMaker
from prophet.engine.scoring import compute_payoff
from prophet.engine.types import (
    Agent,
    AgentResponse,
    DecisionMode,
    Outcome,
    Task,
)

try:
    from prophet.utils.wandb_logger import WandbRun
except Exception:
    WandbRun = None  # noqa: N816 — optional dep

log = logging.getLogger("prophet.orchestrator")


@dataclass
class RunConfig:
    cycle_seed: int = 42
    market_seed: int = 0
    out_dir: Path = field(default_factory=lambda: Path("results/runs"))
    max_cost_usd: float = 20.0
    progress: bool = True
    save_traces: bool = True
    abstain_shadow_rate: float = 0.10  # fraction of PASS outcomes that get a shadow re-attempt
    notes: str = ""
    wandb: bool = False
    wandb_project: str = "prophet"
    wandb_tags: list[str] | None = None


@dataclass(slots=True)
class RunResult:
    run_id: str
    agent_name: str
    outcomes: list[Outcome]
    total_cost_usd: float
    wall_time_s: float
    config: RunConfig
    summary: dict[str, Any] = field(default_factory=dict)


class Orchestrator:
    """Runs an Agent through a Task list and produces an OutCome list."""

    def __init__(self, market: MarketMaker | None = None, log_level: int = logging.INFO) -> None:
        self.market = market or MarketMaker()
        logging.getLogger("prophet").setLevel(log_level)

    def run(
        self,
        agent: Agent,
        tasks: Iterable[Task],
        config: RunConfig | None = None,
    ) -> RunResult:
        cfg = config or RunConfig()
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        run_id = f"{int(time.time())}-{uuid.uuid4().hex[:8]}-{agent.name.replace('/', '_')}"
        run_dir = cfg.out_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        log.info("PROPHET run %s — agent=%s", run_id, agent.name)

        tasks_list = list(tasks)
        rng = np.random.default_rng(cfg.cycle_seed)
        outcomes: list[Outcome] = []
        total_cost = 0.0
        t0 = time.time()

        # Optional W&B run context
        wandb_ctx = None
        if cfg.wandb and WandbRun is not None:
            wandb_ctx = WandbRun(
                project=cfg.wandb_project,
                config={
                    "agent": agent.name,
                    "cycle_seed": cfg.cycle_seed,
                    "market_seed": cfg.market_seed,
                    "n_tasks": len(tasks_list),
                    "max_cost_usd": cfg.max_cost_usd,
                },
                tags=cfg.wandb_tags or ["prophet", agent.name.split(":")[0]],
                name=run_id,
            )
            try:
                wandb_ctx.__enter__()
            except Exception as e:
                log.warning("wandb enter failed: %s", e)
                wandb_ctx = None

        for i, task in enumerate(tasks_list):
            if total_cost > cfg.max_cost_usd:
                log.warning(
                    "Aborting: total_cost $%.2f exceeded max $%.2f",
                    total_cost,
                    cfg.max_cost_usd,
                )
                break
            offer = self.market.offer(task)
            try:
                resp = agent.respond(task, offer)
            except Exception as e:  # safety net; do not crash entire run
                log.exception("agent.respond raised on %s: %s", task.task_id, e)
                resp = AgentResponse(
                    task_id=task.task_id,
                    mode=DecisionMode.PASS,
                    confidence=0.5,
                    answer=None,
                    reasoning=None,
                    metadata={"error": repr(e)},
                )
            total_cost += float(resp.cost_usd or 0.0)

            # Grade
            success: bool | None
            judge_score: float | None = None
            if resp.mode == DecisionMode.PASS:
                success = None
            elif resp.answer is None:
                # The agent committed (TAKE/QUOTE) but produced no answer.
                # That is a failed attempt under the protocol — pay the fine.
                success = False
            else:
                try:
                    success = bool(task.verifier(resp.answer))
                except Exception as e:
                    log.exception("verifier raised on %s: %s", task.task_id, e)
                    success = False

            attempt, calib, total = compute_payoff(resp, success, offer)

            metadata: dict[str, Any] = {}
            if resp.mode == DecisionMode.PASS and rng.random() < cfg.abstain_shadow_rate:
                metadata["shadow_requested"] = True

            out = Outcome(
                task_id=task.task_id,
                family=task.family,
                difficulty=task.difficulty,
                response=resp,
                offer=offer,
                success=success,
                payoff_attempt=attempt,
                payoff_calib=calib,
                payoff_total=total,
                judge_score=judge_score,
                metadata=metadata,
            )
            outcomes.append(out)

            if cfg.progress and (i % 10 == 0 or i == len(tasks_list) - 1):
                log.info(
                    "[%4d/%4d] family=%s mode=%s P̂=%.2f y=%s payoff=%.2f cost=$%.4f total=$%.4f",
                    i + 1,
                    len(tasks_list),
                    task.family,
                    resp.mode.value,
                    resp.confidence,
                    str(success),
                    total,
                    resp.cost_usd,
                    total_cost,
                )

        elapsed = time.time() - t0

        # Persist
        run_record = {
            "run_id": run_id,
            "agent": agent.name,
            "cycle_seed": cfg.cycle_seed,
            "market_seed": cfg.market_seed,
            "wall_time_s": elapsed,
            "total_cost_usd": total_cost,
            "n_tasks": len(outcomes),
            "n_take": sum(1 for o in outcomes if o.response.mode == DecisionMode.TAKE),
            "n_quote": sum(1 for o in outcomes if o.response.mode == DecisionMode.QUOTE),
            "n_pass": sum(1 for o in outcomes if o.response.mode == DecisionMode.PASS),
            "net_payoff": sum(o.payoff_total for o in outcomes),
            "notes": cfg.notes,
        }
        (run_dir / "summary.json").write_text(json.dumps(run_record, indent=2, default=str))

        if cfg.save_traces:
            with (run_dir / "outcomes.jsonl").open("w") as f:
                for o in outcomes:
                    rec = _outcome_to_serial(o)
                    f.write(json.dumps(rec, default=str) + "\n")

        result = RunResult(
            run_id=run_id,
            agent_name=agent.name,
            outcomes=outcomes,
            total_cost_usd=total_cost,
            wall_time_s=elapsed,
            config=cfg,
            summary=run_record,
        )
        if wandb_ctx is not None:
            try:
                wandb_ctx.log_summary(run_record)
                wandb_ctx.log_outcomes(outcomes)
            except Exception as e:
                log.warning("wandb log failed: %s", e)
            finally:
                try:
                    wandb_ctx.__exit__(None, None, None)
                except Exception:
                    pass
        log.info("Run %s done — net payoff %.2f (cost $%.4f, %.1fs)", run_id, run_record["net_payoff"], total_cost, elapsed)
        return result


def _outcome_to_serial(o: Outcome) -> dict[str, Any]:
    d = asdict(o)
    # task verifier is not serializable; drop it via response.metadata cleaning
    return d
