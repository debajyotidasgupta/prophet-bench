"""ConcurrentRunner: wraps a synchronous Agent for concurrent batch dispatch.

Most agent adapters in PROPHET are synchronous (the vendor SDKs we use are
sync-first, and that keeps the engine simple). When the workload is many small
network-bound calls, running them sequentially is hugely wasteful — but we
still want determinism in the *output ordering* so downstream scoring stays
reproducible.

This runner gives us:
  * `anyio.to_thread.run_sync` to push each `agent.respond` onto a worker
    thread, so a sync agent gets parallelism without async refactoring;
  * a `CapacityLimiter` to cap in-flight requests (max_concurrency);
  * an optional `aiolimiter.AsyncLimiter` to additionally cap requests-per-second
    (useful for provider rate-limits);
  * deterministic ordering: results come back in the *same order* as inputs.

Errors are caught and converted to fallback PASS responses so a single bad
task doesn't crash the whole batch (mirroring the orchestrator's behaviour).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import anyio

from prophet.engine.types import Agent, AgentResponse, DecisionMode, MarketOffer, Task

log = logging.getLogger("prophet.agents.concurrent")


class ConcurrentRunner:
    """Run an Agent over a batch of (task, offer) pairs concurrently.

    Args:
      agent: any object implementing the Agent protocol (sync `.respond`).
      max_concurrency: maximum in-flight calls at any moment.
      per_sec_limit: optional global requests-per-second cap (None = unlimited).
    """

    def __init__(
        self,
        agent: Agent,
        max_concurrency: int = 8,
        per_sec_limit: float | None = None,
    ) -> None:
        self.agent = agent
        self.max_concurrency = max(1, int(max_concurrency))
        self.per_sec_limit = per_sec_limit

    # ---- public API ---------------------------------------------------

    def run_batch(
        self,
        tasks: Sequence[Task],
        market_maker: Any,
    ) -> list[tuple[Task, MarketOffer, AgentResponse]]:
        """Run agent.respond over a batch with bounded concurrency.

        `market_maker` only needs `.offer(task) -> MarketOffer`.
        Returns results in the SAME order as the input `tasks`.
        """
        tasks_list = list(tasks)
        offers = [market_maker.offer(t) for t in tasks_list]
        results: list[AgentResponse | None] = [None] * len(tasks_list)

        async def _runner() -> None:
            limiter = anyio.CapacityLimiter(self.max_concurrency)
            rate = None
            if self.per_sec_limit and self.per_sec_limit > 0:
                from aiolimiter import AsyncLimiter

                # `aiolimiter` rate is per second window of length 1.0
                rate = AsyncLimiter(max_rate=self.per_sec_limit, time_period=1.0)

            async def _one(i: int, task: Task, offer: MarketOffer) -> None:
                async with limiter:
                    if rate is not None:
                        async with rate:
                            results[i] = await anyio.to_thread.run_sync(
                                self.agent.respond, task, offer
                            )
                    else:
                        results[i] = await anyio.to_thread.run_sync(
                            self.agent.respond, task, offer
                        )

            async with anyio.create_task_group() as tg:
                for i, (task, offer) in enumerate(zip(tasks_list, offers, strict=False)):
                    tg.start_soon(_one, i, task, offer)

        anyio.run(_runner)

        # Replace any None (shouldn't happen unless run_sync raised) with PASS.
        finalised: list[tuple[Task, MarketOffer, AgentResponse]] = []
        for task, offer, resp in zip(tasks_list, offers, results, strict=False):
            if resp is None:  # pragma: no cover - defensive
                resp = AgentResponse(
                    task_id=task.task_id,
                    mode=DecisionMode.PASS,
                    confidence=0.5,
                    answer=None,
                    reasoning=None,
                    metadata={"error": "concurrent_runner_lost_result"},
                )
            finalised.append((task, offer, resp))
        return finalised

    def __repr__(self) -> str:
        return (
            f"ConcurrentRunner(agent={self.agent.name!r}, "
            f"max_concurrency={self.max_concurrency}, "
            f"per_sec_limit={self.per_sec_limit})"
        )


__all__ = ["ConcurrentRunner"]
