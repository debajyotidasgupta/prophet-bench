"""PROPHET CLI — `prophet smoke`, `prophet run`, `prophet analyze`."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

app = typer.Typer(add_completion=False, no_args_is_help=True, rich_markup_mode="rich")
console = Console()


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


def _load_env() -> None:
    cwd_env = Path.cwd() / ".env"
    if cwd_env.exists():
        load_dotenv(cwd_env)
    else:
        load_dotenv()


@app.command()
def smoke(
    n: int = typer.Option(10, help="Number of math tasks to run for smoke test."),
    seed: int = typer.Option(42),
    log_level: str = typer.Option("INFO"),
) -> None:
    """Run a CPU-only smoke test on the math family using baselines.

    Exercises the entire pipeline without requiring any API key.
    """
    _setup_logging(log_level)
    _load_env()
    from prophet.agents import AlwaysPassAgent, AlwaysTakeAgent, OracleAgent, RandomAgent
    from prophet.engine.market import MarketMaker
    from prophet.engine.orchestrator import Orchestrator, RunConfig
    from prophet.families import get_family
    from prophet.engine.scoring import (
        adaptive_ece,
        brier_score,
        ece,
        log_score,
        model_overreach_point,
        total_payoff,
    )
    fam = get_family("math")
    tasks = fam.generate(n=n, seed=seed)
    market = MarketMaker(market_seed=seed)
    orch = Orchestrator(market=market)
    agents = [
        RandomAgent(seed=seed),
        AlwaysTakeAgent(),
        AlwaysPassAgent(),
        OracleAgent(seed=seed, correctness_rate=1.0),
        OracleAgent(name="baseline:oracle-noisy", seed=seed, correctness_rate=0.7),
    ]
    table = Table(title=f"PROPHET smoke ({n} tasks)")
    table.add_column("agent")
    table.add_column("payoff", justify="right")
    table.add_column("ECE", justify="right")
    table.add_column("Brier", justify="right")
    table.add_column("logloss", justify="right")
    table.add_column("MOP-diff", justify="right")
    for ag in agents:
        result = orch.run(ag, tasks, RunConfig(cycle_seed=seed, max_cost_usd=10.0, progress=False))
        confs, ys, diffs = [], [], []
        for o in result.outcomes:
            if o.success is None:
                continue
            confs.append(float(o.response.confidence))
            ys.append(int(bool(o.success)))
            diffs.append(float(o.difficulty))
        if not confs:
            row = (ag.name, f"{total_payoff(result.outcomes):.1f}", "—", "—", "—", "—")
        else:
            mop = model_overreach_point(confs, ys, diffs).mop_difficulty
            row = (
                ag.name,
                f"{total_payoff(result.outcomes):.1f}",
                f"{ece(confs, ys):.3f}",
                f"{brier_score(confs, ys):.3f}",
                f"{log_score(confs, ys):.3f}",
                "—" if mop is None else f"{mop:.2f}",
            )
        table.add_row(*row)
    console.print(table)


@app.command()
def list_families() -> None:
    """List available task families."""
    from prophet.families import list_families as _ls
    for n in _ls():
        console.print(f"• {n}")


@app.command()
def run(
    agent: str = typer.Option(..., help="Agent URI, e.g. openai:gpt-5-mini, vllm:Qwen/Qwen3-7B-Instruct"),
    families: str = typer.Option("math", help="Comma-separated family names or 'all'."),
    n: int = typer.Option(50, help="Tasks per family."),
    seed: int = typer.Option(42),
    max_cost: float = typer.Option(20.0, help="Hard cost cap in USD."),
    out_dir: Path = typer.Option(Path("results/runs"), help="Output directory."),
    log_level: str = typer.Option("INFO"),
) -> None:
    """Run an agent through one or more families."""
    _setup_logging(log_level)
    _load_env()
    from prophet.agents import build_agent
    from prophet.engine.market import MarketMaker
    from prophet.engine.orchestrator import Orchestrator, RunConfig
    from prophet.families import get_family, list_families as _ls

    families_list = _ls() if families == "all" else [s.strip() for s in families.split(",") if s.strip()]
    tasks = []
    for fname in families_list:
        fam = get_family(fname)
        tasks.extend(fam.generate(n=n, seed=seed))
    market = MarketMaker(market_seed=seed)
    orch = Orchestrator(market=market)
    ag = build_agent(agent)
    cfg = RunConfig(cycle_seed=seed, market_seed=seed, out_dir=out_dir, max_cost_usd=max_cost)
    result = orch.run(ag, tasks, cfg)
    console.print(f"[green]Run finished[/green]: {result.run_id}  net=${result.summary['net_payoff']:.1f}  cost=${result.total_cost_usd:.4f}")


@app.command()
def analyze(run_dir: Path) -> None:
    """Run the analysis pipeline on a saved run directory."""
    from prophet.analysis.report import build_report
    build_report(run_dir)
    console.print(f"[green]Report written to[/green] {run_dir}/report.html")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
