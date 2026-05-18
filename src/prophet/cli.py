"""PROPHET CLI — `prophet smoke`, `prophet run`, `prophet analyze`."""

from __future__ import annotations

import logging
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
    from prophet.engine.scoring import (
        brier_score,
        ece,
        log_score,
        model_overreach_point,
        total_payoff,
    )
    from prophet.families import get_family
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
    concurrency: int = typer.Option(1, help="Concurrent agent calls (>=1). Routes through ConcurrentRunner when >1."),
    per_sec_limit: float = typer.Option(0.0, help="Optional requests-per-second cap (0 = unlimited)."),
    max_tokens: int = typer.Option(2048, help="Max tokens per agent response (reasoning + visible)."),
    temperature: float = typer.Option(0.0, help="Sampling temperature (0 = greedy)."),
    wandb: bool = typer.Option(False, help="Enable Weights & Biases logging."),
    wandb_project: str = typer.Option("prophet", help="W&B project name."),
    log_level: str = typer.Option("INFO"),
    difficulty_range: str = typer.Option(
        "0.0,1.0",
        help="Restrict generated tasks to difficulty band 'lo,hi' (inclusive). "
        "Use e.g. '0.97,1.0' to run only T_extreme tier.",
    ),
) -> None:
    """Run an agent through one or more families."""
    _setup_logging(log_level)
    _load_env()
    from prophet.agents import ConcurrentRunner, build_agent
    from prophet.engine.market import MarketMaker
    from prophet.engine.orchestrator import Orchestrator, RunConfig
    from prophet.families import get_family
    from prophet.families import list_families as _ls

    families_list = _ls() if families == "all" else [s.strip() for s in families.split(",") if s.strip()]
    try:
        lo_str, hi_str = difficulty_range.split(",")
        diff_band = (float(lo_str), float(hi_str))
    except Exception as e:
        raise typer.BadParameter(f"--difficulty-range must be 'lo,hi' (got {difficulty_range!r}): {e}")
    tasks = []
    for fname in families_list:
        fam = get_family(fname)
        tasks.extend(fam.generate(n=n, seed=seed, difficulty_range=diff_band))
    market = MarketMaker(market_seed=seed)
    orch = Orchestrator(market=market)
    ag = build_agent(agent, max_tokens=max_tokens, temperature=temperature)
    if concurrency > 1:
        runner = ConcurrentRunner(
            ag,
            max_concurrency=concurrency,
            per_sec_limit=per_sec_limit or None,
        )
        ag = _PrecomputedRespondAgent(ag, runner, tasks, market)
    cfg = RunConfig(
        cycle_seed=seed,
        market_seed=seed,
        out_dir=out_dir,
        max_cost_usd=max_cost,
        wandb=wandb,
        wandb_project=wandb_project,
    )
    result = orch.run(ag, tasks, cfg)
    console.print(f"[green]Run finished[/green]: {result.run_id}  net=${result.summary['net_payoff']:.1f}  cost=${result.total_cost_usd:.4f}")


class _PrecomputedRespondAgent:
    """Adapter that pre-runs the batch concurrently, then serves cached AgentResponses one-by-one.

    The orchestrator iterates tasks sequentially and calls `agent.respond(task, offer)` for each.
    To preserve its scoring/persistence path while still using ConcurrentRunner for the heavy
    network work, we pre-compute every response in parallel and replay them in order.
    """

    def __init__(self, inner, runner, tasks, market) -> None:
        self.name = inner.name
        self.family_support = getattr(inner, "family_support", None)
        self._inner = inner
        triples = runner.run_batch(tasks, market)
        self._by_task_id = {t.task_id: resp for (t, _o, resp) in triples}

    def respond(self, task, offer):
        resp = self._by_task_id.get(task.task_id)
        if resp is None:
            return self._inner.respond(task, offer)
        return resp


@app.command()
def analyze(run_dir: Path) -> None:
    """Run the analysis pipeline on a saved run directory."""
    from prophet.analysis.report import build_report
    build_report(run_dir)
    console.print(f"[green]Report written to[/green] {run_dir}/report.html")


@app.command()
def status(
    runs_dir: Path = typer.Argument(Path("results"), help="Top-level results directory."),
) -> None:
    """Quick summary of all runs found under `runs_dir`."""
    _load_env()
    if not runs_dir.exists():
        console.print(f"[yellow]No runs dir at[/yellow] {runs_dir}")
        return
    total_cost = 0.0
    n_runs = 0
    table = Table(title=f"PROPHET runs under {runs_dir}")
    table.add_column("run")
    table.add_column("agent")
    table.add_column("n", justify="right")
    table.add_column("net payoff", justify="right")
    table.add_column("cost USD", justify="right")
    table.add_column("wall s", justify="right")
    import json as _json
    for child in sorted(runs_dir.rglob("summary.json")):
        try:
            s = _json.loads(child.read_text())
        except Exception:
            continue
        table.add_row(
            child.parent.name[:60],
            s.get("agent", "—"),
            str(s.get("n_tasks", 0)),
            f"{s.get('net_payoff', 0):.1f}",
            f"${s.get('total_cost_usd', 0):.4f}",
            f"{s.get('wall_time_s', 0):.1f}",
        )
        total_cost += float(s.get("total_cost_usd", 0))
        n_runs += 1
    console.print(table)
    console.print(f"[bold]Total runs:[/bold] {n_runs}   [bold]Total cost:[/bold] ${total_cost:.4f}")


@app.command()
def leaderboard(
    runs_dir: Path = typer.Argument(..., help="Directory of completed runs."),
    out_dir: Path = typer.Option(None, help="Where to write the leaderboard (defaults to <runs_dir>/../leaderboard)."),
) -> None:
    """Produce a canonical Markdown + JSON + LaTeX leaderboard."""
    from prophet.analysis.leaderboard import build_leaderboard

    out = build_leaderboard(runs_dir, out_dir=out_dir)
    console.print(f"[green]Leaderboard written to[/green] {out}")


@app.command("analyze-compare")
def analyze_compare(
    runs_dir: Path = typer.Argument(..., help="Directory of completed runs (each subdir contains outcomes.jsonl)."),
    out_dir: Path = typer.Option(None, help="Where to write the comparison report (defaults to <runs_dir>/../report)."),
    seed: int = typer.Option(0, help="Seed for permutation/bootstrap."),
) -> None:
    """Cross-run comparison: paired tests, Pareto frontier, per-family heatmaps."""
    from prophet.analysis.comparison_report import write_full_report

    out = write_full_report(runs_dir, out_dir=out_dir, seed=seed)
    console.print(f"[green]Comparison report written to[/green] {out}")


@app.command("calibrate")
def calibrate(
    families: str = typer.Option("all", help="Comma-separated families or 'all'."),
    n: int = typer.Option(200, help="Tasks per family."),
    seed: int = typer.Option(42),
    agents: str = typer.Option(
        "baseline:oracle-noisy",
        help="Comma-separated reference agent URIs.",
    ),
    out: Path = typer.Option(Path("data/reference"), help="Output directory."),
    max_cost_per_agent: float = typer.Option(3.0),
) -> None:
    """Calibrate per-task empirical difficulty using a reference panel."""
    import subprocess

    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parents[2] / "scripts" / "calibrate_difficulty.py"),
        "--families",
        families,
        "--n",
        str(n),
        "--seed",
        str(seed),
        "--agents",
        agents,
        "--out",
        str(out),
        "--max-cost-per-agent",
        str(max_cost_per_agent),
    ]
    subprocess.run(cmd, check=False)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
