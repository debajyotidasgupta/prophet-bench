"""Public leaderboard generation.

After each matrix run, this produces a single canonical leaderboard:
  * Sorted by net payoff (descending).
  * With ECE, Brier, accuracy, abstention precision, cost.
  * With bootstrap 95% CIs.
  * With Pareto-front membership flags.
  * As both JSON (machine) + Markdown (humans) + LaTeX (paper).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from prophet.analysis.compare import AgentRun, headline_table
from prophet.analysis.pareto import PointRecord, pareto_front

log = logging.getLogger("prophet.analysis.leaderboard")


def _summary_cost(run_dir: Path) -> float:
    p = run_dir / "summary.json"
    if not p.exists():
        return 0.0
    try:
        return float(json.loads(p.read_text()).get("total_cost_usd", 0.0))
    except Exception:
        return 0.0


def _discover_runs(root: Path) -> tuple[list[AgentRun], dict[str, float]]:
    """Discover runs and pool by agent name.

    When the same agent is run more than once (multi-seed rotation, retries,
    batch continuation) we POOL outcomes across all matching directories and
    SUM costs across them. This is the only way to give a single canonical
    leaderboard row per agent. Task IDs are seed-prefixed at generation time
    so pooled task_id dicts do not collide across runs of the same agent.
    """
    by_name: dict[str, list[dict]] = {}
    canonical_dir: dict[str, Path] = {}
    cost_by_name: dict[str, float] = {}
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if not (child / "outcomes.jsonl").exists():
            continue
        name = child.name.split("-", 2)[-1] if "-" in child.name else child.name
        outs = _load_outcomes(child)
        by_name.setdefault(name, []).extend(outs)
        canonical_dir.setdefault(name, child)
        cost_by_name[name] = cost_by_name.get(name, 0.0) + _summary_cost(child)
    runs: list[AgentRun] = []
    for name in by_name:
        outs = by_name[name]
        runs.append(
            AgentRun(
                name=name,
                run_dir=canonical_dir[name],
                outcomes=outs,
                by_task={o["task_id"]: o for o in outs},
            )
        )
    return runs, cost_by_name


def _load_outcomes(run_dir: Path) -> list[dict]:
    p = run_dir / "outcomes.jsonl"
    if not p.exists():
        return []
    out: list[dict] = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def build_leaderboard(runs_dir: Path, out_dir: Path | None = None, ci: float = 0.95) -> Path:
    runs, cost_by_name = _discover_runs(runs_dir)
    if not runs:
        raise RuntimeError(f"No completed runs found under {runs_dir}")
    out_dir = (out_dir or runs_dir.parent / "leaderboard").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    table = headline_table(runs, ci=ci, seed=0)
    # `headline_table` skips agents with zero committed outcomes (no acc/ECE
    # defined). The marketplace still defines net_payoff for these (PASS pays
    # -delta_pass). Inject a synthetic row for any such agent so the canonical
    # abstention floor (baseline:always-pass) is visible in the leaderboard.
    seen_names = {s.name for s in table}
    from prophet.analysis.stats import AgentSummary
    for r in runs:
        if r.name in seen_names:
            continue
        pays = [float(o["payoff_total"]) for o in r.outcomes]
        n_pay = len(pays)
        if n_pay == 0:
            continue
        net = sum(pays)
        # Use NaN for committed-only metrics; the table renderer prints them
        # as 'nan' which is the right honest signal.
        nan = float("nan")
        table.append(
            AgentSummary(
                name=r.name,
                n=0,
                accuracy=nan, accuracy_ci=(nan, nan),
                ece=nan, ece_ci=(nan, nan),
                brier=nan, brier_ci=(nan, nan),
                logloss=nan, logloss_ci=(nan, nan),
                net_payoff=net,
                net_payoff_ci=(net, net),
            )
        )

    records = [
        {
            "name": s.name,
            "n": s.n,
            "accuracy": round(s.accuracy, 4),
            "accuracy_ci": [round(s.accuracy_ci[0], 4), round(s.accuracy_ci[1], 4)],
            "ece": round(s.ece, 4),
            "ece_ci": [round(s.ece_ci[0], 4), round(s.ece_ci[1], 4)],
            "brier": round(s.brier, 4),
            "brier_ci": [round(s.brier_ci[0], 4), round(s.brier_ci[1], 4)],
            "logloss": round(s.logloss, 4),
            "logloss_ci": [round(s.logloss_ci[0], 4), round(s.logloss_ci[1], 4)],
            "net_payoff": round(s.net_payoff, 2),
            "net_payoff_ci": [round(s.net_payoff_ci[0], 2), round(s.net_payoff_ci[1], 2)],
            "cost_usd": round(cost_by_name.get(s.name, 0.0), 4),
        }
        for s in table
    ]
    records.sort(key=lambda r: -r["net_payoff"])
    # Pareto on (ECE, net_payoff) — computed *excluding* baselines so the
    # leaderboard "Pareto" star is among real systems. The oracle is on the
    # frontier by construction (perfect calibration + max payoff); listing
    # it crowds out the interesting comparison.
    real_records = [r for r in records if not r["name"].startswith("baseline:")]
    pts = [
        PointRecord(
            name=r["name"],
            payoff=r["net_payoff"],
            ece=r["ece"],
            brier=r["brier"],
            accuracy=r["accuracy"],
            cost_usd=r["cost_usd"],
        )
        for r in real_records
    ]
    front = {p.name for p in pareto_front(pts, axes=("ece", "payoff"), maximise={"payoff", "accuracy"})}
    for r in records:
        r["pareto_optimal"] = r["name"] in front
        # Mark baselines explicitly so the table is honest
        if r["name"].startswith("baseline:"):
            r["pareto_optimal_among_real"] = False
        else:
            r["pareto_optimal_among_real"] = r["name"] in front

    (out_dir / "leaderboard.json").write_text(json.dumps(records, indent=2))

    # Markdown
    md = [
        "# PROPHET — leaderboard",
        "",
        "| Rank | Agent | N | Accuracy | ECE | Brier | Log-loss | Net Payoff | Cost USD | Pareto |",
        "|------|---|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(records, 1):
        md.append(
            f"| {i} | {r['name']} | {r['n']} "
            f"| {r['accuracy']:.3f} ({r['accuracy_ci'][0]:.3f}, {r['accuracy_ci'][1]:.3f}) "
            f"| {r['ece']:.3f} ({r['ece_ci'][0]:.3f}, {r['ece_ci'][1]:.3f}) "
            f"| {r['brier']:.3f} "
            f"| {r['logloss']:.3f} "
            f"| {r['net_payoff']:.1f} ({r['net_payoff_ci'][0]:.1f}, {r['net_payoff_ci'][1]:.1f}) "
            f"| ${r['cost_usd']:.4f} "
            f"| {'★' if r['pareto_optimal'] else ''} |"
        )
    (out_dir / "leaderboard.md").write_text("\n".join(md))

    # LaTeX (paper-ready)
    tex = [
        r"\begin{table}[t]",
        r"\centering\small",
        r"\caption{PROPHET headline leaderboard (95\% bootstrap CIs). $\star$ = Pareto-optimal on $(\ECE, \mathcal{P})$.}\label{tab:headline}",
        r"\begin{tabular}{rlrrrrrr}",
        r"\toprule",
        r"\# & Agent & Acc. & ECE & Brier & Log-loss & Net payoff & Cost \$ \\",
        r"\midrule",
    ]
    for i, r in enumerate(records, 1):
        star = r"$\star$" if r["pareto_optimal"] else ""
        agent_tex = r["name"].replace("_", "-").replace("/", "-")
        tex.append(
            f"{i} & {agent_tex}{star} & {r['accuracy']:.3f} & {r['ece']:.3f} "
            f"& {r['brier']:.3f} & {r['logloss']:.3f} & {r['net_payoff']:.0f} & {r['cost_usd']:.3f} \\\\"
        )
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (out_dir / "leaderboard.tex").write_text("\n".join(tex))

    log.info("Leaderboard written → %s", out_dir)
    return out_dir


def add_cli(app):  # for cli.py to pull in
    def leaderboard(runs_dir: Path):
        out = build_leaderboard(runs_dir)
        from rich.console import Console
        Console().print(f"[green]Leaderboard written to[/green] {out}")
    return leaderboard
