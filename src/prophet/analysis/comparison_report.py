"""Multi-run comparison report.

Given a directory containing multiple per-agent runs, produce:
  * a headline CSV / Markdown table of agent metrics with bootstrap CIs;
  * a pairwise significance matrix (paired tests + Holm-Bonferroni);
  * Pareto frontier scatter plots;
  * per-family heatmaps (accuracy, ECE).

Used as `prophet analyze-compare results/full_eval/runs`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np

from prophet.analysis.compare import AgentRun, headline_table, pairwise_significance
from prophet.analysis.plots import (
    calibration_curve_multi,
    pareto_scatter,
    per_family_heatmap,
)

log = logging.getLogger("prophet.analysis.compare-report")


def _discover_runs(root: Path) -> list[AgentRun]:
    runs: list[AgentRun] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if not (child / "outcomes.jsonl").exists():
            continue
        # Run-id format: <epoch>-<hash>-<agent>
        name = child.name.split("-", 2)[-1] if "-" in child.name else child.name
        runs.append(AgentRun.from_dir(child, name=name))
    return runs


def _per_family_metric(run: AgentRun, metric: str) -> dict[str, float]:
    from collections import defaultdict

    from prophet.engine.scoring import brier_score, ece, log_score

    by_family: dict[str, list[dict]] = defaultdict(list)
    for o in run.outcomes:
        if o.get("success") is None:
            continue
        by_family[o["family"]].append(o)
    out = {}
    for fam, rows in by_family.items():
        confs = [float(r["response"]["confidence"]) for r in rows]
        ys = [1 if r["success"] else 0 for r in rows]
        if metric == "accuracy":
            out[fam] = float(np.mean(ys))
        elif metric == "ece":
            out[fam] = ece(confs, ys)
        elif metric == "brier":
            out[fam] = brier_score(confs, ys)
        elif metric == "logloss":
            out[fam] = log_score(confs, ys)
        elif metric == "payoff":
            out[fam] = float(sum(o["payoff_total"] for o in rows))
    return out


def write_full_report(runs_dir: Path, out_dir: Path | None = None, seed: int = 0) -> Path:
    runs_dir = Path(runs_dir)
    out_dir = (out_dir or runs_dir.parent / "report").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = _discover_runs(runs_dir)
    log.info("discovered %d runs under %s", len(runs), runs_dir)
    table = headline_table(runs, ci=0.95, seed=seed)
    sig = pairwise_significance(runs, seed=seed)
    summary = {
        "headline": [asdict(s) for s in table],
        "pairwise_significance": [
            {"a": a, "b": b, "tests": tests}
            for (a, b), tests in sig.items()
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    # Markdown table
    md = ["# PROPHET — comparison report", "", "## Headline (95% CI)", ""]
    md.append("| agent | n | acc | ECE | Brier | log-loss | net payoff |")
    md.append("|---|---|---|---|---|---|---|")
    for s in table:
        md.append(
            f"| {s.name} | {s.n} | {s.accuracy:.3f} "
            f"[{s.accuracy_ci[0]:.3f},{s.accuracy_ci[1]:.3f}] "
            f"| {s.ece:.3f} [{s.ece_ci[0]:.3f},{s.ece_ci[1]:.3f}] "
            f"| {s.brier:.3f} [{s.brier_ci[0]:.3f},{s.brier_ci[1]:.3f}] "
            f"| {s.logloss:.3f} [{s.logloss_ci[0]:.3f},{s.logloss_ci[1]:.3f}] "
            f"| {s.net_payoff:.1f} [{s.net_payoff_ci[0]:.1f},{s.net_payoff_ci[1]:.1f}] |"
        )
    md.append("")
    md.append("## Pairwise significance (Holm-Bonferroni adjusted)")
    md.append("")
    md.append("| A | B | metric | p_raw | reject H0? |")
    md.append("|---|---|---|---|---|")
    for (a, b), tests in sig.items():
        for m, info in tests.items():
            md.append(f"| {a} | {b} | {m} | {info['p_raw']:.4f} | {'**yes**' if info['reject_at_alpha'] else 'no'} |")
    (out_dir / "report.md").write_text("\n".join(md))

    # Figures
    records = [
        {
            "name": s.name,
            "accuracy": s.accuracy,
            "ece": s.ece,
            "brier": s.brier,
            "net_payoff": s.net_payoff,
            "cost_usd": 0.0,
        }
        for s in table
    ]
    if records:
        pareto_scatter(records, out_dir / "figures" / "pareto_ece_payoff.png", x="ece", y="net_payoff", title="Pareto: ECE × net payoff")
        pareto_scatter(records, out_dir / "figures" / "pareto_brier_payoff.png", x="brier", y="net_payoff", title="Pareto: Brier × net payoff")

        # Reliability multi-line
        series = {}
        for r in runs:
            graded = [o for o in r.outcomes if o.get("success") is not None]
            if len(graded) < 10:
                continue
            confs = [float(o["response"]["confidence"]) for o in graded]
            ys = [1 if o["success"] else 0 for o in graded]
            series[r.name] = (confs, ys)
        if series:
            calibration_curve_multi(series, out_dir / "figures" / "reliability_multi.png")

        # Per-family heatmaps
        families = sorted({o["family"] for r in runs for o in r.outcomes})
        agent_names = [r.name for r in runs]
        for metric_name, vmin, vmax, cmap in [
            ("accuracy", 0.0, 1.0, "RdYlGn"),
            ("ece", 0.0, 0.5, "RdYlGn_r"),
            ("brier", 0.0, 0.5, "RdYlGn_r"),
            ("payoff", None, None, "RdYlGn"),
        ]:
            mat = np.zeros((len(families), len(agent_names)), dtype=float)
            for j, r in enumerate(runs):
                values = _per_family_metric(r, metric_name)
                for i, fam in enumerate(families):
                    mat[i, j] = values.get(fam, np.nan)
            per_family_heatmap(
                mat,
                families,
                agent_names,
                out_dir / "figures" / f"heatmap_{metric_name}.png",
                title=f"Per-family {metric_name}",
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                fmt="{:.2f}" if metric_name != "payoff" else "{:.0f}",
            )

    log.info("comparison report written to %s", out_dir)
    return out_dir
