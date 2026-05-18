#!/usr/bin/env python3
"""Generate paper-ready figures + tables from a matrix run.

Outputs:
  docs/paper/figures/headline_table.tex
  docs/paper/figures/pareto_ece_payoff.pdf
  docs/paper/figures/per_family_ece_heatmap.pdf
  docs/paper/figures/per_family_payoff_heatmap.pdf
  docs/paper/figures/reliability_multi.pdf
  docs/paper/figures/mop_by_family.pdf

Usage:
  python scripts/make_paper_figures.py --runs-dir results/openrouter_matrix/runs
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

log = logging.getLogger("prophet.paper_figs")


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path, required=True)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs/paper/figures"),
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from prophet.analysis.compare import AgentRun, headline_table
    from prophet.analysis.comparison_report import write_full_report
    from prophet.analysis.plots import (
        calibration_curve_multi,
        pareto_scatter,
        per_family_heatmap,
        reliability_diagram,
    )
    from prophet.analysis.transfer import write_transfer_report
    from prophet.engine.scoring import (
        brier_score,
        ece,
        log_score,
        model_overreach_point,
    )

    # 1) Full report
    write_full_report(args.runs_dir, args.out_dir / "report")
    log.info("wrote comparison report to %s", args.out_dir / "report")

    # 2) Transfer report
    write_transfer_report(args.runs_dir, args.out_dir / "transfer")
    log.info("wrote transfer report")

    # 3) LaTeX-style headline table
    runs = []
    for child in sorted(args.runs_dir.iterdir()):
        if (child / "outcomes.jsonl").exists():
            name = child.name.split("-", 2)[-1] if "-" in child.name else child.name
            runs.append(AgentRun.from_dir(child, name=name))
    table = headline_table(runs, ci=0.95)
    table_sorted = sorted(table, key=lambda s: -s.net_payoff)
    tex_lines = [
        r"\begin{table}[t]",
        r"\centering\small",
        r"\caption{PROPHET headline results --- 12 task families $\times$ "
        f"N tasks per family. CIs are bootstrap 95\\%. Higher net payoff is better; lower ECE / Brier is better."
        r"}\label{tab:headline}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Agent & N & Accuracy & ECE & Brier & Net payoff \\",
        r"\midrule",
    ]
    for s in table_sorted:
        tex_lines.append(
            f"{s.name.replace('_','-')} & {s.n} & "
            f"{s.accuracy:.3f} & {s.ece:.3f} & {s.brier:.3f} & {s.net_payoff:.0f} \\\\"
        )
    tex_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (args.out_dir / "headline_table.tex").write_text("\n".join(tex_lines))
    log.info("wrote headline_table.tex")

    # 4) MOP figure
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # MOP per family per agent (one panel per family, agent on x-axis)
        families = sorted({o["family"] for r in runs for o in r.outcomes})
        rows = []
        for r in runs:
            graded = [o for o in r.outcomes if o.get("success") is not None]
            for fam in families:
                subset = [o for o in graded if o["family"] == fam]
                if len(subset) < 10:
                    continue
                confs = [float(o["response"]["confidence"]) for o in subset]
                ys = [1 if o["success"] else 0 for o in subset]
                diffs = [float(o["difficulty"]) for o in subset]
                m = model_overreach_point(confs, ys, diffs, n_bins=6)
                rows.append(
                    {
                        "agent": r.name,
                        "family": fam,
                        "max_overreach": m.max_overreach,
                        "mop": m.mop_difficulty,
                    }
                )
        if rows:
            import pandas as pd

            df = pd.DataFrame(rows)
            # Duplicate (agent, family) entries appear when an agent was re-run
            # under different caps; aggregate by max so the worst-case overreach
            # surfaces in the figure.
            pivot = df.pivot_table(
                index="agent",
                columns="family",
                values="max_overreach",
                aggfunc="max",
            ).fillna(0.0)
            fig, ax = plt.subplots(figsize=(1.0 + 0.6 * len(families), 0.5 + 0.4 * len(pivot)))
            im = ax.imshow(pivot.values, cmap="OrRd", vmin=0.0, vmax=0.5)
            ax.set_xticks(range(len(families)))
            ax.set_xticklabels(families, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(len(pivot)))
            ax.set_yticklabels(pivot.index.tolist(), fontsize=8)
            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    ax.text(j, i, f"{pivot.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
            fig.colorbar(im, ax=ax, shrink=0.8)
            ax.set_title("Max-overreach (max(P̂_bar - acc_bar)) per family / agent")
            fig.tight_layout()
            fig.savefig(args.out_dir / "mop_by_family.png", dpi=150)
            fig.savefig(args.out_dir / "mop_by_family.pdf")
            plt.close(fig)
            log.info("wrote mop_by_family.{png,pdf}")
    except Exception as e:  # noqa: BLE001
        log.warning("MOP figure failed: %s", e)

    log.info("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
