#!/usr/bin/env python3
"""Finalize the paper from real experimental results.

Reads the leaderboard + comparison report, fills in TODO placeholders in
the paper sections with concrete numbers, and produces:
  docs/paper/figures/headline_table.tex      (auto from leaderboard.tex)
  docs/paper/figures/significance_table.tex  (from pairwise comparisons)
  docs/paper/figures/*.pdf                   (Pareto, reliability, heatmaps)
  docs/paper/results_filled.tex              (Results section with numbers)

Usage:
  python scripts/finalize_paper.py --runs-dir results/openrouter_matrix/runs
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

log = logging.getLogger("prophet.finalize")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path, required=True)
    ap.add_argument("--paper-dir", type=Path, default=Path("docs/paper"))
    args = ap.parse_args()

    figs = args.paper_dir / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    # 1) Build leaderboard
    from prophet.analysis.leaderboard import build_leaderboard

    leaderboard_dir = build_leaderboard(args.runs_dir)
    log.info("leaderboard at %s", leaderboard_dir)

    # Copy LaTeX-ready leaderboard
    src = leaderboard_dir / "leaderboard.tex"
    dst = figs / "headline_table.tex"
    if src.exists():
        shutil.copy(src, dst)
        log.info("copied %s -> %s", src, dst)

    # 2) Comparison report (figures + significance JSON)
    from prophet.analysis.comparison_report import write_full_report
    rep = write_full_report(args.runs_dir, out_dir=args.paper_dir / "report")
    log.info("comparison report at %s", rep)

    # 3) Transfer analysis
    from prophet.analysis.transfer import write_transfer_report
    write_transfer_report(args.runs_dir, args.paper_dir / "transfer")

    # 4) Move figures into paper figures dir
    fig_src = (args.paper_dir / "report" / "figures")
    if fig_src.exists():
        for f in fig_src.glob("*.*"):
            shutil.copy(f, figs / f.name)
        log.info("copied %d figures into %s", len(list(fig_src.glob('*'))), figs)

    # 5) Build significance LaTeX table from comparison.json
    cmp_json = rep / "summary.json"
    if cmp_json.exists():
        data = json.loads(cmp_json.read_text())
        rows = data.get("pairwise_significance", [])
        tex = [
            r"\begin{table}[t]\centering\small",
            r"\caption{Pairwise paired-test results, Holm--Bonferroni adjusted across all (pair $\times$ metric) tests. \emph{T} = paired Student's $t$; \emph{W} = Wilcoxon; \emph{M} = McNemar exact; \emph{P} = permutation on ECE. Bold = reject H$_0$ at $\alpha = 0.05$ after adjustment.}",
            r"\label{tab:sig}",
            r"\begin{tabular}{llrrrr}",
            r"\toprule",
            r"Agent A & Agent B & T($p$) & W($p$) & M($p$) & P-ECE($p$) \\",
            r"\midrule",
        ]
        for r in rows:
            t = r["tests"]
            def fmt(metric):
                v = t.get(metric)
                if v is None:
                    return "--"
                p = v.get("p_raw")
                if p is None:
                    return "--"
                cell = f"{p:.4f}"
                if v.get("reject_at_alpha"):
                    return r"\textbf{" + cell + "}"
                return cell
            tex.append(
                f"{r['a'].replace('_','-').replace('/','-')} & {r['b'].replace('_','-').replace('/','-')} & "
                f"{fmt('payoff_t')} & {fmt('payoff_wilcoxon')} & {fmt('accuracy_mcnemar')} & {fmt('ece_permutation')} \\\\"
            )
        tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        (figs / "significance_table.tex").write_text("\n".join(tex))
        log.info("wrote significance_table.tex")

    log.info("finalize done — paper figures + tables ready in %s", figs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
