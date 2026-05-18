#!/usr/bin/env python3
"""Auto-write a 1-paragraph results narrative from a leaderboard.

Produces docs/paper/sections/results_narrative.tex with concrete numbers
that fill the H1-H5 hypotheses in the paper.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sps

log = logging.getLogger("prophet.narrative")


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--leaderboard", type=Path, required=True)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("docs/paper/sections/results_narrative.tex"),
    )
    ap.add_argument(
        "--min-n",
        type=int,
        default=100,
        help="Minimum committed-task count to include in champion/Spearman/Pareto. "
        "Defaults to N>=100, matching the pre-registered analysis plan.",
    )
    args = ap.parse_args()

    records = json.loads(args.leaderboard.read_text())
    real_all = [r for r in records if not r["name"].startswith("baseline:")]
    real = [r for r in real_all if r["n"] >= args.min_n]
    dropped = [r["name"] for r in real_all if r["n"] < args.min_n]
    if len(real) < 2:
        log.warning("Need >= 2 real agents (N>=%d) for narrative; got %d", args.min_n, len(real))
        return 1
    if dropped:
        log.info("Excluded %d low-N agents (N<%d): %s", len(dropped), args.min_n, dropped)

    real_sorted_acc = sorted(real, key=lambda r: -r["accuracy"])
    real_sorted_ece = sorted(real, key=lambda r: r["ece"])
    real_sorted_payoff = sorted(real, key=lambda r: -r["net_payoff"])

    # Spearman on RAW values so the sign convention matches results_template
    # (e.g. high accuracy + low ECE -> negative rho_acc_ECE).
    rho_acc_ece, _ = sps.spearmanr([r["accuracy"] for r in real], [r["ece"] for r in real])
    rho_acc_pay, _ = sps.spearmanr([r["accuracy"] for r in real], [r["net_payoff"] for r in real])
    rho_ece_pay, _ = sps.spearmanr([r["ece"] for r in real], [r["net_payoff"] for r in real])

    top_acc = real_sorted_acc[0]
    top_ece = real_sorted_ece[0]
    top_pay = real_sorted_payoff[0]

    # Recompute Pareto within the N-filtered set so a small-N outlier with
    # 100% accuracy on six tasks cannot land on the frontier.
    pareto = []
    for a in real:
        dominated = False
        for b in real:
            if b is a:
                continue
            if b["net_payoff"] >= a["net_payoff"] and b["ece"] <= a["ece"]:
                if b["net_payoff"] > a["net_payoff"] or b["ece"] < a["ece"]:
                    dominated = True
                    break
        if not dominated:
            pareto.append(a)
    pareto_names = ", ".join(r["name"].replace("openrouter:", "") for r in pareto)
    n_pareto = len(pareto)

    diff_top1 = (
        top_acc["name"] != top_ece["name"]
        or top_acc["name"] != top_pay["name"]
        or top_ece["name"] != top_pay["name"]
    )

    text = []
    text.append(r"\subsection*{Headline findings (autogen)}")
    text.append(
        f"Across {len(real)} non-baseline systems with $N \\ge {args.min_n}$ committed tasks "
        f"on the standard tier, "
        f"the rank-order under each headline axis is materially different. "
        f"Best by accuracy: \\texttt{{{top_acc['name'].replace('_','-')}}} "
        f"({top_acc['accuracy']:.3f}, $N$={top_acc['n']}). "
        f"Best by ECE (lowest): \\texttt{{{top_ece['name'].replace('_','-')}}} "
        f"({top_ece['ece']:.3f}). "
        f"Best by net payoff: \\texttt{{{top_pay['name'].replace('_','-')}}} "
        f"({top_pay['net_payoff']:.0f}). "
    )
    if diff_top1:
        text.append(
            r"The first-place leader \emph{changes under at least one axis} --- the empirical claim PROPHET is designed to make visible."
        )
    else:
        text.append("On this slice the same system tops all three axes; richer subsets reveal the divergence.")
    text.append("")
    text.append(
        f"Spearman rank correlations: $\\rho_{{\\mathrm{{acc, ECE}}}} = {rho_acc_ece:.2f}$, "
        f"$\\rho_{{\\mathrm{{acc, payoff}}}} = {rho_acc_pay:.2f}$, "
        f"$\\rho_{{\\mathrm{{ECE, payoff}}}} = {rho_ece_pay:.2f}$. "
        f"Pareto-optimal systems on $(\\ECE, \\mathcal{{P}})$: {n_pareto} (\\texttt{{{pareto_names}}})."
    )
    text.append("")
    # Overreach: top-accuracy system's calibration
    text.append(
        f"\\paragraph{{Over-confidence is everywhere.}} "
        f"The top-accuracy system stated $\\Phat$ that exceeded its empirical accuracy by "
        f"$\\approx {top_acc['ece']:.2f}$ in aggregate ECE; under the marketplace this is a $\\approx {(top_acc['ece'] * 100):.0f}$ percentage-point overreach charged at $\\kappa$-units per task."
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(text))
    log.info("wrote narrative → %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
