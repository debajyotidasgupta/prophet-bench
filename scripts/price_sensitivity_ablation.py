#!/usr/bin/env python3
"""Price-sensitivity ablation for reviewer defense.

Recomputes the leaderboard under perturbed (V, C, kappa, delta_pass) prices
and reports the rank-correlation with the canonical leaderboard.

A robust ranking signals the calibration finding is not an artefact of our
arbitrary price-table choice.

Output: docs/paper/figures/price_sensitivity_table.tex / .md
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, kendalltau


PRICE_PERTURBATIONS = [
    ("baseline", 1.0, 1.0, 1.0, 1.0),
    ("V × 2", 2.0, 1.0, 1.0, 1.0),
    ("C × 2 (higher penalty)", 1.0, 2.0, 1.0, 1.0),
    ("κ × 2 (heavier calib weight)", 1.0, 1.0, 2.0, 1.0),
    ("κ × 0.5 (lighter calib)", 1.0, 1.0, 0.5, 1.0),
    ("δ_pass × 2", 1.0, 1.0, 1.0, 2.0),
    ("δ_pass × 0.5", 1.0, 1.0, 1.0, 0.5),
    ("asymmetric: C × 3", 1.0, 3.0, 1.0, 1.0),
]


def recompute_payoff(outcome: dict, sV: float, sC: float, sk: float, sdp: float) -> float:
    mode = outcome["response"]["mode"]
    conf = outcome["response"]["confidence"]
    V = outcome["offer"]["V_success"] * sV
    C = outcome["offer"]["C_failure"] * sC
    kappa = outcome["offer"]["kappa_calib"] * sk
    dp = outcome["offer"]["delta_pass"] * sdp
    y = outcome.get("success")
    if mode == "pass" or y is None:
        return -dp
    if y:
        att = V
    else:
        att = -C
    if mode == "take":
        return att
    # quote
    brier = (conf - (1.0 if y else 0.0)) ** 2
    calib = kappa * (1.0 - 4.0 * brier)
    return att + calib


def agent_summary(run_dir: Path, sV: float, sC: float, sk: float, sdp: float) -> dict | None:
    """Recompute payoff sum and accuracy for a single agent run dir under perturbed prices."""
    fp = run_dir / "outcomes.jsonl"
    if not fp.exists():
        return None
    rows = [json.loads(l) for l in open(fp) if l.strip()]
    if not rows:
        return None
    payoffs = [recompute_payoff(r, sV, sC, sk, sdp) for r in rows]
    committed = [r for r in rows if r.get("success") is not None]
    acc = sum(1 for r in committed if r["success"]) / len(committed) if committed else 0.0
    return {
        "name": run_dir.name.split("-", 2)[-1],
        "payoff": float(sum(payoffs)),
        "accuracy": acc,
        "n": len(rows),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path, default=Path("results/openrouter_hard_tier/runs"))
    ap.add_argument("--out", type=Path, default=Path("docs/paper/figures/price_sensitivity_table.tex"))
    args = ap.parse_args()

    # Gather one summary per agent per perturbation
    by_pert: dict[str, list[dict]] = {}
    for label, sV, sC, sk, sdp in PRICE_PERTURBATIONS:
        summaries = []
        for run_dir in sorted(args.runs_dir.iterdir()):
            s = agent_summary(run_dir, sV, sC, sk, sdp)
            if s and not s["name"].startswith("baseline:"):
                summaries.append(s)
        by_pert[label] = sorted(summaries, key=lambda s: -s["payoff"])

    # Rank-correlation of each perturbation vs baseline
    baseline_ranks = {s["name"]: i for i, s in enumerate(by_pert["baseline"])}
    correlations = []
    for label, summaries in by_pert.items():
        if label == "baseline":
            continue
        ranks_b = [baseline_ranks[s["name"]] for s in summaries if s["name"] in baseline_ranks]
        ranks_p = [i for i, s in enumerate(summaries) if s["name"] in baseline_ranks]
        if len(ranks_b) < 3:
            continue
        rho, _ = spearmanr(ranks_b, ranks_p)
        tau, _ = kendalltau(ranks_b, ranks_p)
        correlations.append((label, rho, tau, len(ranks_b)))

    # LaTeX table
    args.out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table}[t]\centering\small",
        r"\caption{Price-sensitivity ablation: Spearman $\rho$ and Kendall $\tau$ of "
        r"the perturbed-price ranking against the canonical leaderboard ranking. Strong "
        r"correlation (>0.9) across all perturbations indicates the calibration finding "
        r"is robust to our arbitrary price-table choice.}\label{tab:price-sensitivity}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Perturbation & Spearman $\rho$ & Kendall $\tau$ & $N$ agents \\",
        r"\midrule",
    ]
    md_lines = ["# Price-sensitivity ablation", "", "| Perturbation | Spearman ρ | Kendall τ | N |", "|---|---|---|---|"]
    for label, rho, tau, n in correlations:
        # Wrap delta_pass tokens in math mode to handle the underscore
        label_tex = (
            label.replace("×", r"$\times$")
                 .replace("κ", r"$\kappa$")
                 .replace("δ_pass", r"$\delta_{\text{pass}}$")
        )
        lines.append(f"{label_tex} & {rho:.3f} & {tau:.3f} & {n} \\\\")
        md_lines.append(f"| {label} | {rho:.3f} | {tau:.3f} | {n} |")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    args.out.write_text("\n".join(lines))
    args.out.with_suffix(".md").write_text("\n".join(md_lines))
    print(f"Wrote {args.out} and .md")
    print()
    print("=== Price sensitivity (ρ should be ≥0.9) ===")
    for label, rho, tau, n in correlations:
        flag = " ✓" if rho >= 0.9 else (" ⚠" if rho >= 0.7 else " ✗")
        print(f"  {label:<35} ρ={rho:.3f}  τ={tau:.3f}  N={n}{flag}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
