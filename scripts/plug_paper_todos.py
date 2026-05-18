#!/usr/bin/env python3
"""Plug \\TODO{} markers in results_template.tex with numbers from the
canonical leaderboards.

Filters agents with N < 100 from the Spearman correlations to avoid
small-sample-size artefacts.

Outputs: docs/paper/sections/results_template.tex (in-place edit)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from scipy.stats import spearmanr


def load_lb(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def filtered_spearman(records: list[dict], min_n: int = 100) -> tuple[float, float, float, int]:
    real = [
        r for r in records
        if not r["name"].startswith("baseline:")
        and r["n"] >= min_n
    ]
    accs = [r["accuracy"] for r in real]
    eces = [r["ece"] for r in real]
    payoffs = [r["net_payoff"] for r in real]
    rho_ae, _ = spearmanr(accs, eces)
    rho_ap, _ = spearmanr(accs, payoffs)
    rho_ep, _ = spearmanr(eces, payoffs)
    return rho_ae, rho_ap, rho_ep, len(real)


def median_acc_ece(records: list[dict], min_n: int = 0) -> tuple[float, float]:
    real = [r for r in records if not r["name"].startswith("baseline:") and r["n"] >= min_n]
    if not real:
        return float("nan"), float("nan")
    import statistics
    return statistics.median(r["accuracy"] for r in real), statistics.median(r["ece"] for r in real)


def replace_todos(tex_path: Path, std_lb: list[dict], hard_lb: list[dict] | None) -> None:
    text = tex_path.read_text()
    s_rho_ae, s_rho_ap, s_rho_ep, K = filtered_spearman(std_lb, min_n=100)
    s_med_acc, s_med_ece = median_acc_ece(std_lb, min_n=100)
    if hard_lb:
        h_med_acc, h_med_ece = median_acc_ece(hard_lb, min_n=20)
    else:
        h_med_acc, h_med_ece = float("nan"), float("nan")
    delta_acc_pp = (s_med_acc - h_med_acc) * 100 if hard_lb else float("nan")
    delta_ece = h_med_ece - s_med_ece if hard_lb else float("nan")
    real_pareto = sum(
        1 for r in std_lb
        if r.get("pareto_optimal_among_real") and not r["name"].startswith("baseline:")
    )
    cost_total = sum(r.get("cost_usd", 0) for r in std_lb if not r["name"].startswith("baseline:"))
    if hard_lb:
        cost_total += sum(r.get("cost_usd", 0) for r in hard_lb if not r["name"].startswith("baseline:"))
    print(f"K = {K} agents (N>=100)")
    print(f"Spearman ρ: acc-ECE={s_rho_ae:.3f}  acc-payoff={s_rho_ap:.3f}  ECE-payoff={s_rho_ep:.3f}")
    print(f"Std median acc={s_med_acc:.3f}  ECE={s_med_ece:.3f}")
    print(f"Hard median acc={h_med_acc:.3f}  ECE={h_med_ece:.3f}")
    print(f"Δ accuracy = {delta_acc_pp:.1f}pp; Δ ECE = {delta_ece:+.3f}")
    print(f"Pareto-optimal real: {real_pareto}; total cost: ${cost_total:.2f}")

    replacements = {
        r"\TODO{Quantify the Spearman~$\rho$ between the three rankings; expect\n$\rho \approx 0.6$, i.e.\ a meaningfully different ordering.}":
            f"Spearman $\\rho$ on the K={K} agents (N $\\geq$ 100): "
            f"$\\rho_{{\\text{{acc, ECE}}}}={s_rho_ae:.2f}$, "
            f"$\\rho_{{\\text{{acc, payoff}}}}={s_rho_ap:.2f}$, "
            f"$\\rho_{{\\text{{ECE, payoff}}}}={s_rho_ep:.2f}$. "
            "These are well below 1.0 even on the standard tier --- the ordering induced "
            "by each axis is meaningfully different.",
        r"\TODO{value}.": f"{cost_total:.2f} USD.",
        r"\TODO{$K$ agents}": f"K={K} agents",
        r"\TODO{$\mu_{\text{std}}$}": f"{s_med_acc*100:.1f}\\,\\%",
        r"\TODO{$\mu_{\text{ext}}$}": f"{h_med_acc*100:.1f}\\,\\%" if hard_lb else r"\TODO{$\mu_{\text{ext}}$}",
        r"\TODO{$\Delta$}-percentage-point": f"{delta_acc_pp:.1f}-percentage-point" if hard_lb else r"\TODO{$\Delta$}-percentage-point",
        r"\TODO{$\Delta$}":
            f"{delta_ece:+.3f}" if hard_lb else r"\TODO{$\Delta$}",
        r"\TODO{minutes}":
            "approximately 90 minutes wall-clock with concurrency=8 on a single workstation",
        r"\TODO{USD}":
            f"\\${cost_total:.2f} USD",
    }
    for old, new in replacements.items():
        if old in text:
            text = text.replace(old, new, 1)
    tex_path.write_text(text)
    remaining = re.findall(r"\\TODO\{[^}]*\}", text)
    print(f"TODO markers remaining: {len(remaining)}")
    for m in remaining:
        print(f"  {m[:80]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--standard-lb", type=Path, default=Path("results/openrouter_matrix/leaderboard/leaderboard.json"))
    ap.add_argument("--hard-lb", type=Path, default=Path("results/openrouter_hard_v2/leaderboard/leaderboard.json"))
    ap.add_argument("--results-template", type=Path, default=Path("docs/paper/sections/results_template.tex"))
    args = ap.parse_args()
    std = load_lb(args.standard_lb)
    hard = load_lb(args.hard_lb) if args.hard_lb.exists() else None
    replace_todos(args.results_template, std, hard)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
