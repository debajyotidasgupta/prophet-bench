#!/usr/bin/env python3
"""Combine standard-tier and T_extreme leaderboards into a single paper table.

Reads (post-dedup, multi-seed-pooled by agent name):
  results/openrouter_matrix/leaderboard/leaderboard.json   (Standard tier)
  results/openrouter_hard_v2/leaderboard/leaderboard.json  (T_extreme tier,
      seeds {42, 7, 1729} on Pareto-essentials, seed 42 on the full panel)

Writes:
  docs/paper/figures/two_tier_leaderboard.tex
  docs/paper/figures/two_tier_leaderboard.md

The --min-n flags apply a minimum-N filter that gates BOTH (a) inclusion in
the output table and (b) the $\\star$ Pareto-optimal flag. Defaults follow the
paper's pre-registered analysis plan (N>=100 standard, N>=20 hard).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _short(name: str) -> str:
    if name.startswith("baseline:"):
        return name.split(":", 1)[1]
    if name.startswith("openrouter:"):
        return name.split(":", 1)[1].replace("/", "-")
    return name


def _pareto_filter(records: list[dict]) -> set[str]:
    real = [r for r in records if not r["name"].startswith("baseline:")]
    out: set[str] = set()
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
            out.add(a["name"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--standard",
        type=Path,
        default=Path("results/openrouter_matrix/leaderboard/leaderboard.json"),
    )
    ap.add_argument(
        "--hard",
        type=Path,
        default=Path("results/openrouter_hard_v2/leaderboard/leaderboard.json"),
    )
    ap.add_argument("--min-n-std", type=int, default=100)
    ap.add_argument("--min-n-hard", type=int, default=20)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("docs/paper/figures/two_tier_leaderboard.tex"),
    )
    args = ap.parse_args()

    if not args.standard.exists():
        print(f"ERROR: standard tier leaderboard not found at {args.standard}")
        return 1
    if not args.hard.exists():
        print(f"WARN: hard tier leaderboard not found at {args.hard} — writing single-tier only")
        hard_by_name: dict = {}
    else:
        hard_recs = json.loads(args.hard.read_text())
        hard_by_name = {r["name"]: r for r in hard_recs}

    std_recs = json.loads(args.standard.read_text())
    std_by_name = {r["name"]: r for r in std_recs}

    # Union the agent sets: include agents that exist on either tier so a
    # hard-tier-only run (e.g. GPT-5.2 was added late) isn't silently dropped.
    std_names = set(std_by_name.keys())
    hard_names = set(hard_by_name.keys()) if hard_by_name else set()
    for name in sorted(hard_names - std_names):
        h = hard_by_name[name]
        std_recs.append({
            "name": name,
            "n": 0,
            "accuracy": float("nan"),
            "ece": float("nan"),
            "brier": float("nan"),
            "logloss": float("nan"),
            "net_payoff": float("-inf"),  # sort to bottom
            "cost_usd": 0.0,
            "pareto_optimal": False,
            "pareto_optimal_among_real": False,
            "_std_missing": True,
        })

    # Sort by standard-tier net payoff descending; hard-only rows sink to the bottom
    std_recs.sort(key=lambda r: -r["net_payoff"])

    # Recompute Pareto restricted to agents with N >= threshold (so a small-N
    # outlier with 100% accuracy on 6 tasks cannot land on the frontier).
    std_filt = [r for r in std_recs if r["n"] >= args.min_n_std]
    hard_filt = [r for r in (hard_by_name.values() if hard_by_name else []) if r["n"] >= args.min_n_hard]
    std_pareto = _pareto_filter(std_filt)
    hard_pareto = _pareto_filter(hard_filt)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # LaTeX
    tex_lines = [
        r"\begin{table}[t]",
        r"\centering\small",
        r"\caption{Two-tier PROPHET leaderboard. \textbf{Std} = standard tier "
        r"(12 families, near-ceiling difficulty). \textbf{Ext} = "
        r"T\textsubscript{extreme} (5 families, $\delta \ge 0.97$). "
        r"Net payoff in PROPHET-cents; ECE on $[0, 1]$. "
        r"Multi-seed re-runs are pooled by agent name. "
        rf"$\star$ = Pareto-optimal on $(\ECE, \mathcal{{P}})$ among agents with "
        rf"$N \ge {args.min_n_std}$ (Std) / $N \ge {args.min_n_hard}$ (Ext); "
        r"small-$N$ rows are shown for transparency but do not receive the marker.}",
        r"\label{tab:two-tier}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{lrrrrr|rrrrr}",
        r"\toprule",
        r" & \multicolumn{5}{c}{Standard tier} & \multicolumn{5}{c}{T\textsubscript{extreme}} \\",
        r"\cmidrule(lr){2-6}\cmidrule(lr){7-11}",
        r"Agent & Acc & ECE & $\mathcal{P}$ & Cost & \# & Acc & ECE & $\mathcal{P}$ & Cost & \# \\",
        r"\midrule",
    ]
    md_lines = [
        "# PROPHET two-tier leaderboard",
        "",
        "| Agent | Std Acc | Std ECE | Std Payoff | Std $ | Std N | "
        "Ext Acc | Ext ECE | Ext Payoff | Ext $ | Ext N |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in std_recs:
        name = _short(r["name"])
        h = hard_by_name.get(r["name"])
        std_missing = r.get("_std_missing", False)
        if std_missing:
            std_acc_s = std_ece_s = std_p_s = std_cost_s = "---"
            std_n_s = "---"
        else:
            std_acc_s = f"{r['accuracy']:.3f}"
            std_ece_s = f"{r['ece']:.3f}"
            std_p_s = f"{r['net_payoff']:.0f}"
            std_cost_s = f"{r['cost_usd']:.2f}"
            std_n_s = f"{r['n']}"
        if h:
            hard_acc_s = f"{h['accuracy']:.3f}"
            hard_ece_s = f"{h['ece']:.3f}"
            hard_p_s = f"{h['net_payoff']:.0f}"
            hard_cost_s = f"{h['cost_usd']:.2f}"
            hard_n_s = f"{h['n']}"
        else:
            hard_acc_s = hard_ece_s = hard_p_s = hard_cost_s = hard_n_s = "---"
        std_star = r"$\star$" if (not std_missing and r["name"] in std_pareto) else ""
        hard_star = r"$\star$" if (h and h["name"] in hard_pareto) else ""
        agent_tex = name.replace("_", "-")
        tex_lines.append(
            f"{agent_tex} & {std_acc_s}{std_star} & {std_ece_s} & "
            f"{std_p_s} & {std_cost_s} & {std_n_s} & "
            f"{hard_acc_s}{hard_star} & {hard_ece_s} & {hard_p_s} & {hard_cost_s} & {hard_n_s}"
            + r" \\"
        )
        md_lines.append(
            f"| {name} | {std_acc_s} | {std_ece_s} | {std_p_s} | "
            f"${std_cost_s} | {std_n_s} | "
            f"{hard_acc_s} | {hard_ece_s} | {hard_p_s} | ${hard_cost_s} | {hard_n_s} |"
        )
    tex_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table}",
        ]
    )
    args.out.write_text("\n".join(tex_lines))
    md_path = args.out.with_suffix(".md")
    md_path.write_text("\n".join(md_lines))
    print(f"Wrote {args.out}")
    print(f"Wrote {md_path}")

    # Print a quick stdout summary too
    print()
    print("=" * 80)
    print(f"{'Agent':<40} {'Std Acc':>8} {'Ext Acc':>8} {'Std ECE':>8} {'Ext ECE':>8}")
    for r in std_recs[:15]:
        h = hard_by_name.get(r["name"])
        print(
            f"{_short(r['name']):<40} "
            f"{r['accuracy']:>8.3f} "
            f"{(h['accuracy'] if h else float('nan')):>8.3f} "
            f"{r['ece']:>8.3f} "
            f"{(h['ece'] if h else float('nan')):>8.3f}"
        )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
