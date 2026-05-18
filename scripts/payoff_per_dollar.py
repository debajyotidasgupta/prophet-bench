#!/usr/bin/env python3
"""Compute payoff-per-dollar leaderboard and the (cost, payoff) Pareto frontier.

H3 in the paper claims open-weight agents (Qwen3-235B-thinking) compete on
the cost-conscious deployment axis even when dominated on the (ECE, payoff)
axis. This script makes that claim quantitative.

Reads:
  results/openrouter_matrix/leaderboard/leaderboard.json
  results/openrouter_hard_v2/leaderboard/leaderboard.json
Writes:
  docs/paper/figures/payoff_per_dollar.tex/.md
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


def _cost_payoff_pareto(records: list[dict]) -> set[str]:
    """Pareto frontier on (cost_usd, net_payoff): minimise cost, maximise payoff."""
    out: set[str] = set()
    real = [r for r in records if not r["name"].startswith("baseline:") and r.get("cost_usd", 0) > 0]
    for a in real:
        dominated = False
        for b in real:
            if b is a:
                continue
            if b["cost_usd"] <= a["cost_usd"] and b["net_payoff"] >= a["net_payoff"]:
                if b["cost_usd"] < a["cost_usd"] or b["net_payoff"] > a["net_payoff"]:
                    dominated = True
                    break
        if not dominated:
            out.add(a["name"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--standard", type=Path,
                    default=Path("results/openrouter_matrix/leaderboard/leaderboard.json"))
    ap.add_argument("--hard", type=Path,
                    default=Path("results/openrouter_hard_v2/leaderboard/leaderboard.json"))
    ap.add_argument("--min-n-std", type=int, default=100)
    ap.add_argument("--min-n-hard", type=int, default=20)
    ap.add_argument("--out", type=Path,
                    default=Path("docs/paper/figures/payoff_per_dollar.tex"))
    args = ap.parse_args()

    std = json.loads(args.standard.read_text())
    hard = json.loads(args.hard.read_text())

    std_filt = [r for r in std if r["n"] >= args.min_n_std and not r["name"].startswith("baseline:")]
    hard_filt = [r for r in hard if r["n"] >= args.min_n_hard and not r["name"].startswith("baseline:")]

    std_pareto = _cost_payoff_pareto(std_filt)
    hard_pareto = _cost_payoff_pareto(hard_filt)

    # Per-agent rows for the union
    by_name: dict[str, dict] = {}
    for r in std_filt:
        by_name.setdefault(r["name"], {})["std"] = r
    for r in hard_filt:
        by_name.setdefault(r["name"], {})["hard"] = r

    rows = []
    for name, entry in by_name.items():
        s = entry.get("std")
        h = entry.get("hard")
        if s and s.get("cost_usd", 0) > 0:
            std_ppd = s["net_payoff"] / s["cost_usd"]
            std_payoff = s["net_payoff"]
            std_cost = s["cost_usd"]
        else:
            std_ppd = std_payoff = std_cost = float("nan")
        if h and h.get("cost_usd", 0) > 0:
            hard_ppd = h["net_payoff"] / h["cost_usd"]
            hard_payoff = h["net_payoff"]
            hard_cost = h["cost_usd"]
        else:
            hard_ppd = hard_payoff = hard_cost = float("nan")
        rows.append({
            "name": name,
            "std_payoff": std_payoff,
            "std_cost": std_cost,
            "std_ppd": std_ppd,
            "hard_payoff": hard_payoff,
            "hard_cost": hard_cost,
            "hard_ppd": hard_ppd,
            "std_pareto_cost_payoff": name in std_pareto,
            "hard_pareto_cost_payoff": name in hard_pareto,
        })

    # Sort by std_ppd descending (NaN sinks to bottom)
    rows.sort(key=lambda r: -r["std_ppd"] if r["std_ppd"] == r["std_ppd"] else float("inf"))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tex = [
        r"\begin{table}[t]\centering\small",
        r"\caption{Payoff per US dollar of inference spend, by agent and tier. "
        r"$\bullet$ = Pareto-optimal on the $(\mathrm{cost}, \mathcal{P})$ frontier "
        rf"(min-$N$ filter: Std $N \ge {args.min_n_std}$, Ext $N \ge {args.min_n_hard}$). "
        r"This is the deployment-economics view that H3 invokes: an open-weight system "
        r"that is dominated on $(\ECE, \mathcal{P})$ can still be Pareto-optimal here.}",
        r"\label{tab:payoff-per-dollar}",
        r"\begin{tabular}{lrrr|rrr}",
        r"\toprule",
        r" & \multicolumn{3}{c}{Standard tier} & \multicolumn{3}{c}{T\textsubscript{extreme}} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"Agent & $\mathcal{P}$ & Cost \$ & $\mathcal{P}/\$$ & $\mathcal{P}$ & Cost \$ & $\mathcal{P}/\$$ \\",
        r"\midrule",
    ]
    md = ["# Payoff per dollar", "",
          "| Agent | Std $\\mathcal{P}$ | Std $ | Std $\\mathcal{P}$/$ | Hard $\\mathcal{P}$ | Hard $ | Hard $\\mathcal{P}$/$ |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        name = _short(r["name"]).replace("_", "-")
        sm = r"$\bullet$" if r["std_pareto_cost_payoff"] else ""
        hm = r"$\bullet$" if r["hard_pareto_cost_payoff"] else ""
        if r["std_payoff"] == r["std_payoff"]:
            std_cells = f"{r['std_payoff']:.0f}{sm} & {r['std_cost']:.2f} & {r['std_ppd']:.0f}"
        else:
            std_cells = "--- & --- & ---"
        if r["hard_payoff"] == r["hard_payoff"]:
            hard_cells = f"{r['hard_payoff']:.0f}{hm} & {r['hard_cost']:.2f} & {r['hard_ppd']:.0f}"
        else:
            hard_cells = "--- & --- & ---"
        tex.append(f"{name} & {std_cells} & {hard_cells} \\\\")
        md.append(f"| {name} | {std_cells.replace(' & ', ' | ')} | {hard_cells.replace(' & ', ' | ')} |")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    args.out.write_text("\n".join(tex))
    md_out = args.out.with_suffix(".md")
    md_out.write_text("\n".join(md))
    print(f"Wrote {args.out}")
    print(f"Wrote {md_out}")
    print()
    print("Std cost-payoff Pareto:", sorted(std_pareto))
    print("Hard cost-payoff Pareto:", sorted(hard_pareto))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
