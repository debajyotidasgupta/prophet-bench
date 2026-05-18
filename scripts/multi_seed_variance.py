#!/usr/bin/env python3
"""Multi-seed variance plot for reviewer defense.

Once Phase B (seeds 42, 7, 1729 on Pareto-essentials) completes, this
script produces a figure showing accuracy + ECE distributions across the 3
seeds for the 4 Pareto-essential agents (Opus, GPT-5, GPT-5.2,
DeepSeek-R1).

Output: docs/paper/figures/multi_seed_variance.pdf
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PARETO_AGENTS = [
    "openrouter:anthropic/claude-opus-4.7",
    "openrouter:openai/gpt-5",
    "openrouter:openai/gpt-5.2",
    "openrouter:deepseek/deepseek-r1",
]
SEEDS = [42, 7, 1729]


def agent_metrics(run_dir: Path) -> dict[str, float]:
    rows = [json.loads(l) for l in open(run_dir / "outcomes.jsonl") if l.strip()]
    committed = [r for r in rows if r.get("success") is not None]
    if not committed:
        return {"acc": float("nan"), "ece": float("nan"), "payoff": 0.0, "n": 0}
    acc = sum(1 for r in committed if r["success"]) / len(committed)
    # ECE: simple 10-bin
    bins = [[] for _ in range(10)]
    for r in committed:
        conf = r["response"]["confidence"]
        bi = min(int(conf * 10), 9)
        bins[bi].append((conf, 1 if r["success"] else 0))
    ece = 0.0
    total = len(committed)
    for b in bins:
        if not b:
            continue
        avg_conf = sum(c for c, _ in b) / len(b)
        acc_b = sum(y for _, y in b) / len(b)
        ece += (len(b) / total) * abs(avg_conf - acc_b)
    payoff = sum(r.get("payoff_total", 0) or 0 for r in rows)
    return {"acc": acc, "ece": ece, "payoff": float(payoff), "n": len(rows)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path, default=Path("results/openrouter_hard_v2/runs"))
    ap.add_argument("--out", type=Path, default=Path("docs/paper/figures/multi_seed_variance.pdf"))
    args = ap.parse_args()

    by_agent_seed: dict[tuple[str, int], dict] = {}
    for run_dir in sorted(args.runs_dir.iterdir()):
        if not (run_dir / "outcomes.jsonl").exists():
            continue
        s = json.loads((run_dir / "summary.json").read_text())
        agent = s["agent"]
        # Recover seed from cycle_seed if present
        seed = s.get("cycle_seed", s.get("seed", 42))
        if agent in PARETO_AGENTS:
            m = agent_metrics(run_dir)
            by_agent_seed[(agent, int(seed))] = m

    # Plot grouped bars: per agent, 3 seeds, 2 panels (acc, ECE)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    agent_short = {
        "openrouter:anthropic/claude-opus-4.7": "Opus 4.7",
        "openrouter:openai/gpt-5": "GPT-5",
        "openrouter:openai/gpt-5.2": "GPT-5.2",
        "openrouter:deepseek/deepseek-r1": "DeepSeek-R1",
    }
    x = np.arange(len(PARETO_AGENTS))
    width = 0.22
    for ax, metric, ylabel in [(ax1, "acc", "Accuracy (committed)"),
                                 (ax2, "ece", "ECE (10-bin)")]:
        for i, seed in enumerate(SEEDS):
            vals = [by_agent_seed.get((a, seed), {}).get(metric, float("nan"))
                    for a in PARETO_AGENTS]
            ax.bar(x + (i - 1) * width, vals, width, label=f"seed={seed}")
        ax.set_xticks(x)
        ax.set_xticklabels([agent_short[a] for a in PARETO_AGENTS], rotation=0)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Multi-seed variance on T_extreme — Pareto-essential agents")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    fig.savefig(args.out.with_suffix(".png"), dpi=140)
    plt.close(fig)
    print(f"Wrote {args.out}")
    print()
    print("=== Variance summary (μ ± stddev across 3 seeds) ===")
    for agent in PARETO_AGENTS:
        accs = [by_agent_seed.get((agent, s), {}).get("acc", float("nan")) for s in SEEDS]
        eces = [by_agent_seed.get((agent, s), {}).get("ece", float("nan")) for s in SEEDS]
        accs_v = [a for a in accs if not np.isnan(a)]
        eces_v = [e for e in eces if not np.isnan(e)]
        if accs_v:
            print(f"  {agent_short[agent]:<14} acc: μ={np.mean(accs_v):.3f} σ={np.std(accs_v):.3f}    "
                  f"ECE: μ={np.mean(eces_v):.3f} σ={np.std(eces_v):.3f}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
