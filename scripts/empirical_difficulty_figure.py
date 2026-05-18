#!/usr/bin/env python3
"""Generate the empirical-difficulty calibration figure for the paper appendix.

For each T_extreme generator, plot panel-mean accuracy across the 16 paid
agents. The target band [5%, 20%] is highlighted. This is the reviewer-defense
evidence that our hand-assigned difficulty tier is at the right operating
point.

Output: docs/paper/figures/empirical_difficulty.{pdf,png}
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("results/openrouter_hard_tier/runs"),
        help="Hard-tier matrix runs directory.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("docs/paper/figures/empirical_difficulty.pdf"),
    )
    args = ap.parse_args()

    import sys
    sys.path.insert(0, "src")
    from prophet.families.math_family import MathFamily
    from prophet.families.reasoning_family import ReasoningFamily
    from prophet.families.code_family import CodeFamily
    from prophet.families.knowledge_family import KnowledgeFamily
    from prophet.families.multimodal_family import MultimodalFamily

    FAMS = {
        "math": MathFamily(),
        "reasoning": ReasoningFamily(),
        "code": CodeFamily(),
        "knowledge": KnowledgeFamily(),
        "multimodal": MultimodalFamily(),
    }

    # Build (family, task_idx_str) -> generator name
    task_gen = {}
    for fname, fam in FAMS.items():
        tasks = fam.generate(n=40, seed=42, difficulty_range=(0.97, 1.0))
        for t in tasks:
            task_gen[(fname, t.task_id)] = t.metadata.get("generator", "unknown")

    per_gen = defaultdict(lambda: {"committed": 0, "correct": 0, "pass": 0})
    n_agents = 0
    for run_dir in sorted(args.runs_dir.iterdir()):
        if not (run_dir / "outcomes.jsonl").exists():
            continue
        agent = run_dir.name.split("-", 2)[-1]
        if agent.startswith("baseline:"):
            continue
        n_agents += 1
        for line in open(run_dir / "outcomes.jsonl"):
            if not line.strip():
                continue
            r = json.loads(line)
            gen = task_gen.get((r["family"], r["task_id"]), "unknown")
            key = (r["family"], gen)
            if r.get("success") is None:
                per_gen[key]["pass"] += 1
            else:
                per_gen[key]["committed"] += 1
                if r["success"]:
                    per_gen[key]["correct"] += 1

    rows: list[tuple[str, str, float, int, int]] = []
    for (fam, gen), s in sorted(per_gen.items()):
        n = s["committed"]
        if n == 0:
            continue
        acc = s["correct"] / n
        rows.append((fam, gen, acc, n, s["pass"]))
    rows.sort(key=lambda r: r[2])  # ascending difficulty

    family_colors = {
        "math": "#1f77b4",
        "reasoning": "#ff7f0e",
        "code": "#2ca02c",
        "knowledge": "#d62728",
        "multimodal": "#9467bd",
    }

    fig, ax = plt.subplots(figsize=(9, 5.5))
    y_pos = np.arange(len(rows))
    for i, (fam, gen, acc, n, npass) in enumerate(rows):
        ax.barh(i, acc * 100, color=family_colors.get(fam, "#888"), alpha=0.85, label=fam)
        ax.text(acc * 100 + 1, i, f" {n} commits, {npass} pass", va="center", fontsize=8)
    # Target band 5-20%
    ax.axvspan(5, 20, color="green", alpha=0.10)
    ax.axvline(20, color="green", ls="--", lw=1.0, alpha=0.5)
    ax.axvline(5, color="green", ls="--", lw=1.0, alpha=0.5)
    ax.text(12.5, len(rows), "target band [5%, 20%]", color="green", ha="center", fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{fam}/{gen.replace('_gen_text_', '')}" for fam, gen, _, _, _ in rows], fontsize=8)
    ax.set_xlabel("Panel-mean committed accuracy (%)")
    ax.set_xlim(0, 110)
    ax.set_title(f"T-extreme empirical difficulty across the {n_agents}-agent panel")
    # Dedup legend
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    uniq_h, uniq_l = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            uniq_h.append(h)
            uniq_l.append(l)
    ax.legend(uniq_h, uniq_l, loc="lower right", fontsize=8, framealpha=0.9)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    fig.savefig(args.out.with_suffix(".png"), dpi=140)
    plt.close(fig)
    print(f"Wrote {args.out} + .png")
    print()
    print("=== Empirical difficulty per generator ===")
    for fam, gen, acc, n, npass in rows:
        flag = "✓" if 0.05 <= acc <= 0.5 else ("⚠ easy" if acc > 0.5 else "⚠ hard")
        print(f"  {fam:<12} {gen[:35]:<35} acc={acc*100:5.1f}%  n={n:3d}  pass={npass:3d}  {flag}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
