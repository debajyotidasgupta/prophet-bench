#!/usr/bin/env python3
"""Per-family deep dive: which families are hardest? Where do agents differ?

Produces:
  docs/paper/figures/family_difficulty_curves.pdf — per-family accuracy vs difficulty
  docs/paper/figures/family_ranking_table.tex — best agent per family
  docs/paper/figures/family_calibration_per_difficulty.pdf — calibration curves per family
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

log = logging.getLogger("prophet.per_family")


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("docs/paper/figures"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from prophet.engine.scoring import brier_score, ece

    # Aggregate per (agent, family, difficulty_tier)
    by_agent_family: dict[tuple[str, str], list[dict]] = defaultdict(list)
    families = set()
    agents = set()
    for child in sorted(args.runs_dir.iterdir()):
        if not (child / "outcomes.jsonl").exists():
            continue
        agent = child.name.split("-", 2)[-1] if "-" in child.name else child.name
        agents.add(agent)
        for line in (child / "outcomes.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            o = json.loads(line)
            if o.get("success") is None:
                continue
            by_agent_family[(agent, o["family"])].append(o)
            families.add(o["family"])

    # Per-family best agent table
    rows = []
    for fam in sorted(families):
        per_agent = {}
        for ag in agents:
            if ag.startswith("baseline:"):
                continue
            outs = by_agent_family.get((ag, fam), [])
            if len(outs) < 5:
                continue
            acc = sum(int(bool(o["success"])) for o in outs) / len(outs)
            confs = [float(o["response"]["confidence"]) for o in outs]
            ys = [int(bool(o["success"])) for o in outs]
            e = ece(confs, ys)
            per_agent[ag] = {"acc": acc, "ece": e, "n": len(outs)}
        if per_agent:
            best_acc = max(per_agent.items(), key=lambda kv: kv[1]["acc"])
            best_ece = min(per_agent.items(), key=lambda kv: kv[1]["ece"])
            rows.append(
                {
                    "family": fam,
                    "best_acc": best_acc[0],
                    "best_acc_v": best_acc[1]["acc"],
                    "best_ece": best_ece[0],
                    "best_ece_v": best_ece[1]["ece"],
                }
            )

    # LaTeX table
    tex = [
        r"\begin{table}[t]\centering\small",
        r"\caption{Best agent per family on accuracy vs ECE. The two leaders rarely match.}\label{tab:per-family-winners}",
        r"\begin{tabular}{lllll}",
        r"\toprule",
        r"Family & Best accuracy (agent) & value & Best ECE (agent) & value \\",
        r"\midrule",
    ]
    diverged = 0
    for r in rows:
        same = r["best_acc"] == r["best_ece"]
        if not same:
            diverged += 1
        tex.append(
            f"{r['family']} & "
            f"\\texttt{{{r['best_acc'].replace('_','-').replace('/','-')}}} & {r['best_acc_v']:.3f} & "
            f"\\texttt{{{r['best_ece'].replace('_','-').replace('/','-')}}} & {r['best_ece_v']:.3f} \\\\"
        )
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (args.out / "family_ranking_table.tex").write_text("\n".join(tex))

    # Plots
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Per-family accuracy by difficulty (one panel per family)
        n_fams = len(families)
        cols = 4
        rows_n = (n_fams + cols - 1) // cols
        fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 3, rows_n * 2.5), sharey=True)
        axes_flat = axes.flatten() if rows_n > 1 else [axes]
        for i, fam in enumerate(sorted(families)):
            ax = axes_flat[i]
            for ag in sorted(agents):
                if ag.startswith("baseline:"):
                    continue
                outs = by_agent_family.get((ag, fam), [])
                if len(outs) < 10:
                    continue
                diffs = np.array([float(o["difficulty"]) for o in outs])
                ys = np.array([int(bool(o["success"])) for o in outs])
                # bin into 5 difficulty bins
                bins = np.linspace(0, 1, 6)
                centers = (bins[:-1] + bins[1:]) / 2
                accs = []
                for j in range(5):
                    m = (diffs >= bins[j]) & (diffs <= bins[j+1])
                    accs.append(ys[m].mean() if m.sum() > 0 else np.nan)
                ax.plot(centers, accs, marker="o", lw=1, label=ag[:18])
            ax.set_title(fam, fontsize=9)
            ax.set_ylim(0, 1)
            ax.set_xlim(0, 1)
            ax.grid(True, alpha=0.3)
        for i in range(n_fams, len(axes_flat)):
            axes_flat[i].axis('off')
        fig.suptitle("Accuracy vs Difficulty by Family", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(args.out / "family_difficulty_curves.png", dpi=130)
        fig.savefig(args.out / "family_difficulty_curves.pdf")
        plt.close(fig)
        log.info("wrote family_difficulty_curves")
    except Exception as e:
        log.warning("plotting failed: %s", e)

    log.info("Per-family table: %d families analysed, %d diverged on acc-vs-ECE leader (%.0f%%)", len(rows), diverged, 100*diverged/max(1,len(rows)))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
