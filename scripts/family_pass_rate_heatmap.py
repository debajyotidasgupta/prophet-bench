#!/usr/bin/env python3
"""Per-family PASS-rate heatmap + commit-count heatmap.

The accuracy heatmap renders NaN whenever an agent committed to zero
tasks in a family (acc = correct/committed is undefined at N=0). This
isn't a bug -- it's the marketplace signal that the agent strategically
abstained on every task of that family. This script surfaces that
behaviour directly:

  heatmap_pass_rate.pdf  -- fraction of tasks where the agent chose PASS
  heatmap_committed_n.pdf -- raw count of committed tasks per (agent, family)

Together with heatmap_accuracy these tell the full per-family story:
  PASS=100% -> NaN in accuracy heatmap   (strategic family-level abstention)
  PASS=0%   -> dense accuracy values     (agent commits to everything)

Writes:
  docs/paper/figures/heatmap_pass_rate.{pdf,png}
  docs/paper/figures/heatmap_committed_n.{pdf,png}
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


def _short(name: str) -> str:
    if name.startswith("baseline:"):
        return name.split(":", 1)[1]
    if name.startswith("openrouter:"):
        return name.split(":", 1)[1].split("/", 1)[-1].replace("_", "-")
    return name


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path,
                    default=Path("results/openrouter_matrix/runs"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("docs/paper/figures"))
    ap.add_argument("--min-dispatched", type=int, default=5,
                    help="Skip (agent, family) cells with fewer than this many "
                    "DISPATCHED tasks; these are non-runs, not abstentions.")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Pool by (agent, family): count dispatched, committed
    dispatched: dict[tuple[str, str], int] = defaultdict(int)
    committed: dict[tuple[str, str], int] = defaultdict(int)
    agents, families = set(), set()
    for d in sorted(args.runs_dir.iterdir()):
        p = d / "outcomes.jsonl"
        if not p.exists():
            continue
        name = d.name.split("-", 2)[-1] if "-" in d.name else d.name
        agents.add(name)
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            o = json.loads(line)
            fam = o.get("family")
            if not fam:
                continue
            families.add(fam)
            dispatched[(name, fam)] += 1
            if o.get("success") is not None:
                committed[(name, fam)] += 1

    # Exclude baseline:always-take which doesn't help the story (it
    # trivially has 0% PASS by construction); keep always-pass to anchor
    # the upper bound at 100% PASS.
    agents_sorted = sorted(
        [a for a in agents if a != "baseline:always-take"],
        key=lambda a: (a.startswith("baseline:"), a),
    )
    families_sorted = sorted(families)

    n_a = len(agents_sorted)
    n_f = len(families_sorted)
    pass_rate = np.full((n_f, n_a), np.nan)
    commit_count = np.full((n_f, n_a), 0, dtype=int)
    for i, fam in enumerate(families_sorted):
        for j, ag in enumerate(agents_sorted):
            disp = dispatched.get((ag, fam), 0)
            comm = committed.get((ag, fam), 0)
            commit_count[i, j] = comm
            if disp >= args.min_dispatched:
                pass_rate[i, j] = (disp - comm) / disp

    short_agents = [_short(a) for a in agents_sorted]

    # ---- Heatmap 1: PASS rate ----
    fig, ax = plt.subplots(figsize=(0.6 + 0.55 * n_a, 0.6 + 0.45 * n_f))
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad(color="lightgrey")
    masked = np.ma.array(pass_rate, mask=np.isnan(pass_rate))
    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(n_a))
    ax.set_xticklabels(short_agents, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n_f))
    ax.set_yticklabels(families_sorted, fontsize=8)
    for i in range(n_f):
        for j in range(n_a):
            v = pass_rate[i, j]
            txt = "n/a" if np.isnan(v) else f"{v*100:.0f}%"
            color = "white" if (not np.isnan(v) and v > 0.55) else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6, color=color)
    fig.colorbar(im, ax=ax, shrink=0.7, label="PASS rate")
    ax.set_title(
        f"Per-family PASS rate (fraction of dispatched tasks where agent chose PASS).\n"
        f"Cells where PASS=100\\% are exactly the NaN cells in heatmap_accuracy: "
        f"strategic family-level abstention.",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(args.out_dir / "heatmap_pass_rate.png", dpi=140)
    fig.savefig(args.out_dir / "heatmap_pass_rate.pdf")
    plt.close(fig)
    print(f"Wrote {args.out_dir / 'heatmap_pass_rate.pdf'}")

    # ---- Heatmap 2: committed N (raw count) ----
    fig, ax = plt.subplots(figsize=(0.6 + 0.55 * n_a, 0.6 + 0.45 * n_f))
    im = ax.imshow(commit_count, aspect="auto", cmap="viridis", vmin=0, vmax=20)
    ax.set_xticks(range(n_a))
    ax.set_xticklabels(short_agents, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n_f))
    ax.set_yticklabels(families_sorted, fontsize=8)
    for i in range(n_f):
        for j in range(n_a):
            v = commit_count[i, j]
            color = "white" if v < 8 else "black"
            ax.text(j, i, str(v), ha="center", va="center", fontsize=6, color=color)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Committed task count")
    ax.set_title(
        "Per-family committed-task count. Zeros explain the NaNs in heatmap_accuracy.",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(args.out_dir / "heatmap_committed_n.png", dpi=140)
    fig.savefig(args.out_dir / "heatmap_committed_n.pdf")
    plt.close(fig)
    print(f"Wrote {args.out_dir / 'heatmap_committed_n.pdf'}")

    # ---- Summary stats for the appendix ----
    print()
    print("=== Strategic-abstention cluster (>=2 families with 100% PASS) ===")
    rows = []
    for ag in agents_sorted:
        all_pass = sum(
            1 for fam in families_sorted
            if dispatched.get((ag, fam), 0) >= args.min_dispatched
            and committed.get((ag, fam), 0) == 0
        )
        if all_pass >= 2:
            rows.append((ag, all_pass))
    for ag, n in sorted(rows, key=lambda kv: -kv[1]):
        print(f"  {ag:<55} families with 100% PASS: {n}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
