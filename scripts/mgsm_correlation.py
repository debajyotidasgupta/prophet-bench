#!/usr/bin/env python3
"""Cross-correlation of MGSM accuracy vs PROPHET-multilingual metrics.

After `mgsm_external_baseline.py` writes per-agent MGSM accuracies,
this script joins them to PROPHET-multilingual accuracy / ECE / payoff
from `results/openrouter_matrix/runs/` and computes Spearman + Pearson
rank correlations. Emits a paper-ready table + a scatter.

Output:
  docs/paper/figures/mgsm_correlation.{tex,md,json}
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


def load_mgsm(runs_dir: Path) -> dict[str, dict]:
    """agent_short -> { 'overall_acc': float, 'per_lang': {...} }."""
    out: dict[str, dict] = {}
    for p in sorted(runs_dir.glob("*.json")):
        d = json.loads(p.read_text())
        s = d["summary"]
        agent = s["agent"].replace("/", "-").replace("_", "-")
        out[agent] = s
    return out


def load_prophet_multi(runs_dir: Path) -> dict[str, dict]:
    """agent_short -> {'acc': ..., 'ece': ..., 'payoff_per_task': ...}."""
    confidences: dict[str, list[float]] = defaultdict(list)
    correct: dict[str, list[int]] = defaultdict(list)
    payoffs: dict[str, list[float]] = defaultdict(list)
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        p = d / "outcomes.jsonl"
        if not p.exists():
            continue
        agent = d.name.split("-", 2)[-1]
        if agent.startswith("baseline:"):
            continue
        agent = agent.replace("openrouter:", "").replace("_", "-").replace("/", "-")
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            o = json.loads(line)
            if o.get("family") != "multilingual":
                continue
            payoffs[agent].append(float(o.get("payoff_total", 0) or 0))
            if o.get("success") is None:  # PASS
                continue
            mode = o["response"].get("mode")
            if mode == "pass":
                continue
            confidences[agent].append(float(o["response"].get("confidence", 0.5)))
            correct[agent].append(1 if o["success"] else 0)
    out: dict[str, dict] = {}
    for a, c in confidences.items():
        if not c:
            continue
        confs = np.asarray(c)
        ys = np.asarray(correct[a])
        bins = np.linspace(0, 1, 11)
        ece = 0.0
        for k in range(10):
            mask = (confs >= bins[k]) & (confs < bins[k + 1])
            if k == 9:
                mask = (confs >= bins[k]) & (confs <= bins[k + 1])
            if mask.sum() == 0:
                continue
            ece += mask.sum() / len(confs) * abs(confs[mask].mean() - ys[mask].mean())
        out[a] = {
            "acc": float(ys.mean()),
            "ece": float(ece),
            "n": int(len(ys)),
            "payoff_per_task": float(np.mean(payoffs[a])) if payoffs[a] else None,
        }
    return out


def _spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(xs))
    ry = np.argsort(np.argsort(ys))
    return float(np.corrcoef(rx, ry)[0, 1])


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mgsm-dir", type=Path,
                    default=Path("results/mgsm/runs"))
    ap.add_argument("--prophet-dir", type=Path,
                    default=Path("results/openrouter_matrix/runs"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("docs/paper/figures"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    mgsm = load_mgsm(args.mgsm_dir)
    prophet = load_prophet_multi(args.prophet_dir)

    print(f"MGSM agents: {len(mgsm)}    PROPHET-multilingual agents: {len(prophet)}")
    # Join on agent slug. Allow loose match (mgsm slug vs prophet slug normalisation).
    def norm(s: str) -> str:
        return s.lower().replace("-2507", "").replace("a22b-thinking", "thinking")

    joined: list[dict] = []
    for ma, ms in mgsm.items():
        n_ma = norm(ma)
        match = None
        for pa, ps in prophet.items():
            if norm(pa).endswith(n_ma.split("-", 1)[-1]) or n_ma.split("-", 1)[-1] in norm(pa):
                match = (pa, ps)
                break
        if not match:
            print(f"  no PROPHET match for {ma}")
            continue
        pa, ps = match
        joined.append({
            "agent": ma,
            "mgsm_acc": ms["overall_acc"],
            "prophet_acc": ps["acc"],
            "prophet_ece": ps["ece"],
            "prophet_payoff_per_task": ps["payoff_per_task"],
            "n_mgsm": ms["n_calls"],
            "n_prophet": ps["n"],
        })

    if len(joined) < 3:
        print("Not enough agents joined to compute correlation; aborting.")
        return 1

    mgsm_accs = [j["mgsm_acc"] for j in joined]
    p_accs = [j["prophet_acc"] for j in joined]
    p_eces = [j["prophet_ece"] for j in joined]
    p_payoffs = [j["prophet_payoff_per_task"] for j in joined]

    correlations = {
        "spearman_mgsm_vs_prophet_acc": _spearman(mgsm_accs, p_accs),
        "pearson_mgsm_vs_prophet_acc": _pearson(mgsm_accs, p_accs),
        "spearman_mgsm_vs_prophet_ece": _spearman(mgsm_accs, p_eces),
        "pearson_mgsm_vs_prophet_ece": _pearson(mgsm_accs, p_eces),
        "spearman_mgsm_vs_prophet_payoff": _spearman(mgsm_accs, p_payoffs),
        "pearson_mgsm_vs_prophet_payoff": _pearson(mgsm_accs, p_payoffs),
        "n_agents": len(joined),
    }
    print()
    print("=== Cross-correlation: MGSM vs PROPHET-multilingual ===")
    for k, v in correlations.items():
        print(f"  {k:<40} {v}")

    # Save artefacts
    (args.out_dir / "mgsm_correlation.json").write_text(
        json.dumps({"correlations": correlations, "joined": joined}, indent=2))

    md = ["# MGSM cross-correlation\n",
          f"N agents = {len(joined)}\n",
          "| agent | MGSM acc | PROPHET acc | PROPHET ECE | PROPHET payoff/task |",
          "|---|---|---|---|---|"]
    for j in sorted(joined, key=lambda r: -r["mgsm_acc"]):
        pp_md = (f"{j['prophet_payoff_per_task']:.1f}"
                 if j['prophet_payoff_per_task'] is not None else "n/a")
        md.append(f"| {j['agent']} | {j['mgsm_acc']:.3f} | {j['prophet_acc']:.3f} | "
                  f"{j['prophet_ece']:.3f} | {pp_md} |")
    md.append("")
    md.append("## Correlations\n")
    for k, v in correlations.items():
        if isinstance(v, float):
            md.append(f"- {k} = {v:.3f}")
        else:
            md.append(f"- {k} = {v}")
    (args.out_dir / "mgsm_correlation.md").write_text("\n".join(md))

    tex = [r"\begin{table}[t]\centering\small",
           r"\caption{MGSM external-benchmark cross-correlation. MGSM accuracy is the "
           r"overall correct rate across 4 languages (en, es, fr, de), 30 problems per "
           r"language, with answer extracted as the last integer in the response. "
           f"Joined to PROPHET-multilingual on the same {len(joined)} agents. "
           r"Spearman rank correlations:}",
           r"\label{tab:mgsm-correlation}",
           r"\begin{tabular}{lrrrr}",
           r"\toprule",
           r"Agent & MGSM acc & PROPHET acc & PROPHET ECE & PROPHET P/task \\",
           r"\midrule"]
    for j in sorted(joined, key=lambda r: -r["mgsm_acc"]):
        pp = (f"{j['prophet_payoff_per_task']:.1f}"
              if j['prophet_payoff_per_task'] is not None else "---")
        tex.append(f"\\texttt{{{j['agent']}}} & {j['mgsm_acc']:.3f} & {j['prophet_acc']:.3f} "
                   f"& {j['prophet_ece']:.3f} & {pp} \\\\")
    tex.append(r"\midrule")
    sa = correlations['spearman_mgsm_vs_prophet_acc']
    se = correlations['spearman_mgsm_vs_prophet_ece']
    sp = correlations['spearman_mgsm_vs_prophet_payoff']
    tex.append(rf"\multicolumn{{2}}{{l}}{{Spearman $\rho$ (MGSM acc, $\cdot$)}} & "
               rf"{sa:.3f} & {se:.3f} & {sp:.3f} \\")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}\end{table}")
    (args.out_dir / "mgsm_correlation.tex").write_text("\n".join(tex))

    # Scatter plot
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(mgsm_accs, p_accs, c="tab:blue", label="PROPHET acc")
    ax.set_xlabel("MGSM accuracy")
    ax.set_ylabel("PROPHET-multilingual accuracy")
    ax.set_title(f"MGSM × PROPHET-multilingual  (N = {len(joined)} agents)")
    ax.grid(True, alpha=0.3)
    rho = correlations["spearman_mgsm_vs_prophet_acc"]
    ax.text(0.05, 0.95, fr"$\rho_S = {rho:.3f}$",
            transform=ax.transAxes, va="top", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out_dir / "mgsm_correlation.pdf")
    fig.savefig(args.out_dir / "mgsm_correlation.png", dpi=140)
    plt.close(fig)
    print(f"\nWrote {args.out_dir / 'mgsm_correlation.*'}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
