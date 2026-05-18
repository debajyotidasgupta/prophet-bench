#!/usr/bin/env python3
"""EMNLP-1: per-language breakdown of the multilingual family.

Strong NLP-venue reviewers will want to know WHICH languages each model
struggles with, not a single aggregate ECE for the entire multilingual
family. This script:

  1. Regenerates the multilingual tasks from the seeds used in the matrix.
  2. Joins each outcome to its task's language via metadata fields
     (lang_code / lang / target_lang -- generator-specific).
  3. Computes per-(agent, language) accuracy, ECE, and N for the
     standard-tier outcomes pool.
  4. Writes a paper-ready table + a heatmap.

Outputs:
  docs/paper/figures/multilingual_per_language.tex
  docs/paper/figures/multilingual_per_language.md
  docs/paper/figures/multilingual_per_language.json
  docs/paper/figures/heatmap_multilingual_per_lang.{pdf,png}
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _extract_language(meta: dict) -> str | None:
    """Return ISO-like language code from generator-specific metadata."""
    for key in ("lang_code", "lang", "target_lang"):
        if key in meta and meta[key]:
            return str(meta[key])
    return None


def _ece(confs: list[float], ys: list[int], n_bins: int = 10) -> float:
    if not confs:
        return float("nan")
    confs = np.asarray(confs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    out = 0.0
    n = len(confs)
    for i in range(n_bins):
        mask = (confs >= bins[i]) & (confs <= bins[i + 1] if i == n_bins - 1 else confs < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_conf = confs[mask].mean()
        bin_acc = ys[mask].mean()
        out += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return float(out)


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
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, "src")
    from prophet.families.multilingual_family import MultilingualFamily

    # Regenerate task pool, build task_id -> language map.
    lang_by_id: dict[str, str] = {}
    for seed in (42, 7, 1729):
        for n in (20, 40, 100, 240):
            tasks = MultilingualFamily().generate(n=n, seed=seed)
            for t in tasks:
                lang = _extract_language(t.metadata)
                if lang and t.task_id not in lang_by_id:
                    lang_by_id[t.task_id] = lang

    print(f"Regenerated multilingual task pool: {len(lang_by_id)} unique task_ids")
    print(f"Languages observed: {sorted(set(lang_by_id.values()))}")

    # Walk outcomes, bucket by (agent, language)
    confidences: dict[tuple[str, str], list[float]] = defaultdict(list)
    correct: dict[tuple[str, str], list[int]] = defaultdict(list)
    dispatched: dict[tuple[str, str], int] = defaultdict(int)
    agents = set()
    langs = set()

    for d in sorted(args.runs_dir.iterdir()):
        p = d / "outcomes.jsonl"
        if not p.exists():
            continue
        name = d.name.split("-", 2)[-1]
        if name.startswith("baseline:"):
            continue
        agents.add(name)
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            o = json.loads(line)
            if o.get("family") != "multilingual":
                continue
            lang = lang_by_id.get(o["task_id"])
            if lang is None:
                continue  # cognate cross-lingual task -- skip
            langs.add(lang)
            dispatched[(name, lang)] += 1
            if o.get("success") is not None:
                confidences[(name, lang)].append(float(o["response"]["confidence"]))
                correct[(name, lang)].append(int(bool(o["success"])))

    agents_sorted = sorted(agents)
    langs_sorted = sorted(langs)

    # Build per-cell stats
    cells = []
    n_a = len(agents_sorted)
    n_l = len(langs_sorted)
    acc_mat = np.full((n_l, n_a), np.nan)
    ece_mat = np.full((n_l, n_a), np.nan)
    n_mat = np.zeros((n_l, n_a), dtype=int)
    for i, lang in enumerate(langs_sorted):
        for j, ag in enumerate(agents_sorted):
            disp = dispatched.get((ag, lang), 0)
            cs = confidences.get((ag, lang), [])
            ys = correct.get((ag, lang), [])
            n_com = len(ys)
            n_mat[i, j] = n_com
            if n_com >= 3:
                acc = sum(ys) / n_com
                e = _ece(cs, ys)
                acc_mat[i, j] = acc
                ece_mat[i, j] = e
                cells.append({
                    "agent": ag, "language": lang,
                    "n_dispatched": disp, "n_committed": n_com,
                    "accuracy": round(acc, 4), "ece": round(e, 4),
                })

    # JSON
    (args.out_dir / "multilingual_per_language.json").write_text(json.dumps({
        "languages": langs_sorted, "agents": agents_sorted,
        "cells": cells,
    }, indent=2))

    # Markdown table
    md = ["# Per-language accuracy on the multilingual family", "",
          "| Agent | " + " | ".join(langs_sorted) + " |",
          "|---|" + "|".join(["---"] * n_l) + "|"]
    for j, ag in enumerate(agents_sorted):
        row = [_short(ag)]
        for i, lang in enumerate(langs_sorted):
            v = acc_mat[i, j]
            n = n_mat[i, j]
            cell = f"{v:.2f} (N={n})" if not np.isnan(v) else f"n/a (N={n})"
            row.append(cell)
        md.append("| " + " | ".join(row) + " |")
    (args.out_dir / "multilingual_per_language.md").write_text("\n".join(md))

    # LaTeX
    tex = [
        r"\begin{table}[t]\centering\small",
        r"\caption{Per-language committed accuracy on the multilingual "
        r"family, by agent and ISO language code. \texttt{n/a} = $<3$ "
        r"committed tasks (strategic abstention or insufficient sample). "
        r"Reasoning-tuned agents abstain more on lower-resource languages; "
        r"non-reasoning agents commit and lose accuracy.}",
        r"\label{tab:multilingual-per-lang}",
        r"\begin{tabular}{l" + "r" * n_l + r"}",
        r"\toprule",
        r"Agent & " + " & ".join(langs_sorted) + r" \\",
        r"\midrule",
    ]
    for j, ag in enumerate(agents_sorted):
        row = [_short(ag).replace("_", "-")]
        for i in range(n_l):
            v = acc_mat[i, j]
            cell = f"{v:.2f}" if not np.isnan(v) else r"--"
            row.append(cell)
        tex.append(" & ".join(row) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (args.out_dir / "multilingual_per_language.tex").write_text("\n".join(tex))

    # Heatmap
    fig, ax = plt.subplots(figsize=(0.6 + 0.55 * n_a, 0.6 + 0.45 * n_l))
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color="lightgrey")
    masked = np.ma.array(acc_mat, mask=np.isnan(acc_mat))
    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(n_a))
    ax.set_xticklabels([_short(a) for a in agents_sorted], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n_l))
    ax.set_yticklabels(langs_sorted, fontsize=9)
    for i in range(n_l):
        for j in range(n_a):
            v = acc_mat[i, j]
            txt = f"{v*100:.0f}" if not np.isnan(v) else "n/a"
            color = "white" if (not np.isnan(v) and v < 0.5) else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6, color=color)
    fig.colorbar(im, ax=ax, shrink=0.6, label="Accuracy")
    ax.set_title("Multilingual family: per-language committed accuracy by agent", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out_dir / "heatmap_multilingual_per_lang.png", dpi=140)
    fig.savefig(args.out_dir / "heatmap_multilingual_per_lang.pdf")
    plt.close(fig)

    print(f"Wrote per-language artefacts. Cells with N>=3: {len(cells)}")

    # Top-line stats
    print()
    print("=== Per-language panel-mean accuracy (across agents with N>=3) ===")
    for lang in langs_sorted:
        vals = [acc_mat[langs_sorted.index(lang), j] for j in range(n_a)
                if not np.isnan(acc_mat[langs_sorted.index(lang), j])]
        if vals:
            print(f"  {lang}:  mean acc = {np.mean(vals):.3f}   N agents = {len(vals)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
