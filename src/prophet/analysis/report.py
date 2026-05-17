"""Render a per-run HTML report with the key tables + plots.

Used by `prophet analyze <run_dir>`. Loads outcomes.jsonl, computes
calibration / Brier / payoff / MOP, and emits report.html + figures/*.png.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

log = logging.getLogger("prophet.analysis.report")


def _load_outcomes(run_dir: Path) -> list[dict]:
    p = run_dir / "outcomes.jsonl"
    if not p.exists():
        raise FileNotFoundError(p)
    out = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def build_report(run_dir: Path) -> Path:
    from prophet.engine.scoring import (
        adaptive_ece,
        brier_score,
        calibration_curve,
        ece,
        log_score,
        model_overreach_point,
    )
    outcomes = _load_outcomes(run_dir)
    # Reconstruct arrays
    by_family: dict[str, list[dict]] = defaultdict(list)
    for o in outcomes:
        by_family[o["family"]].append(o)
    rows: list[dict] = []
    for fam, outs in by_family.items():
        graded = [o for o in outs if o["success"] is not None]
        passed = [o for o in outs if o["success"] is None]
        if graded:
            confs = np.array([o["response"]["confidence"] for o in graded], dtype=float)
            ys = np.array([1 if o["success"] else 0 for o in graded], dtype=float)
            diffs = np.array([o["difficulty"] for o in graded], dtype=float)
            net = sum(o["payoff_total"] for o in outs)
            row = {
                "family": fam,
                "n": len(outs),
                "n_graded": len(graded),
                "n_passed": len(passed),
                "acc": float(ys.mean()),
                "ece": float(ece(confs, ys)),
                "adaptive_ece": float(adaptive_ece(confs, ys)),
                "brier": float(brier_score(confs, ys)),
                "logloss": float(log_score(confs, ys)),
                "net_payoff": net,
            }
            mop = model_overreach_point(confs, ys, diffs).mop_difficulty
            row["mop_difficulty"] = mop
        else:
            row = {
                "family": fam,
                "n": len(outs),
                "n_graded": 0,
                "n_passed": len(passed),
                "acc": None,
                "ece": None,
                "adaptive_ece": None,
                "brier": None,
                "logloss": None,
                "net_payoff": sum(o["payoff_total"] for o in outs),
                "mop_difficulty": None,
            }
        rows.append(row)
    rows.sort(key=lambda r: -(r["net_payoff"] or 0))

    # Aggregate / overall
    graded_all = [o for o in outcomes if o["success"] is not None]
    overall = {}
    if graded_all:
        confs = np.array([o["response"]["confidence"] for o in graded_all], dtype=float)
        ys = np.array([1 if o["success"] else 0 for o in graded_all], dtype=float)
        diffs = np.array([o["difficulty"] for o in graded_all], dtype=float)
        overall = {
            "n": len(outcomes),
            "n_graded": len(graded_all),
            "acc": float(ys.mean()),
            "ece": float(ece(confs, ys)),
            "brier": float(brier_score(confs, ys)),
            "logloss": float(log_score(confs, ys)),
            "net_payoff": sum(o["payoff_total"] for o in outcomes),
        }
        mop = model_overreach_point(confs, ys, diffs).mop_difficulty
        overall["mop_difficulty"] = mop

    figs_dir = run_dir / "figures"
    figs_dir.mkdir(exist_ok=True)

    # Plot reliability diagram (overall)
    try:
        import matplotlib.pyplot as plt
        if graded_all:
            centers, accs, counts = calibration_curve(confs, ys, n_bins=10)
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfect")
            mask = counts > 0
            ax.plot(centers[mask], accs[mask], "o-", label="agent")
            ax.set_xlabel("Stated confidence  P̂")
            ax.set_ylabel("Empirical accuracy")
            ax.set_title("Reliability diagram (overall)")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.legend()
            fig.tight_layout()
            fig.savefig(figs_dir / "reliability_overall.png", dpi=150)
            plt.close(fig)
    except Exception as e:
        log.warning("Plotting failed: %s", e)

    # HTML
    html_lines = [
        "<!doctype html><html><head><meta charset=utf-8><title>PROPHET run report</title>",
        "<style>body{font-family:system-ui;margin:20px;max-width:1000px}",
        "table{border-collapse:collapse;margin:1em 0;width:100%}",
        "th,td{border:1px solid #ccc;padding:6px 10px;text-align:right}",
        "th:first-child,td:first-child{text-align:left}",
        "h2{margin-top:2em}</style></head><body>",
        f"<h1>PROPHET — {run_dir.name}</h1>",
    ]
    if overall:
        html_lines.append("<h2>Overall</h2><table>")
        for k, v in overall.items():
            html_lines.append(f"<tr><th>{k}</th><td>{v}</td></tr>")
        html_lines.append("</table>")
    html_lines.append("<h2>Per family</h2><table>")
    if rows:
        cols = list(rows[0].keys())
        html_lines.append("<tr>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>")
        for r in rows:
            html_lines.append("<tr>" + "".join(f"<td>{r.get(c)}</td>" for c in cols) + "</tr>")
    html_lines.append("</table>")
    html_lines.append('<h2>Reliability diagram</h2>')
    html_lines.append('<img src="figures/reliability_overall.png" style="max-width:600px"/>')
    html_lines.append("</body></html>")
    out = run_dir / "report.html"
    out.write_text("\n".join(html_lines))
    log.info("Report written to %s", out)
    return out
