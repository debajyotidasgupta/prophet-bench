"""Cross-family calibration transfer analysis.

H3 in the paper: "Does a model that is well-calibrated on family A remain
well-calibrated on family B?"

We compute, for each agent, the rank correlation (Spearman ρ) between its
per-family ECE rankings across families. Low rho ⇒ calibration is
family-specific; high rho ⇒ calibration is a global trait of the agent.

We also compute *per-difficulty* calibration curves to expose the MOP
phenomenon — at what difficulty band does each agent break.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as sps

from prophet.engine.scoring import brier_score, ece, log_score, model_overreach_point

log = logging.getLogger("prophet.analysis.transfer")


def per_family_ece(outcomes: list[dict]) -> dict[str, float]:
    by_family = defaultdict(list)
    for o in outcomes:
        if o.get("success") is None:
            continue
        by_family[o["family"]].append(o)
    return {
        fam: ece(
            [float(r["response"]["confidence"]) for r in rows],
            [1 if r["success"] else 0 for r in rows],
        )
        for fam, rows in by_family.items()
        if len(rows) >= 10
    }


def cross_family_correlation(
    runs: dict[str, list[dict]],
    metric: str = "ece",
) -> dict[str, float]:
    """Per-agent: rank correlation of family rankings against the panel-average ranking.

    runs: {agent_name: list-of-outcomes}
    Returns {agent_name: spearman_rho}.
    """
    fns = {"ece": per_family_ece, "brier": per_family_brier}
    fn = fns.get(metric, per_family_ece)
    # Compute per-(agent, family) metric
    rows = {}
    families = set()
    for ag, outs in runs.items():
        rows[ag] = fn(outs)
        families.update(rows[ag].keys())
    families = sorted(families)
    # Panel-mean ranks per family
    mat = np.full((len(rows), len(families)), np.nan, dtype=float)
    agents = list(rows.keys())
    for i, ag in enumerate(agents):
        for j, fam in enumerate(families):
            if fam in rows[ag]:
                mat[i, j] = rows[ag][fam]
    panel_mean = np.nanmean(mat, axis=0)
    panel_rank = sps.rankdata(panel_mean)
    out: dict[str, float] = {}
    for i, ag in enumerate(agents):
        row = mat[i]
        mask = np.isfinite(row)
        if mask.sum() < 3:
            out[ag] = float("nan")
            continue
        rho, _ = sps.spearmanr(row[mask], panel_rank[mask])
        out[ag] = float(rho)
    return out


def per_family_brier(outcomes: list[dict]) -> dict[str, float]:
    by_family = defaultdict(list)
    for o in outcomes:
        if o.get("success") is None:
            continue
        by_family[o["family"]].append(o)
    return {
        fam: brier_score(
            [float(r["response"]["confidence"]) for r in rows],
            [1 if r["success"] else 0 for r in rows],
        )
        for fam, rows in by_family.items()
        if len(rows) >= 10
    }


def difficulty_bin_curves(outcomes: list[dict], n_bins: int = 8) -> dict:
    """Per-difficulty-bin (accuracy, mean(P̂), Brier, log-loss).

    Used to plot 'capability vs claimed-confidence' curves separating MOP
    from over-attempt.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2.0
    acc = np.full(n_bins, np.nan)
    pbar = np.full(n_bins, np.nan)
    n = np.zeros(n_bins, dtype=int)
    confs_all, ys_all = [], []
    for o in outcomes:
        if o.get("success") is None:
            continue
        d = float(o["difficulty"])
        confs_all.append(float(o["response"]["confidence"]))
        ys_all.append(1 if o["success"] else 0)
        i = min(int(d * n_bins), n_bins - 1)
        n[i] += 1
    confs_arr = np.array(confs_all)
    ys_arr = np.array(ys_all)
    diffs = np.array([float(o["difficulty"]) for o in outcomes if o.get("success") is not None])
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (diffs >= lo) & (diffs < hi if i < n_bins - 1 else diffs <= hi)
        if m.sum() == 0:
            continue
        acc[i] = float(ys_arr[m].mean())
        pbar[i] = float(confs_arr[m].mean())
    return {
        "centers": centers.tolist(),
        "accuracy": acc.tolist(),
        "claimed_confidence": pbar.tolist(),
        "n_per_bin": n.tolist(),
        "ece_overall": float(ece(confs_arr.tolist(), ys_arr.tolist())) if len(confs_arr) else None,
    }


def write_transfer_report(runs_dir: Path, out_dir: Path | None = None) -> Path:
    runs_dir = Path(runs_dir)
    out_dir = (out_dir or runs_dir.parent / "transfer").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = {}
    for child in sorted(runs_dir.iterdir()):
        if not child.is_dir():
            continue
        outs_p = child / "outcomes.jsonl"
        if not outs_p.exists():
            continue
        outs = [json.loads(line) for line in outs_p.read_text().splitlines() if line.strip()]
        name = child.name.split("-", 2)[-1] if "-" in child.name else child.name
        runs[name] = outs
    summary = {
        "cross_family_ece_correlation": cross_family_correlation(runs, "ece"),
        "cross_family_brier_correlation": cross_family_correlation(runs, "brier"),
        "per_agent_difficulty_curves": {
            ag: difficulty_bin_curves(outs) for ag, outs in runs.items()
        },
    }
    (out_dir / "transfer.json").write_text(json.dumps(summary, indent=2, default=float))
    log.info("transfer report → %s", out_dir / "transfer.json")
    return out_dir / "transfer.json"
