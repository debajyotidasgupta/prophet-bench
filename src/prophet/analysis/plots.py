"""Paper-quality plots: reliability diagrams, Pareto frontiers, per-family heatmaps.

All figures are produced as PNG and PDF (vector). PDF is the source format
for the paper; PNGs are for HTML reports.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np

log = logging.getLogger("prophet.analysis.plots")


def _safe_import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception as e:
        log.warning("matplotlib unavailable: %s", e)
        return None


def reliability_diagram(
    confidences: Sequence[float],
    outcomes: Sequence[int],
    n_bins: int = 10,
    out_path: Path | None = None,
    title: str = "Reliability diagram",
) -> Path | None:
    plt = _safe_import_matplotlib()
    if plt is None:
        return None
    from prophet.engine.scoring import calibration_curve, ece

    centers, accs, counts = calibration_curve(confidences, outcomes, n_bins=n_bins)
    e_val = ece(confidences, outcomes, n_bins=n_bins)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(5, 6), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )
    ax1.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="perfectly calibrated")
    mask = counts > 0
    width = 1.0 / n_bins
    ax1.bar(
        centers[mask],
        accs[mask],
        width=width * 0.9,
        align="center",
        color="#3a7bd5",
        edgecolor="black",
        alpha=0.85,
        label="empirical accuracy",
    )
    ax1.bar(
        centers[mask],
        centers[mask] - accs[mask],
        bottom=accs[mask],
        width=width * 0.9,
        color="#d65a31",
        alpha=0.4,
        label="gap",
    )
    ax1.set_ylabel("Empirical accuracy")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_title(f"{title}\nECE = {e_val:.3f}")
    ax1.legend(loc="upper left", fontsize=8)
    ax2.bar(centers, counts, width=width * 0.9, align="center", color="#6c757d")
    ax2.set_xlabel("Stated confidence P̂")
    ax2.set_ylabel("Count")
    fig.tight_layout()
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    return out_path


def pareto_scatter(
    records: list,
    out_path: Path,
    x: str = "ece",
    y: str = "net_payoff",
    title: str = "PROPHET Pareto frontier",
    annotate: bool = True,
) -> Path | None:
    plt = _safe_import_matplotlib()
    if plt is None:
        return None
    from prophet.analysis.pareto import PointRecord, pareto_front

    # Map external axis names → PointRecord field names.
    # external "net_payoff" is identical to PointRecord.payoff (we keep
    # external `net_payoff` for the headline-table interface).
    field_alias = {"net_payoff": "payoff"}
    fx = field_alias.get(x, x)
    fy = field_alias.get(y, y)
    pts = [
        PointRecord(
            name=r["name"],
            payoff=r.get("net_payoff", r.get("payoff", 0.0)),
            ece=r["ece"],
            brier=r["brier"],
            accuracy=r["accuracy"],
            cost_usd=r.get("cost_usd", 0.0),
        )
        for r in records
    ]
    front = pareto_front(pts, axes=(fx, fy), maximise={"payoff", "accuracy"})
    front_names = {r.name for r in front}
    fig, ax = plt.subplots(figsize=(6, 5))
    for p in pts:
        on_front = p.name in front_names
        ax.scatter(
            getattr(p, fx),
            getattr(p, fy),
            s=70 if on_front else 40,
            c=("#d65a31" if on_front else "#6c757d"),
            edgecolor="black" if on_front else "none",
            zorder=3 if on_front else 2,
            label=None,
        )
        if annotate:
            ax.annotate(p.name, (getattr(p, fx), getattr(p, fy)), fontsize=7, xytext=(4, 4), textcoords="offset points")
    if front:
        front_sorted = sorted(front, key=lambda r: getattr(r, fx))
        ax.plot([getattr(r, fx) for r in front_sorted], [getattr(r, fy) for r in front_sorted], "--", color="#d65a31", alpha=0.7)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    return out_path


def per_family_heatmap(
    matrix: np.ndarray,
    families: list[str],
    agents: list[str],
    out_path: Path,
    title: str = "Per-family metric",
    cmap: str = "RdYlGn",
    vmin: float | None = None,
    vmax: float | None = None,
    fmt: str = "{:.2f}",
) -> Path | None:
    plt = _safe_import_matplotlib()
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(0.6 + 0.7 * len(agents), 0.6 + 0.5 * len(families)))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(agents)))
    ax.set_xticklabels(agents, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(families)))
    ax.set_yticklabels(families, fontsize=8)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, fmt.format(matrix[i, j]), ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    return out_path


def calibration_curve_multi(
    series: dict[str, tuple[Sequence[float], Sequence[int]]],
    out_path: Path,
    n_bins: int = 10,
    title: str = "Reliability — agents",
) -> Path | None:
    plt = _safe_import_matplotlib()
    if plt is None:
        return None
    from prophet.engine.scoring import calibration_curve, ece

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    colors = plt.get_cmap("tab10")
    for i, (name, (confs, ys)) in enumerate(series.items()):
        centers, accs, counts = calibration_curve(confs, ys, n_bins=n_bins)
        mask = counts > 0
        e = ece(confs, ys, n_bins=n_bins)
        ax.plot(centers[mask], accs[mask], "o-", color=colors(i), label=f"{name} (ECE={e:.3f})")
    ax.set_xlabel("Stated confidence P̂")
    ax.set_ylabel("Empirical accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(fontsize=7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    return out_path
