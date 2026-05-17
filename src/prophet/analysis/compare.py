"""Cross-run comparison: paired tests between agents, multiple-comparison correction.

Used after multiple `prophet run` commands. Loads each run's outcomes.jsonl,
aligns by task_id, and produces a comparison table.
"""

from __future__ import annotations

import json
import logging
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from prophet.analysis.stats import (
    AgentSummary,
    bootstrap_ci_brier,
    bootstrap_ci_ece,
    bootstrap_ci_mean,
    compare_two_agents,
    holm_bonferroni,
    permutation_test,
    summarize_agent,
)
from prophet.engine.scoring import brier_score, ece

log = logging.getLogger("prophet.analysis.compare")


def _load_outcomes(run_dir: Path) -> list[dict]:
    p = run_dir / "outcomes.jsonl"
    if not p.exists():
        raise FileNotFoundError(p)
    out: list[dict] = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


@dataclass(slots=True)
class AgentRun:
    name: str
    run_dir: Path
    outcomes: list[dict] = field(default_factory=list)
    by_task: dict[str, dict] = field(default_factory=dict)

    @classmethod
    def from_dir(cls, run_dir: Path, name: str | None = None) -> "AgentRun":
        outs = _load_outcomes(run_dir)
        name = name or (run_dir.name)
        return cls(
            name=name,
            run_dir=run_dir,
            outcomes=outs,
            by_task={o["task_id"]: o for o in outs},
        )


def aligned_pairs(a: AgentRun, b: AgentRun) -> list[tuple[dict, dict]]:
    """Return list of (outcome_a, outcome_b) on tasks common to both runs."""
    common = set(a.by_task) & set(b.by_task)
    return [(a.by_task[t], b.by_task[t]) for t in sorted(common)]


def per_task_payoffs_correct(outs: list[dict]) -> tuple[list[float], list[int], list[float]]:
    pay = [float(o["payoff_total"]) for o in outs]
    # 0 if not graded (PASS)
    corr = [1 if o.get("success") else 0 for o in outs]
    conf = [float(o["response"]["confidence"]) for o in outs]
    return pay, corr, conf


def headline_table(
    runs: list[AgentRun], n_boot: int = 5000, ci: float = 0.95, seed: int = 0
) -> list[AgentSummary]:
    out: list[AgentSummary] = []
    for r in runs:
        graded = [o for o in r.outcomes if o["success"] is not None]
        if not graded:
            continue
        c = [float(o["response"]["confidence"]) for o in graded]
        y = [1 if o["success"] else 0 for o in graded]
        pay = [float(o["payoff_total"]) for o in r.outcomes]
        out.append(summarize_agent(c, y, pay, name=r.name, n_boot=n_boot, ci=ci, seed=seed))
    return out


def pairwise_significance(
    runs: list[AgentRun], correction: str = "holm", alpha: float = 0.05, seed: int = 0
) -> dict[tuple[str, str], dict[str, float]]:
    """All pairwise comparisons; returns {(A,B): {metric: p, ...}, ...}.

    Holm-Bonferroni correction applied across all pairs × metrics.
    """
    pairs: list[tuple[str, str, dict[str, float], dict[str, float]]] = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            a, b = runs[i], runs[j]
            pairs_ij = aligned_pairs(a, b)
            if not pairs_ij:
                continue
            outs_a = [p[0] for p in pairs_ij]
            outs_b = [p[1] for p in pairs_ij]
            pay_a, cor_a, conf_a = per_task_payoffs_correct(outs_a)
            pay_b, cor_b, conf_b = per_task_payoffs_correct(outs_b)
            tests = compare_two_agents(pay_a, pay_b, cor_a, cor_b)
            # Permutation test for ECE difference (use only graded outcomes)
            graded_idx = [
                k for k, (oa, ob) in enumerate(zip(outs_a, outs_b))
                if oa.get("success") is not None and ob.get("success") is not None
            ]
            if graded_idx:
                ca = np.array([conf_a[k] for k in graded_idx])
                cb = np.array([conf_b[k] for k in graded_idx])
                ya = np.array([cor_a[k] for k in graded_idx])
                yb = np.array([cor_b[k] for k in graded_idx])
                perm = permutation_test(
                    a_metric_fn=lambda c, o: ece(c.tolist(), o.tolist()),
                    b_metric_fn=lambda c, o: ece(c.tolist(), o.tolist()),
                    confidences_a=ca,
                    outcomes_a=ya,
                    confidences_b=cb,
                    outcomes_b=yb,
                    n_perm=2000,
                    seed=seed,
                )
                pvals = {
                    "payoff_t": tests.get("payoff_paired_t").pvalue if "payoff_paired_t" in tests else None,
                    "payoff_wilcoxon": tests.get("payoff_wilcoxon").pvalue if "payoff_wilcoxon" in tests else None,
                    "accuracy_mcnemar": tests.get("accuracy_mcnemar").pvalue if "accuracy_mcnemar" in tests else None,
                    "ece_permutation": perm.pvalue,
                }
            else:
                pvals = {
                    "payoff_t": tests.get("payoff_paired_t").pvalue if "payoff_paired_t" in tests else None,
                    "payoff_wilcoxon": tests.get("payoff_wilcoxon").pvalue if "payoff_wilcoxon" in tests else None,
                    "accuracy_mcnemar": tests.get("accuracy_mcnemar").pvalue if "accuracy_mcnemar" in tests else None,
                    "ece_permutation": None,
                }
            pairs.append((a.name, b.name, pvals, {}))
    # Multiple-comparison adjustment: collect all p-values, adjust as one family
    flat_keys: list[tuple[int, str]] = []
    flat_ps: list[float] = []
    for k, (_, _, pvals, _) in enumerate(pairs):
        for metric, p in pvals.items():
            if p is None:
                continue
            flat_keys.append((k, metric))
            flat_ps.append(p)
    if flat_ps:
        if correction == "holm":
            decisions = holm_bonferroni(flat_ps, alpha=alpha)
        else:
            from prophet.analysis.stats import benjamini_hochberg
            decisions = benjamini_hochberg(flat_ps, q=alpha)
    else:
        decisions = []
    adjusted = defaultdict(dict)
    for (k, metric), p, decide in zip(flat_keys, flat_ps, decisions):
        adjusted[(pairs[k][0], pairs[k][1])][metric] = {
            "p_raw": p,
            "reject_at_alpha": decide,
        }
    return dict(adjusted)


def write_comparison_report(
    runs: list[AgentRun],
    out_dir: Path,
    seed: int = 0,
    ci: float = 0.95,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    table = headline_table(runs, ci=ci, seed=seed)
    pairs = pairwise_significance(runs, seed=seed)
    out: dict = {
        "headline": [
            {
                "agent": s.name,
                "n": s.n,
                "accuracy": s.accuracy,
                "accuracy_ci": list(s.accuracy_ci),
                "ece": s.ece,
                "ece_ci": list(s.ece_ci),
                "brier": s.brier,
                "brier_ci": list(s.brier_ci),
                "logloss": s.logloss,
                "logloss_ci": list(s.logloss_ci),
                "net_payoff": s.net_payoff,
                "net_payoff_ci": list(s.net_payoff_ci),
            }
            for s in table
        ],
        "pairwise": [
            {"a": a, "b": b, "tests": tests}
            for (a, b), tests in pairs.items()
        ],
    }
    p = out_dir / "comparison.json"
    p.write_text(json.dumps(out, indent=2, default=float))
    return p
