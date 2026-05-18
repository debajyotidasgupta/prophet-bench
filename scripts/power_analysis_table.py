#!/usr/bin/env python3
"""EMNLP-2: post-hoc power analysis on the headline pairwise contrasts.

Methods section promises power + MDE per headline contrast; this script
delivers it. For each pair of agents on the K=15 N>=100 panel, compute:

  * realised paired N (intersection of completed task_ids)
  * observed Cohen's d on net_payoff differences
  * minimum-detectable Cohen's d at alpha=0.05, power=0.80, paired t-test
  * post-hoc power at the observed effect size

Writes:
  docs/paper/figures/power_analysis_table.{tex,md,json}
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as sps


def _short(name: str) -> str:
    if name.startswith("baseline:"):
        return name.split(":", 1)[1]
    if name.startswith("openrouter:"):
        return name.split(":", 1)[1].split("/", 1)[-1].replace("_", "-")
    return name


def _mde_paired(n: int, alpha: float = 0.05, power: float = 0.80) -> float:
    """Minimum-detectable Cohen's d for a paired t-test at given (n, alpha, power)."""
    if n < 2:
        return float("inf")
    z_alpha = sps.norm.ppf(1 - alpha / 2)
    z_power = sps.norm.ppf(power)
    return (z_alpha + z_power) / np.sqrt(n)


def _post_hoc_power(d: float, n: int, alpha: float = 0.05) -> float:
    """Post-hoc power for paired t-test given observed Cohen's d and n."""
    if n < 2 or not np.isfinite(d):
        return float("nan")
    z_alpha = sps.norm.ppf(1 - alpha / 2)
    ncp = d * np.sqrt(n)
    return float(1.0 - sps.norm.cdf(z_alpha - ncp) + sps.norm.cdf(-z_alpha - ncp))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path,
                    default=Path("results/openrouter_matrix/runs"))
    ap.add_argument("--min-n", type=int, default=100,
                    help="Restrict to agents with at least min-n committed.")
    ap.add_argument("--out", type=Path,
                    default=Path("docs/paper/figures/power_analysis_table.tex"))
    ap.add_argument("--top-k", type=int, default=12,
                    help="Show top-k contrasts in the printed table.")
    args = ap.parse_args()

    # Pool outcomes per agent
    by_agent: dict[str, dict[str, dict]] = defaultdict(dict)
    cost_n: dict[str, int] = defaultdict(int)
    for d in sorted(args.runs_dir.iterdir()):
        p = d / "outcomes.jsonl"
        if not p.exists():
            continue
        name = d.name.split("-", 2)[-1]
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            o = json.loads(line)
            tid = o["task_id"]
            # Keep first occurrence (multi-seed re-runs already pooled
            # downstream; for power-analysis we deduplicate by task_id)
            by_agent[name].setdefault(tid, o)
            cost_n[name] += 1

    # Filter to agents with N>=min-n committed
    real = {}
    for ag, outs in by_agent.items():
        if ag.startswith("baseline:"):
            continue
        n_committed = sum(1 for o in outs.values() if o.get("success") is not None)
        if n_committed >= args.min_n:
            real[ag] = outs

    agents_sorted = sorted(real.keys())
    print(f"Power analysis on {len(agents_sorted)} agents with N>=committed {args.min_n}")

    # For each pair, compute paired-N, observed d on payoff_total, MDE, power
    pairs = []
    for i, a in enumerate(agents_sorted):
        for b in agents_sorted[i + 1:]:
            tids = set(real[a].keys()) & set(real[b].keys())
            if len(tids) < 5:
                continue
            pay_a, pay_b = [], []
            for tid in tids:
                oa = real[a][tid]
                ob = real[b][tid]
                if oa.get("success") is None or ob.get("success") is None:
                    continue
                pay_a.append(float(oa.get("payoff_total", 0)))
                pay_b.append(float(ob.get("payoff_total", 0)))
            n = len(pay_a)
            if n < 5:
                continue
            diffs = np.array(pay_a) - np.array(pay_b)
            mu_d = float(diffs.mean())
            sd_d = float(diffs.std(ddof=1)) if n > 1 else 0.0
            d_cohen = mu_d / sd_d if sd_d > 1e-12 else 0.0
            mde = _mde_paired(n)
            power = _post_hoc_power(abs(d_cohen), n)
            # Also paired t-stat + p
            try:
                tstat, pval = sps.ttest_rel(pay_a, pay_b)
            except Exception:
                tstat, pval = float("nan"), float("nan")
            pairs.append({
                "a": _short(a), "b": _short(b),
                "n_paired": n,
                "mean_diff_payoff": round(mu_d, 2),
                "cohen_d": round(d_cohen, 3),
                "mde_d_at_n": round(mde, 3),
                "post_hoc_power": round(power, 3),
                "p_value_paired_t": round(pval, 4) if np.isfinite(pval) else None,
                "detectable": bool(abs(d_cohen) >= mde),
            })

    # Sort by absolute Cohen's d descending
    pairs.sort(key=lambda r: -abs(r["cohen_d"]))

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # JSON
    (args.out.with_suffix(".json")).write_text(json.dumps({
        "alpha": 0.05, "power_target": 0.80,
        "min_n_panel": args.min_n,
        "n_agents": len(agents_sorted),
        "n_pairs": len(pairs),
        "pairs": pairs,
    }, indent=2))

    # Markdown
    md = ["# Power analysis: paired contrasts on net_payoff", "",
          "| Pair | N_paired | μ_diff | Cohen's d | MDE_d (α=.05, π=.80) | "
          "post-hoc power | p (paired t) | Detectable? |",
          "|---|---|---|---|---|---|---|---|"]
    for r in pairs[:args.top_k]:
        md.append(f"| {r['a']} vs {r['b']} | {r['n_paired']} | "
                  f"{r['mean_diff_payoff']:.0f} | {r['cohen_d']:+.3f} | "
                  f"{r['mde_d_at_n']:.3f} | {r['post_hoc_power']:.3f} | "
                  f"{r['p_value_paired_t'] if r['p_value_paired_t'] is not None else 'n/a'} | "
                  f"{'✓' if r['detectable'] else '✗'} |")
    (args.out.with_suffix(".md")).write_text("\n".join(md))

    # LaTeX
    tex = [
        r"\begin{table}[t]\centering\small",
        rf"\caption{{Post-hoc power analysis on paired net-payoff contrasts. "
        rf"Cohen's $d$ on the per-task payoff differences; minimum-detectable "
        rf"effect (MDE) computed at $\alpha = 0.05$, power $= 0.80$, paired "
        rf"$t$-test. \emph{{Detectable}} = observed $|d| \ge $ MDE. "
        rf"Top {min(args.top_k, len(pairs))} contrasts shown, sorted by "
        rf"$|d|$.}}",
        r"\label{tab:power-analysis}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Agent A & Agent B & $N_{\text{pair}}$ & $\bar{\Delta}\mathcal{P}$ & "
        r"$d$ & MDE & $1-\beta$ \\",
        r"\midrule",
    ]
    for r in pairs[:args.top_k]:
        tex.append(
            f"{r['a']} & {r['b']} & {r['n_paired']} & "
            f"{r['mean_diff_payoff']:+.0f} & {r['cohen_d']:+.3f} & "
            f"{r['mde_d_at_n']:.3f} & {r['post_hoc_power']:.3f} \\\\"
        )
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    args.out.write_text("\n".join(tex))

    # Print summary
    n_detectable = sum(1 for r in pairs if r["detectable"])
    print(f"Pairs analysed: {len(pairs)}")
    print(f"Pairs with |d| >= MDE (detectable at α=0.05, π=0.80): "
          f"{n_detectable} / {len(pairs)} ({100*n_detectable/max(len(pairs),1):.0f}%)")
    print()
    print("Top 8 contrasts:")
    print(f"{'A':<35} {'B':<35} {'N':>4} {'d':>8} {'MDE':>6} {'power':>6}")
    for r in pairs[:8]:
        print(f"  {r['a']:<33} {r['b']:<33} {r['n_paired']:>4} "
              f"{r['cohen_d']:>+8.3f} {r['mde_d_at_n']:>6.3f} {r['post_hoc_power']:>6.3f}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
