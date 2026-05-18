#!/usr/bin/env python3
"""Aggregate multi-seed leaderboard for Phase B hard-tier data.

Pools per-task outcomes across seeds {42, 7, 1729} by agent name, then
computes accuracy / ECE / Brier / payoff with bootstrap CIs on the pooled
sample (N up to 300 per Pareto-essential agent).

Output: results/openrouter_hard_v2/leaderboard_aggregated.{md,tex,json}
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path, default=Path("results/openrouter_hard_v2/runs"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/openrouter_hard_v2/leaderboard"))
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()

    import sys
    sys.path.insert(0, "src")
    from prophet.engine.scoring import ece, brier_score, log_score, bootstrap_ci

    by_agent: dict[str, list[dict]] = defaultdict(list)
    cost_by_agent: dict[str, float] = defaultdict(float)
    for d in sorted(args.runs_dir.iterdir()):
        if not (d / "outcomes.jsonl").exists():
            continue
        name = d.name.split("-", 2)[-1]
        rows = [json.loads(l) for l in open(d / "outcomes.jsonl") if l.strip()]
        by_agent[name].extend(rows)
        sj = d / "summary.json"
        if sj.exists():
            try:
                cost_by_agent[name] += float(json.loads(sj.read_text()).get("total_cost_usd", 0))
            except Exception:
                pass

    records = []
    for name, outcomes in by_agent.items():
        committed = [r for r in outcomes if r.get("success") is not None]
        if not committed:
            continue
        confs = [r["response"]["confidence"] for r in committed]
        ys = [1 if r["success"] else 0 for r in committed]
        pays = [r.get("payoff_total", 0) or 0 for r in outcomes]
        acc = sum(ys) / len(ys)
        e = ece(confs, ys)
        b = brier_score(confs, ys)
        try:
            ll = log_score(confs, ys)
        except Exception:
            ll = float("nan")
        net = float(sum(pays))
        # Bootstrap CIs
        try:
            acc_mu, acc_lo, acc_hi = bootstrap_ci(np.mean, ys, n_boot=args.n_boot, ci=0.95)
        except Exception:
            acc_lo = acc_hi = acc_mu = acc
        try:
            ece_mu, ece_lo, ece_hi = bootstrap_ci(
                lambda subs: ece([c for c, _ in subs], [yy for _, yy in subs]),
                list(zip(confs, ys)),
                n_boot=args.n_boot, ci=0.95,
            )
        except Exception:
            ece_lo = ece_hi = ece_mu = e
        try:
            pay_mu, pay_lo, pay_hi = bootstrap_ci(np.mean, pays, n_boot=args.n_boot, ci=0.95)
            pay_lo *= len(pays); pay_hi *= len(pays)
        except Exception:
            pay_lo = pay_hi = net
        records.append({
            "name": name,
            "n_outcomes": len(outcomes),
            "n_committed": len(committed),
            "n_seeds": len({r.get("metadata", {}).get("cycle_seed") for r in outcomes if r.get("metadata")}),
            "accuracy": round(acc, 4),
            "accuracy_ci": [round(acc_lo, 4), round(acc_hi, 4)],
            "ece": round(e, 4),
            "ece_ci": [round(ece_lo, 4), round(ece_hi, 4)],
            "brier": round(b, 4),
            "logloss": round(ll, 4) if not np.isnan(ll) else None,
            "net_payoff": round(net, 2),
            "net_payoff_ci": [round(pay_lo, 2), round(pay_hi, 2)],
            "cost_usd": round(cost_by_agent[name], 4),
        })
    records.sort(key=lambda r: -r["net_payoff"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "leaderboard_aggregated.json").write_text(json.dumps(records, indent=2))

    # Markdown
    md = [
        "# PROPHET — multi-seed-aggregated leaderboard (hard tier)",
        "",
        "| Rank | Agent | N | Acc | ECE | Brier | Payoff | Cost USD |",
        "|------|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(records, 1):
        md.append(
            f"| {i} | {r['name']} | {r['n_outcomes']} | "
            f"{r['accuracy']:.3f} ({r['accuracy_ci'][0]:.3f}, {r['accuracy_ci'][1]:.3f}) | "
            f"{r['ece']:.3f} ({r['ece_ci'][0]:.3f}, {r['ece_ci'][1]:.3f}) | "
            f"{r['brier']:.3f} | {r['net_payoff']:.0f} | ${r['cost_usd']:.4f} |"
        )
    (args.out_dir / "leaderboard_aggregated.md").write_text("\n".join(md))
    print(f"Wrote {args.out_dir / 'leaderboard_aggregated.md'}")
    print(f"  {len(records)} agents in aggregated leaderboard")
    for r in records[:10]:
        print(f"  {r['name']:<55} N={r['n_outcomes']:3d}  acc={r['accuracy']:.3f}  ECE={r['ece']:.3f}  payoff={r['net_payoff']:7.0f}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
