#!/usr/bin/env python3
"""Calibrate per-task difficulty from a reference panel of agents.

Usage:
  python scripts/calibrate_difficulty.py --families all --n 200 --seed 42 \\
      --agents baseline:oracle-noisy,openai:gpt-5-mini,anthropic:claude-haiku-4-5 \\
      --out data/reference

Each family produces a JSON file at data/reference/<family>.json with:
  { "task_id": {"empirical_difficulty": 0.3, "n_panel": 3, "panel_acc": [...]} }

We then publish a *difficulty calibration table* alongside the static
difficulty tiers — both are used by the orchestrator when forming the
market offer (currently the engine reads the static tier; future versions
can fall back to empirical difficulty when available).

Cost: this is a one-time per-family operation. With cheap models like
gpt-5-mini and claude-haiku-4-5 the full panel is ~$1-3 for 200 tasks
across 12 families.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

log = logging.getLogger("prophet.calibrate")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="all")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--agents",
        default="baseline:oracle-noisy",
        help="Comma-separated agent URIs",
    )
    ap.add_argument("--out", type=Path, default=Path("data/reference"))
    ap.add_argument("--max-cost-per-agent", type=float, default=3.0)
    args = ap.parse_args()
    load_dotenv()
    args.out.mkdir(parents=True, exist_ok=True)

    from prophet.agents import build_agent
    from prophet.engine.market import MarketMaker
    from prophet.engine.orchestrator import Orchestrator, RunConfig
    from prophet.families import get_family, list_families

    family_names = list_families() if args.families == "all" else [s.strip() for s in args.families.split(",")]
    agent_uris = [s.strip() for s in args.agents.split(",") if s.strip()]
    log.info("Calibrating %d families × %d agents × %d tasks", len(family_names), len(agent_uris), args.n)

    by_family: dict[str, dict] = defaultdict(dict)
    for fname in family_names:
        fam = get_family(fname)
        tasks = fam.generate(n=args.n, seed=args.seed)
        for t in tasks:
            by_family[fname][t.task_id] = {
                "static_difficulty": t.difficulty,
                "panel_correct": [],
                "panel_agents": [],
            }

    market = MarketMaker(market_seed=args.seed)
    orch = Orchestrator(market=market)

    for uri in agent_uris:
        try:
            ag = build_agent(uri)
        except Exception as e:
            log.error("Skipping agent %s: %s", uri, e)
            continue
        for fname in family_names:
            fam = get_family(fname)
            tasks = fam.generate(n=args.n, seed=args.seed)
            cfg = RunConfig(
                cycle_seed=args.seed,
                market_seed=args.seed,
                out_dir=args.out / "_calibrate_runs",
                max_cost_usd=args.max_cost_per_agent,
                progress=False,
            )
            res = orch.run(ag, tasks, cfg)
            for o in res.outcomes:
                rec = by_family[fname][o.task_id]
                rec["panel_correct"].append(1 if o.success else 0)
                rec["panel_agents"].append(uri)
            log.info("%s × %s — done", uri, fname)

    # Aggregate empirical difficulty
    for fname, rows in by_family.items():
        for tid, rec in rows.items():
            if rec["panel_correct"]:
                acc = sum(rec["panel_correct"]) / len(rec["panel_correct"])
                rec["empirical_difficulty"] = float(1.0 - acc)
                rec["n_panel"] = len(rec["panel_correct"])
            else:
                rec["empirical_difficulty"] = None
                rec["n_panel"] = 0
        out_p = args.out / f"{fname}.json"
        out_p.write_text(json.dumps(rows, indent=2))
        log.info("wrote %s", out_p)

    log.info("calibration done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
