#!/usr/bin/env python3
"""Run the full PROPHET experimental matrix for the paper.

This orchestrator:
  * Iterates the YAML config in `configs/baseline_runs.yaml` (or a custom path).
  * For each agent, calls `prophet run` with the right family / N / concurrency / cost cap.
  * Skips already-completed runs (idempotent across crashes).
  * Tracks cumulative cost and aborts if it exceeds the global budget.
  * After all runs, invokes `analyze-compare` for a single comparison report.

Usage:
  python scripts/experimental_matrix.py [--config configs/baseline_runs.yaml] \\
      [--budget 50] [--out results/paper] [--skip-baselines] [--skip-closed] \\
      [--skip-open] [--only-agents openai:gpt-5,anthropic:claude-opus-4-7]

Always wrap with `--budget` to hard-cap spending. The script *also* respects
each agent's `cost_cap` (per-agent caps from the config).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

log = logging.getLogger("prophet.matrix")


def _load_cfg(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _summary_path(out_dir: Path, agent_uri: str) -> Path:
    safe = agent_uri.replace("/", "_").replace(":", "_")
    return out_dir / "_state" / f"{safe}.done"


def _run_one(
    agent_uri: str,
    common: dict,
    cost_cap: float,
    concurrency: int,
    per_sec_limit: float,
    out_dir: Path,
    extra_args: list[str],
) -> tuple[int, float]:
    cmd = [
        sys.executable,
        "-m",
        "prophet.cli",
        "run",
        "--agent",
        agent_uri,
        "--families",
        common.get("families", "all"),
        "--n",
        str(common.get("n_per_family", 100)),
        "--seed",
        str(common.get("seed", 42)),
        "--max-cost",
        str(cost_cap),
        "--out-dir",
        str(out_dir / "runs"),
        "--concurrency",
        str(concurrency),
        "--per-sec-limit",
        str(per_sec_limit),
    ] + extra_args
    log.info("→ %s", " ".join(cmd))
    t0 = time.time()
    res = subprocess.run(cmd, env=os.environ.copy())
    return res.returncode, time.time() - t0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/baseline_runs.yaml"))
    ap.add_argument("--budget", type=float, default=50.0)
    ap.add_argument("--out", type=Path, default=Path("results/paper"))
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--skip-closed", action="store_true")
    ap.add_argument("--skip-open", action="store_true")
    ap.add_argument("--only-agents", default="")
    args = ap.parse_args()
    load_dotenv()

    cfg = _load_cfg(args.config)
    common = cfg.get("common", {})
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "_state").mkdir(exist_ok=True)
    cumulative = 0.0

    only = set(s.strip() for s in args.only_agents.split(",") if s.strip())

    groups = []
    if not args.skip_baselines:
        for entry in cfg.get("baselines", []):
            groups.append(("baseline", entry))
    if not args.skip_closed:
        for entry in cfg.get("closed_apis", []):
            groups.append(("closed", entry))
    if not args.skip_open:
        for entry in cfg.get("open_apis", []):
            groups.append(("open", entry))

    for kind, entry in groups:
        uri = entry["uri"]
        if only and uri not in only:
            continue
        cap = float(entry.get("cost_cap", 0.0))
        conc = int(entry.get("concurrency", 1))
        psl = float(entry.get("per_sec_limit", 0.0))
        # Hard global budget check
        if cumulative + cap > args.budget:
            log.warning(
                "Skipping %s — cap $%.2f would exceed global budget $%.2f (cumulative $%.2f)",
                uri,
                cap,
                args.budget,
                cumulative,
            )
            continue
        done = _summary_path(args.out, uri)
        if done.exists():
            log.info("already done: %s", uri)
            continue
        rc, elapsed = _run_one(uri, common, cap, conc, psl, args.out, [])
        if rc == 0:
            done.write_text(json.dumps({"uri": uri, "elapsed_s": elapsed}))
            cumulative += cap
            log.info("✓ %s in %.1fs (cumulative $%.2f / $%.2f)", uri, elapsed, cumulative, args.budget)
        else:
            log.error("✗ %s rc=%d", uri, rc)

    # Final analysis
    log.info("running analyze-compare")
    subprocess.run(
        [sys.executable, "-m", "prophet.cli", "analyze-compare", str(args.out / "runs")],
        check=False,
    )
    log.info("matrix done. total cumulative budget consumed (upper bound) ≈ $%.2f", cumulative)
    return 0


if __name__ == "__main__":
    sys.exit(main())
