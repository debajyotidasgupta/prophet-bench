#!/usr/bin/env python3
"""Push the PROPHET seed dataset + paper results to Hugging Face.

Pushes two repos:
  - debajyotidasgupta/prophet-bench (seeds + per-family task lists)
  - debajyotidasgupta/prophet-results (anonymised per-task outcomes)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

log = logging.getLogger("prophet.release")


def push_seeds(out_dir: Path, repo_id: str, private: bool) -> None:
    try:
        from huggingface_hub import HfApi
    except Exception as e:
        log.error("huggingface_hub missing: %s", e)
        return
    api = HfApi(token=os.environ.get("HUGGINGFACE_TOKEN"))
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(out_dir),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Initial PROPHET seed release",
    )
    log.info("pushed seeds → https://huggingface.co/datasets/%s", repo_id)


def push_results(runs_dir: Path, repo_id: str, private: bool) -> None:
    try:
        from huggingface_hub import HfApi
    except Exception as e:
        log.error("huggingface_hub missing: %s", e)
        return
    api = HfApi(token=os.environ.get("HUGGINGFACE_TOKEN"))
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    # We only push summary.json + outcomes.jsonl per run, plus the leaderboard.
    import tempfile, shutil
    with tempfile.TemporaryDirectory() as tmp:
        tdir = Path(tmp) / "runs"
        tdir.mkdir(parents=True)
        for sub in runs_dir.iterdir():
            if not sub.is_dir():
                continue
            if not (sub / "outcomes.jsonl").exists():
                continue
            dst = tdir / sub.name
            dst.mkdir(parents=True)
            shutil.copy(sub / "outcomes.jsonl", dst / "outcomes.jsonl")
            if (sub / "summary.json").exists():
                shutil.copy(sub / "summary.json", dst / "summary.json")
        # Also include leaderboard
        leaderboard = runs_dir.parent / "leaderboard"
        if leaderboard.exists():
            shutil.copytree(leaderboard, Path(tmp) / "leaderboard")
        api.upload_folder(
            folder_path=tmp,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="PROPHET v1 results",
        )
        log.info("pushed results → https://huggingface.co/datasets/%s", repo_id)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds-dir", type=Path, default=Path("data/hf_dataset"))
    ap.add_argument("--runs-dir", type=Path, default=Path("results/openrouter_matrix/runs"))
    ap.add_argument(
        "--seeds-repo",
        default=os.environ.get("PROPHET_HF_DATASET", "debajyotidasgupta/prophet-bench"),
    )
    ap.add_argument(
        "--results-repo",
        default=os.environ.get("PROPHET_HF_RESULTS", "debajyotidasgupta/prophet-results"),
    )
    ap.add_argument("--private", action="store_true", default=True)
    ap.add_argument("--public", dest="private", action="store_false")
    ap.add_argument("--only", choices=["seeds", "results", "both"], default="both")
    args = ap.parse_args()
    from dotenv import load_dotenv

    load_dotenv()

    if args.only in {"seeds", "both"} and args.seeds_dir.exists():
        push_seeds(args.seeds_dir, args.seeds_repo, args.private)
    if args.only in {"results", "both"} and args.runs_dir.exists():
        push_results(args.runs_dir, args.results_repo, args.private)
    log.info("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
