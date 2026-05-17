#!/usr/bin/env python3
"""Upload a results directory to HF + W&B (best-effort)."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

log = logging.getLogger("prophet.upload")


def upload_to_hf(dir_path: Path, repo_id: str) -> None:
    try:
        from huggingface_hub import HfApi
    except Exception as e:
        log.warning("huggingface_hub missing: %s", e)
        return
    api = HfApi(token=os.environ.get("HUGGINGFACE_TOKEN"))
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
    except Exception as e:
        log.warning("hf create_repo: %s", e)
    api.upload_folder(
        folder_path=str(dir_path),
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo=f"runs/{dir_path.name}",
        commit_message=f"Upload results {dir_path.name}",
    )
    log.info("uploaded to %s", repo_id)


def upload_to_wandb(dir_path: Path, project: str | None = None) -> None:
    try:
        import wandb
    except Exception as e:
        log.warning("wandb missing: %s", e)
        return
    project = project or os.environ.get("WANDB_PROJECT", "prophet")
    run = wandb.init(project=project, job_type="upload", reinit=True)
    artifact = wandb.Artifact(name=f"results-{dir_path.name}", type="prophet-run")
    artifact.add_dir(str(dir_path))
    run.log_artifact(artifact)
    run.finish()


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, type=Path)
    ap.add_argument(
        "--hf-repo",
        default=os.environ.get("PROPHET_HF_REPO", "debajyotidasgupta/prophet-results"),
    )
    ap.add_argument(
        "--wandb-project", default=os.environ.get("WANDB_PROJECT", "prophet")
    )
    args = ap.parse_args()
    d = args.dir.resolve()
    if not d.exists():
        log.error("dir %s missing", d)
        return 2
    upload_to_hf(d, args.hf_repo)
    upload_to_wandb(d, args.wandb_project)
    return 0


if __name__ == "__main__":
    sys.exit(main())
