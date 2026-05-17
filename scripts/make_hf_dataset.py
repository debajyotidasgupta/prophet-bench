#!/usr/bin/env python3
"""Package PROPHET seed lists + reference panel + a sample of generated tasks
as a Hugging Face dataset.

This is what reviewers / users will load via:
  load_dataset("debajyotidasgupta/prophet-bench", split="math")

Schema per family:
  - task_id (str)
  - family (str)
  - difficulty (float)
  - prompt (str)
  - reference_answer (str | null)
  - estimated_seconds (float)
  - generator (str)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

log = logging.getLogger("prophet.hf_dataset")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="Tasks per family.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("data/hf_dataset"))
    ap.add_argument("--push", action="store_true", help="Push to HF hub.")
    ap.add_argument(
        "--repo-id",
        default=os.environ.get("PROPHET_HF_DATASET", "debajyotidasgupta/prophet-bench"),
    )
    args = ap.parse_args()
    load_dotenv()
    args.out.mkdir(parents=True, exist_ok=True)

    from prophet.families import get_family, list_families

    all_records = {}
    for fname in list_families():
        fam = get_family(fname)
        tasks = fam.generate(n=args.n, seed=args.seed)
        recs = []
        for t in tasks:
            recs.append(
                {
                    "task_id": t.task_id,
                    "family": t.family,
                    "difficulty": float(t.difficulty),
                    "prompt": t.prompt,
                    "reference_answer": t.reference_answer,
                    "estimated_seconds": float(t.estimated_seconds),
                    "metadata": {k: v for k, v in (t.metadata or {}).items() if isinstance(v, (str, int, float, bool))},
                }
            )
        all_records[fname] = recs
        (args.out / f"{fname}.jsonl").write_text("\n".join(json.dumps(r) for r in recs))
        log.info("wrote %s (%d records)", args.out / f"{fname}.jsonl", len(recs))

    # Croissant metadata for D&B compliance
    croissant = {
        "@context": "https://schema.org/",
        "@type": "Dataset",
        "name": "PROPHET — calibration benchmark seeds",
        "description": (
            "Procedurally generated tasks for the PROPHET calibration benchmark. "
            "12 families spanning code, math, knowledge, reasoning, writing, "
            "browser, tools, multimodal, data, scientific, multilingual, safety. "
            "Each split contains N=200 reference instances at seed=42."
        ),
        "license": "https://opensource.org/licenses/MIT",
        "url": "https://github.com/debajyotidasgupta/prophet-bench",
        "version": "0.1.0",
        "creator": "PROPHET Authors",
        "datePublished": "2026-05-17",
        "keywords": ["llm", "calibration", "benchmark", "agents", "proper-scoring-rule"],
        "splits": list(all_records.keys()),
    }
    (args.out / "metadata.json").write_text(json.dumps(croissant, indent=2))
    log.info("wrote %s", args.out / "metadata.json")

    if args.push:
        try:
            from huggingface_hub import HfApi
        except Exception as e:
            log.error("huggingface_hub missing: %s", e)
            return 2
        api = HfApi(token=os.environ.get("HUGGINGFACE_TOKEN"))
        api.create_repo(args.repo_id, repo_type="dataset", private=True, exist_ok=True)
        api.upload_folder(
            folder_path=str(args.out),
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message="Initial PROPHET seed release",
        )
        log.info("pushed to https://huggingface.co/datasets/%s", args.repo_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
