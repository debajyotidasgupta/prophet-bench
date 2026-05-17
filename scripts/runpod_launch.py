#!/usr/bin/env python3
"""RunPod GPU launcher for PROPHET — short-lived pod with hard cost cap.

Strategy:
  * Spin up a single A6000 (or A100) pod.
  * Run a self-contained launcher script inside the pod that:
      - pulls the PROPHET repo from GitHub (using GITHUB_TOKEN)
      - installs requirements
      - runs vLLM server with the target open-source model
      - executes the chosen `prophet run` invocation
      - uploads results to W&B + HF (optional)
      - exits, terminating the pod
  * Monitors pod lifecycle from the host. Hard-kills if > MAX_MINUTES.

Usage:
  python scripts/runpod_launch.py --model Qwen/Qwen3-7B-Instruct --families all \\
      --n 100 --max-minutes 45 --gpu A6000

Always set --max-minutes; the script auto-terminates the pod when it runs out.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import requests

log = logging.getLogger("runpod_launch")

RUNPOD_API = "https://api.runpod.io/graphql"
RUNPOD_GPU_TYPES = {
    "A6000": "NVIDIA RTX A6000",
    "A6000Ada": "NVIDIA RTX 6000 Ada Generation",
    "A5000": "NVIDIA RTX A5000",
    "L40S": "NVIDIA L40S",
    "A100-40": "NVIDIA A100 80GB PCIe",
    "A100-80": "NVIDIA A100 80GB PCIe",
    "H100-PCIe": "NVIDIA H100 PCIe",
    "H100-SXM": "NVIDIA H100 80GB HBM3",
}


def _api_key() -> str:
    k = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not k:
        raise SystemExit("RUNPOD_API_KEY is not set in the environment (.env)")
    return k


def _gql(query: str, variables: dict | None = None) -> dict:
    resp = requests.post(
        RUNPOD_API,
        headers={"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json"},
        json={"query": query, "variables": variables or {}},
        timeout=60,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"runpod api {resp.status_code}: {resp.text[:500]}")
    out = resp.json()
    if "errors" in out:
        raise RuntimeError(f"runpod api errors: {out['errors']}")
    return out["data"]


def find_gpu_type_id(gpu: str) -> str:
    # Use REST gpuTypes endpoint for stable lookup; if absent fall back to default.
    # RunPod's GraphQL gpuTypes query is reliable enough.
    q = """
    query GpuTypes {
      gpuTypes {
        id
        displayName
        memoryInGb
        secureCloud
        communityCloud
      }
    }
    """
    types = _gql(q)["gpuTypes"]
    display = RUNPOD_GPU_TYPES.get(gpu, gpu)
    for t in types:
        if t["displayName"] == display:
            return t["id"]
    raise SystemExit(f"GPU {gpu!r} (display={display!r}) not available right now.")


def launch_pod(
    name: str,
    gpu_type_id: str,
    image: str,
    env: dict[str, str],
    docker_args: str,
    disk_gb: int = 60,
) -> str:
    env_pairs = [{"key": k, "value": v} for k, v in env.items()]
    q = """
    mutation PodFindAndDeployOnDemand($input: PodFindAndDeployOnDemandInput!) {
      podFindAndDeployOnDemand(input: $input) {
        id
        imageName
        env
        machineId
        machine { podHostId }
      }
    }
    """
    variables = {
        "input": {
            "cloudType": "ALL",
            "gpuCount": 1,
            "volumeInGb": disk_gb,
            "containerDiskInGb": disk_gb,
            "minVcpuCount": 8,
            "minMemoryInGb": 32,
            "gpuTypeId": gpu_type_id,
            "name": name,
            "imageName": image,
            "dockerArgs": docker_args,
            "ports": "8888/http,22/tcp,8000/http",
            "volumeMountPath": "/workspace",
            "env": env_pairs,
        }
    }
    res = _gql(q, variables)["podFindAndDeployOnDemand"]
    pid = res["id"]
    log.info("launched pod %s", pid)
    return pid


def pod_status(pod_id: str) -> dict:
    q = """
    query Pod($input: PodFilter!) {
      pod(input: $input) {
        id
        desiredStatus
        lastStatusChange
        runtime {
          ports { isIpPublic ip privatePort publicPort type }
          uptimeInSeconds
        }
        costPerHr
      }
    }
    """
    return _gql(q, {"input": {"podId": pod_id}})["pod"]


def terminate_pod(pod_id: str) -> None:
    q = """
    mutation Terminate($input: PodTerminateInput!) {
      podTerminate(input: $input)
    }
    """
    _gql(q, {"input": {"podId": pod_id}})
    log.info("terminated pod %s", pod_id)


def _bootstrap_script(
    repo_url: str,
    branch: str,
    model: str,
    families: str,
    n: int,
    max_cost: float,
    extra_run_args: str,
) -> str:
    return f"""#!/usr/bin/env bash
set -euxo pipefail
apt-get update -q && apt-get install -yq git curl jq tini
cd /workspace
git clone --depth 1 --branch {branch} {repo_url} prophet || (cd prophet && git pull)
cd prophet
pip install -e ".[api,vllm]"
# Start vLLM in background
nohup python -m vllm.entrypoints.openai.api_server \\
    --model {model} \\
    --port 8000 \\
    --gpu-memory-utilization 0.85 \\
    --max-model-len 8192 \\
    --dtype auto \\
    --trust-remote-code &
VLLM_PID=$!

# Wait for vLLM ready
for i in $(seq 1 180); do
    if curl -sf http://localhost:8000/v1/models >/dev/null; then
        echo "vLLM up"; break
    fi
    sleep 2
done

# Run benchmark
mkdir -p /workspace/results
VLLM_API_KEY=EMPTY prophet run --agent vllm:{model} --families {families} --n {n} --max-cost {max_cost} {extra_run_args} --out-dir /workspace/results/runs

# Upload (best-effort)
python scripts/upload_results.py --dir /workspace/results || true

kill $VLLM_PID || true
"""


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Hugging Face model id, e.g. Qwen/Qwen3-7B-Instruct")
    ap.add_argument("--families", default="all")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--max-cost", type=float, default=20.0, help="PROPHET internal cost cap")
    ap.add_argument("--max-minutes", type=int, required=True, help="Hard wall-clock kill switch in minutes.")
    ap.add_argument("--gpu", default="A6000", choices=list(RUNPOD_GPU_TYPES.keys()))
    ap.add_argument("--image", default="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04")
    ap.add_argument("--branch", default="main")
    ap.add_argument("--extra-run-args", default="")
    ap.add_argument("--repo-url", default=None)
    args = ap.parse_args()

    repo_url = args.repo_url or os.environ.get(
        "PROPHET_REPO_URL", "https://github.com/debajyotidasgupta/prophet-bench.git"
    )
    gh_token = os.environ.get("GITHUB_TOKEN", "")
    if gh_token and "github.com" in repo_url and "@" not in repo_url:
        repo_url = repo_url.replace("https://", f"https://{gh_token}@", 1)

    bootstrap = _bootstrap_script(
        repo_url=repo_url,
        branch=args.branch,
        model=args.model,
        families=args.families,
        n=args.n,
        max_cost=args.max_cost,
        extra_run_args=args.extra_run_args,
    )

    # Encode the bootstrap as a startup command. We base64-encode the script
    # so the docker args stay quote-safe.
    import base64

    b64 = base64.b64encode(bootstrap.encode()).decode()
    docker_args = (
        "bash -lc 'echo "
        + b64
        + " | base64 -d > /workspace/bootstrap.sh && bash /workspace/bootstrap.sh 2>&1 | tee /workspace/bootstrap.log'"
    )

    env = {
        "HUGGINGFACE_TOKEN": os.environ.get("HUGGINGFACE_TOKEN", ""),
        "HF_TOKEN": os.environ.get("HUGGINGFACE_TOKEN", ""),
        "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
        "WANDB_PROJECT": os.environ.get("WANDB_PROJECT", "prophet"),
        "GITHUB_TOKEN": gh_token,
        "PROPHET_MAX_RUN_COST_USD": str(args.max_cost),
    }
    gpu_type_id = find_gpu_type_id(args.gpu)
    name = f"prophet-{int(time.time())}-{args.model.split('/')[-1][:16]}"
    pod_id = launch_pod(
        name=name,
        gpu_type_id=gpu_type_id,
        image=args.image,
        env=env,
        docker_args=docker_args,
        disk_gb=80,
    )
    log.info("pod %s launched — monitoring", pod_id)
    t0 = time.time()
    deadline = t0 + args.max_minutes * 60
    try:
        while time.time() < deadline:
            st = pod_status(pod_id)
            rt = st.get("runtime") or {}
            uptime = rt.get("uptimeInSeconds", 0) or 0
            cost = (st.get("costPerHr") or 0) * (uptime / 3600.0)
            log.info(
                "status=%s uptime=%ss cost=$%.3f",
                st.get("desiredStatus"),
                uptime,
                cost,
            )
            if st.get("desiredStatus") == "EXITED":
                log.info("pod exited cleanly")
                break
            time.sleep(20)
        else:
            log.warning("pod %s past deadline — terminating", pod_id)
    finally:
        terminate_pod(pod_id)
    log.info("done — %s ran for %.1f minutes", pod_id, (time.time() - t0) / 60.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
