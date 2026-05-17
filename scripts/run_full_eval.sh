#!/usr/bin/env bash
# Full reproducibility script: runs every baseline + every model on the full
# family matrix and produces the paper-ready figures + tables.
#
# Usage:
#   scripts/run_full_eval.sh [BUDGET_USD] [N_PER_FAMILY] [SEED]
#
# Cost guardrail: stops if total spend exceeds BUDGET_USD across all API
# baselines. Local + vLLM baselines have $0 inference cost.
set -euo pipefail
cd "$(dirname "$0")/.."

BUDGET="${1:-30.0}"
N_PER_FAMILY="${2:-100}"
SEED="${3:-42}"
OUT="${4:-results/full_eval}"

mkdir -p "$OUT"
echo "PROPHET full eval: budget=\$$BUDGET n=$N_PER_FAMILY seed=$SEED → $OUT"

run() {
  local label="$1"; shift
  local cost_cap="$1"; shift
  echo "::: $label  (cap \$$cost_cap)"
  PROPHET_MAX_RUN_COST_USD="$cost_cap" .venv/bin/prophet run \
    --agent "$1" --families all --n "$N_PER_FAMILY" --seed "$SEED" --max-cost "$cost_cap" --out-dir "$OUT/runs" "${@:2}" || echo "WARN: $label failed"
}

# ---------- baselines (free) ----------
run "baseline:random"       0   "baseline:random"
run "baseline:always-take"  0   "baseline:always-take"
run "baseline:always-pass"  0   "baseline:always-pass"
run "baseline:oracle"       0   "baseline:oracle"
run "baseline:oracle-noisy" 0   "baseline:oracle" --notes "force-noisy-via-config"

# ---------- closed-model APIs (small first; halt early if budget tight) ----------
run "openai:gpt-5-mini"   3 "openai:gpt-5-mini" --concurrency 8
run "anthropic:claude-haiku-4-5" 3 "anthropic:claude-haiku-4-5" --concurrency 4
run "google:gemini-3-flash" 3 "google:gemini-3-flash" --concurrency 4
run "openai:gpt-5"         8 "openai:gpt-5" --concurrency 4
run "anthropic:claude-sonnet-4-6" 8 "anthropic:claude-sonnet-4-6" --concurrency 4
run "google:gemini-3.1-pro" 8 "google:gemini-3.1-pro" --concurrency 4

# ---------- open models via Together (cheap, no GPU spin) ----------
run "together:Qwen/Qwen3-7B-Instruct"  2 "together:Qwen/Qwen3-7B-Instruct" --concurrency 8
run "together:Qwen/Qwen3-32B-Instruct" 4 "together:Qwen/Qwen3-32B-Instruct" --concurrency 4
run "together:meta-llama/Llama-4-70B-Instruct" 6 "together:meta-llama/Llama-4-70B-Instruct" --concurrency 4

# ---------- analysis ----------
.venv/bin/prophet analyze "$OUT/runs"

echo "Full eval done. See $OUT/runs."
