#!/usr/bin/env bash
# Full PROPHET experimental matrix via OpenRouter. Cost-budgeted.
# Each agent is run sequentially with a per-agent cost cap. Outputs to
# results/openrouter_matrix/runs/<agent>/.

set -uo pipefail
cd "$(dirname "$0")/.."

OUT="${OUT:-results/openrouter_matrix}"
N="${N:-20}"
SEED="${SEED:-42}"
CONC="${CONC:-6}"
PER_SEC="${PER_SEC:-0}"
FAMILIES="${FAMILIES:-all}"
GLOBAL_LOG="$OUT/_matrix.log"

mkdir -p "$OUT"
echo "PROPHET matrix → $OUT  (n=$N seed=$SEED conc=$CONC families=$FAMILIES)" | tee -a "$GLOBAL_LOG"

# (uri, cost_cap_usd)  — order is roughly cheap → expensive so we get headlines early
declare -a AGENTS=(
  # Baselines (free, run first as sanity)
  "baseline:random|0"
  "baseline:always-take|0"
  "baseline:always-pass|0"
  "baseline:oracle|0"
  # Cheap open via OpenRouter
  "openrouter:openai/gpt-5-nano|0.5"
  "openrouter:qwen/qwen3-32b|0.5"
  "openrouter:meta-llama/llama-4-maverick|0.5"
  "openrouter:deepseek/deepseek-v3.2|0.5"
  "openrouter:openai/gpt-5.4-nano|0.7"
  # Mid tier
  "openrouter:openai/gpt-5-mini|1.0"
  "openrouter:google/gemini-3.1-flash-lite|0.8"
  "openrouter:google/gemini-3-flash-preview|1.0"
  "openrouter:anthropic/claude-haiku-4.5|1.5"
  # Reasoning models
  "openrouter:moonshotai/kimi-k2-thinking|1.5"
  "openrouter:deepseek/deepseek-r1|1.5"
  "openrouter:qwen/qwen3-235b-a22b-thinking-2507|1.0"
  # Premium
  "openrouter:openai/gpt-5|2.0"
  "openrouter:anthropic/claude-sonnet-4.6|2.5"
  "openrouter:google/gemini-3.1-pro-preview|2.0"
  # Frontier (most expensive — last)
  "openrouter:anthropic/claude-opus-4.7|3.5"
)

TOTAL_BUDGET="${TOTAL_BUDGET:-18}"
cumulative=0

for entry in "${AGENTS[@]}"; do
  uri="${entry%%|*}"
  cap="${entry##*|}"

  # Skip if cumulative spend would breach budget
  if (( $(echo "$cumulative + $cap > $TOTAL_BUDGET" | bc -l) )); then
    echo "[SKIP-BUDGET] $uri cap=\$$cap cumulative=\$$cumulative" | tee -a "$GLOBAL_LOG"
    continue
  fi

  safe="$(echo "$uri" | tr '/:' '_' | tr -d ' ')"
  done_marker="$OUT/_state/$safe.done"
  mkdir -p "$OUT/_state"
  if [[ -f "$done_marker" ]]; then
    echo "[SKIP-DONE] $uri" | tee -a "$GLOBAL_LOG"
    continue
  fi

  echo "[RUN] $uri  (cap=\$$cap)" | tee -a "$GLOBAL_LOG"
  start=$(date +%s)

  .venv/bin/prophet run \
    --agent "$uri" \
    --families "$FAMILIES" \
    --n "$N" \
    --seed "$SEED" \
    --max-cost "$cap" \
    --out-dir "$OUT/runs" \
    --concurrency "$CONC" \
    --per-sec-limit "$PER_SEC" \
    --log-level WARNING \
    >> "$GLOBAL_LOG" 2>&1
  rc=$?
  elapsed=$(( $(date +%s) - start ))

  if [[ $rc -eq 0 ]]; then
    echo "[OK] $uri  rc=0  $elapsed"s "" | tee -a "$GLOBAL_LOG"
    touch "$done_marker"
    cumulative=$(echo "$cumulative + $cap" | bc -l)
  else
    echo "[FAIL] $uri  rc=$rc  $elapsed"s | tee -a "$GLOBAL_LOG"
  fi

  echo "  → cumulative budget consumed (upper bound): \$$cumulative / \$$TOTAL_BUDGET" | tee -a "$GLOBAL_LOG"
done

echo "" | tee -a "$GLOBAL_LOG"
echo "All done. Running analyze-compare..." | tee -a "$GLOBAL_LOG"
.venv/bin/prophet analyze-compare "$OUT/runs" >> "$GLOBAL_LOG" 2>&1
echo "Report ready: $OUT/report/" | tee -a "$GLOBAL_LOG"
