#!/usr/bin/env bash
# Parallel PROPHET matrix: runs many agents concurrently via background jobs.
# Trades modest API rate-limit risk for ~10x wall-clock improvement.
#
# Usage:
#   PARALLEL=6 N=20 TOTAL_BUDGET=18 bash scripts/run_openrouter_parallel.sh
set -uo pipefail
cd "$(dirname "$0")/.."

OUT="${OUT:-results/openrouter_matrix}"
N="${N:-20}"
SEED="${SEED:-42}"
CONC="${CONC:-8}"
PARALLEL="${PARALLEL:-6}"
PER_SEC="${PER_SEC:-0}"
FAMILIES="${FAMILIES:-all}"
MAX_TOKENS="${MAX_TOKENS:-3072}"
GLOBAL_LOG="$OUT/_matrix.log"
TOTAL_BUDGET="${TOTAL_BUDGET:-18}"

mkdir -p "$OUT" "$OUT/_state" "$OUT/_per_agent_logs"
echo "PROPHET parallel matrix → $OUT  (n=$N seed=$SEED conc=$CONC parallel=$PARALLEL families=$FAMILIES max_tokens=$MAX_TOKENS budget=$$TOTAL_BUDGET)" | tee -a "$GLOBAL_LOG"

# (uri, cost_cap_usd)
declare -a AGENTS=(
  "baseline:random|0"
  "baseline:always-take|0"
  "baseline:always-pass|0"
  "baseline:oracle|0"
  # Cheapest first so we get something quickly
  "openrouter:openai/gpt-5-nano|0.5"
  "openrouter:qwen/qwen3-32b|0.5"
  "openrouter:meta-llama/llama-4-maverick|0.5"
  "openrouter:meta-llama/llama-4-scout|0.5"
  "openrouter:deepseek/deepseek-v3.2|0.5"
  "openrouter:openai/gpt-5.4-nano|0.7"
  "openrouter:google/gemini-3.1-flash-lite|0.8"
  "openrouter:openai/gpt-5-mini|1.0"
  "openrouter:google/gemini-3-flash-preview|1.0"
  "openrouter:anthropic/claude-haiku-4.5|1.5"
  "openrouter:qwen/qwen3-235b-a22b-thinking-2507|1.0"
  "openrouter:moonshotai/kimi-k2-thinking|1.5"
  "openrouter:deepseek/deepseek-r1|1.5"
  "openrouter:openai/gpt-5|2.0"
  "openrouter:google/gemini-3.1-pro-preview|2.0"
  "openrouter:anthropic/claude-sonnet-4.6|2.5"
  "openrouter:anthropic/claude-opus-4.7|3.5"
)

run_one() {
  local uri="$1"
  local cap="$2"
  local safe="$(echo "$uri" | tr '/:' '_' | tr -d ' ')"
  local done_marker="$OUT/_state/$safe.done"
  local per_log="$OUT/_per_agent_logs/$safe.log"

  if [[ -f "$done_marker" ]]; then
    echo "[SKIP-DONE] $uri" | tee -a "$GLOBAL_LOG"
    return 0
  fi

  echo "[RUN] $uri  (cap=\$$cap conc=$CONC)" | tee -a "$GLOBAL_LOG"
  local start
  start=$(date +%s)

  if .venv/bin/prophet run \
      --agent "$uri" \
      --families "$FAMILIES" \
      --n "$N" \
      --seed "$SEED" \
      --max-cost "$cap" \
      --out-dir "$OUT/runs" \
      --concurrency "$CONC" \
      --per-sec-limit "$PER_SEC" \
      --max-tokens "$MAX_TOKENS" \
      --log-level WARNING \
      >> "$per_log" 2>&1; then
    local elapsed=$(( $(date +%s) - start ))
    echo "[OK] $uri  rc=0  ${elapsed}s" | tee -a "$GLOBAL_LOG"
    touch "$done_marker"
  else
    local rc=$?
    local elapsed=$(( $(date +%s) - start ))
    echo "[FAIL] $uri  rc=$rc  ${elapsed}s" | tee -a "$GLOBAL_LOG"
  fi
}

export -f run_one
export OUT N SEED CONC PER_SEC FAMILIES MAX_TOKENS GLOBAL_LOG

# Run baselines synchronously first (they're free + fast)
for entry in "${AGENTS[@]:0:4}"; do
  uri="${entry%%|*}"
  cap="${entry##*|}"
  run_one "$uri" "$cap"
done

# Run paid agents in parallel batches of $PARALLEL
batch=()
for entry in "${AGENTS[@]:4}"; do
  uri="${entry%%|*}"
  cap="${entry##*|}"
  batch+=("$uri|$cap")
  if [[ "${#batch[@]}" -ge "$PARALLEL" ]]; then
    echo "[BATCH-START] ${#batch[@]} jobs" | tee -a "$GLOBAL_LOG"
    for j in "${batch[@]}"; do
      uri="${j%%|*}"
      cap="${j##*|}"
      run_one "$uri" "$cap" &
    done
    wait
    echo "[BATCH-END]" | tee -a "$GLOBAL_LOG"
    batch=()
  fi
done

# Drain trailing batch
if [[ "${#batch[@]}" -gt 0 ]]; then
  echo "[BATCH-START] ${#batch[@]} jobs (trail)" | tee -a "$GLOBAL_LOG"
  for j in "${batch[@]}"; do
    uri="${j%%|*}"
    cap="${j##*|}"
    run_one "$uri" "$cap" &
  done
  wait
  echo "[BATCH-END]" | tee -a "$GLOBAL_LOG"
fi

echo "" | tee -a "$GLOBAL_LOG"
echo "All done. Running analyze-compare + leaderboard..." | tee -a "$GLOBAL_LOG"
.venv/bin/prophet analyze-compare "$OUT/runs" >> "$GLOBAL_LOG" 2>&1
.venv/bin/prophet leaderboard "$OUT/runs" >> "$GLOBAL_LOG" 2>&1
echo "Reports ready in $OUT" | tee -a "$GLOBAL_LOG"
