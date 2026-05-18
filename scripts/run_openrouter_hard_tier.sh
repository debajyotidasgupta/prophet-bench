#!/usr/bin/env bash
# PROPHET hard-tier (T_extreme, d ≥ 0.97) matrix runner.
# Targets <20% accuracy for frontier models to validate calibration spread
# under genuine difficulty.
#
# Usage:
#   PARALLEL=6 N=10 bash scripts/run_openrouter_hard_tier.sh
set -uo pipefail
cd "$(dirname "$0")/.."

OUT="${OUT:-results/openrouter_hard_tier}"
N="${N:-8}"
SEED="${SEED:-42}"
CONC="${CONC:-6}"
PARALLEL="${PARALLEL:-5}"
PER_SEC="${PER_SEC:-0}"
FAMILIES="${FAMILIES:-math,reasoning,code,knowledge,multimodal}"
DIFFICULTY="${DIFFICULTY:-0.97,1.0}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
GLOBAL_LOG="$OUT/_matrix.log"
TOTAL_BUDGET="${TOTAL_BUDGET:-15}"

mkdir -p "$OUT" "$OUT/_state" "$OUT/_per_agent_logs"
echo "PROPHET hard-tier matrix → $OUT  (n=$N seed=$SEED families=$FAMILIES difficulty=$DIFFICULTY max_tokens=$MAX_TOKENS budget=\$$TOTAL_BUDGET)" | tee -a "$GLOBAL_LOG"

# Baselines + frontier panel. New GPT-5.x lineage added per user request.
# Cost caps lean tight since each run is ~60 tasks (6 fams × 10 each).
declare -a AGENTS=(
  "baseline:oracle|0"
  "baseline:random|0"
  "baseline:always-pass|0"
  # Cheap reference panel
  "openrouter:openai/gpt-5-nano|0.3"
  "openrouter:openai/gpt-5.4-nano|0.3"
  "openrouter:google/gemini-3.1-flash-lite|0.3"
  "openrouter:meta-llama/llama-4-scout|0.3"
  "openrouter:meta-llama/llama-4-maverick|0.4"
  "openrouter:deepseek/deepseek-v3.2|0.4"
  "openrouter:qwen/qwen3-32b|0.4"
  # Mid-tier frontier
  "openrouter:openai/gpt-5-mini|0.6"
  "openrouter:google/gemini-3-flash-preview|0.6"
  "openrouter:anthropic/claude-haiku-4.5|0.8"
  "openrouter:anthropic/claude-sonnet-4.6|1.0"
  # Heavy frontier — top calibration candidates
  "openrouter:openai/gpt-5|1.5"
  "openrouter:anthropic/claude-opus-4.7|2.0"
  "openrouter:google/gemini-3.1-pro-preview|1.5"
  # NEW: GPT-5.x lineage (user-requested additions)
  "openrouter:openai/gpt-5.1|1.2"
  "openrouter:openai/gpt-5.2|1.5"
  "openrouter:openai/gpt-5.4|2.0"
  "openrouter:openai/gpt-5.5|2.5"
  # Reasoning specialists
  "openrouter:deepseek/deepseek-r1|1.0"
  "openrouter:qwen/qwen3-235b-a22b-thinking-2507|0.8"
  "openrouter:moonshotai/kimi-k2-thinking|1.0"
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

  echo "[RUN] $uri  (cap=\$$cap d=$DIFFICULTY conc=$CONC)" | tee -a "$GLOBAL_LOG"
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
      --difficulty-range "$DIFFICULTY" \
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
export OUT N SEED CONC PER_SEC FAMILIES MAX_TOKENS DIFFICULTY GLOBAL_LOG

# Baselines first (free + fast)
for entry in "${AGENTS[@]:0:3}"; do
  uri="${entry%%|*}"
  cap="${entry##*|}"
  run_one "$uri" "$cap"
done

# Paid agents in batches of $PARALLEL
batch=()
for entry in "${AGENTS[@]:3}"; do
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

# Drain
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
echo "All done. Running analyze-compare + leaderboard for hard tier..." | tee -a "$GLOBAL_LOG"
.venv/bin/prophet analyze-compare "$OUT/runs" >> "$GLOBAL_LOG" 2>&1
.venv/bin/prophet leaderboard "$OUT/runs" >> "$GLOBAL_LOG" 2>&1
echo "Reports ready in $OUT" | tee -a "$GLOBAL_LOG"
