#!/usr/bin/env bash
# Phase B: N=20 hard-tier matrix on hardened generators across 3 seeds for variance.
# Per the top-tier-submission checklist.
set -uo pipefail
cd "$(dirname "$0")/.."

OUT="${OUT:-results/openrouter_hard_v2}"
N="${N:-20}"
CONC="${CONC:-6}"
PARALLEL="${PARALLEL:-4}"
FAMILIES="${FAMILIES:-math,reasoning,code,knowledge,multimodal}"
DIFFICULTY="${DIFFICULTY:-0.97,1.0}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
GLOBAL_LOG="$OUT/_matrix.log"

mkdir -p "$OUT" "$OUT/_state" "$OUT/_per_agent_logs"
echo "PROPHET hard-tier Phase B (multi-seed N=$N) → $OUT" | tee -a "$GLOBAL_LOG"

# Seeds: 42 for full panel; 7 and 1729 for Pareto-essential 4 only.
SEED_42_AGENTS=(
  "openrouter:openai/gpt-5-nano|0.3"
  "openrouter:openai/gpt-5.4-nano|0.4"
  "openrouter:google/gemini-3.1-flash-lite|0.4"
  "openrouter:meta-llama/llama-4-scout|0.3"
  "openrouter:meta-llama/llama-4-maverick|0.4"
  "openrouter:qwen/qwen3-32b|0.4"
  "openrouter:deepseek/deepseek-v3.2|0.5"
  "openrouter:openai/gpt-5-mini|0.6"
  "openrouter:google/gemini-3-flash-preview|0.6"
  "openrouter:anthropic/claude-haiku-4.5|1.0"
  "openrouter:google/gemini-3.1-pro-preview|2.0"
  "openrouter:qwen/qwen3-235b-a22b-thinking-2507|1.0"
  "openrouter:openai/gpt-5|2.0"
  "openrouter:openai/gpt-5.2|2.0"
  "openrouter:anthropic/claude-opus-4.7|2.5"
  "openrouter:deepseek/deepseek-r1|1.2"
)

# Pareto-essentials for multi-seed variance (4 agents × 2 extra seeds)
PARETO_SEEDS_7_1729=(
  "openrouter:openai/gpt-5|2.0"
  "openrouter:openai/gpt-5.2|2.0"
  "openrouter:anthropic/claude-opus-4.7|2.5"
  "openrouter:deepseek/deepseek-r1|1.2"
)

run_one() {
  local uri="$1"
  local cap="$2"
  local seed="$3"
  local safe="$(echo "${uri}_s${seed}" | tr '/:' '_' | tr -d ' ')"
  local done_marker="$OUT/_state/$safe.done"
  local per_log="$OUT/_per_agent_logs/$safe.log"

  if [[ -f "$done_marker" ]]; then
    echo "[SKIP-DONE] $uri seed=$seed" | tee -a "$GLOBAL_LOG"
    return 0
  fi

  echo "[RUN] $uri seed=$seed (cap=\$$cap)" | tee -a "$GLOBAL_LOG"
  local start; start=$(date +%s)

  if .venv/bin/prophet run \
      --agent "$uri" \
      --families "$FAMILIES" \
      --n "$N" \
      --seed "$seed" \
      --max-cost "$cap" \
      --out-dir "$OUT/runs" \
      --concurrency "$CONC" \
      --max-tokens "$MAX_TOKENS" \
      --difficulty-range "$DIFFICULTY" \
      --log-level WARNING \
      >> "$per_log" 2>&1; then
    local elapsed=$(( $(date +%s) - start ))
    echo "[OK] $uri seed=$seed  rc=0  ${elapsed}s" | tee -a "$GLOBAL_LOG"
    touch "$done_marker"
  else
    local rc=$?
    local elapsed=$(( $(date +%s) - start ))
    echo "[FAIL] $uri seed=$seed  rc=$rc  ${elapsed}s" | tee -a "$GLOBAL_LOG"
  fi
}

export -f run_one
export OUT N CONC FAMILIES MAX_TOKENS DIFFICULTY GLOBAL_LOG

# Baselines first (free)
for s in 42 7 1729; do
  for agent in "baseline:oracle" "baseline:random" "baseline:always-pass"; do
    run_one "$agent" "0" "$s"
  done
done

# Phase B-1: seed=42 on full panel
echo "=== Phase B-1: seed=42 on full agent panel ===" | tee -a "$GLOBAL_LOG"
batch=()
for entry in "${SEED_42_AGENTS[@]}"; do
  uri="${entry%%|*}"; cap="${entry##*|}"
  batch+=("$uri|$cap|42")
  if [[ "${#batch[@]}" -ge "$PARALLEL" ]]; then
    echo "[BATCH-START] ${#batch[@]}" | tee -a "$GLOBAL_LOG"
    for j in "${batch[@]}"; do
      uri="${j%%|*}"; rest="${j#*|}"; cap="${rest%%|*}"; s="${rest##*|}"
      run_one "$uri" "$cap" "$s" &
    done
    wait
    echo "[BATCH-END]" | tee -a "$GLOBAL_LOG"
    batch=()
  fi
done
if [[ "${#batch[@]}" -gt 0 ]]; then
  for j in "${batch[@]}"; do
    uri="${j%%|*}"; rest="${j#*|}"; cap="${rest%%|*}"; s="${rest##*|}"
    run_one "$uri" "$cap" "$s" &
  done
  wait
fi

# Phase B-2: seeds 7, 1729 on Pareto-essential 4 only
echo "=== Phase B-2: seeds 7,1729 on Pareto-essential agents ===" | tee -a "$GLOBAL_LOG"
batch=()
for s in 7 1729; do
  for entry in "${PARETO_SEEDS_7_1729[@]}"; do
    uri="${entry%%|*}"; cap="${entry##*|}"
    batch+=("$uri|$cap|$s")
    if [[ "${#batch[@]}" -ge "$PARALLEL" ]]; then
      echo "[BATCH-START] ${#batch[@]}" | tee -a "$GLOBAL_LOG"
      for j in "${batch[@]}"; do
        uri="${j%%|*}"; rest="${j#*|}"; cap="${rest%%|*}"; ss="${rest##*|}"
        run_one "$uri" "$cap" "$ss" &
      done
      wait
      echo "[BATCH-END]" | tee -a "$GLOBAL_LOG"
      batch=()
    fi
  done
done
if [[ "${#batch[@]}" -gt 0 ]]; then
  for j in "${batch[@]}"; do
    uri="${j%%|*}"; rest="${j#*|}"; cap="${rest%%|*}"; ss="${rest##*|}"
    run_one "$uri" "$cap" "$ss" &
  done
  wait
fi

echo "" | tee -a "$GLOBAL_LOG"
echo "Phase B done. Building leaderboard..." | tee -a "$GLOBAL_LOG"
.venv/bin/prophet analyze-compare "$OUT/runs" >> "$GLOBAL_LOG" 2>&1
.venv/bin/prophet leaderboard "$OUT/runs" >> "$GLOBAL_LOG" 2>&1
echo "Reports ready in $OUT" | tee -a "$GLOBAL_LOG"
