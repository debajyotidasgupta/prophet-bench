#!/usr/bin/env bash
# Pruned hard-tier runner — only essential frontier + user-requested GPT-5.2.
# Skips already-done agents via state markers.
set -uo pipefail
cd "$(dirname "$0")/.."

OUT="${OUT:-results/openrouter_hard_tier}"
N="${N:-8}"
SEED="${SEED:-42}"
CONC="${CONC:-6}"
PARALLEL="${PARALLEL:-3}"
FAMILIES="${FAMILIES:-math,reasoning,code,knowledge,multimodal}"
DIFFICULTY="${DIFFICULTY:-0.97,1.0}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
GLOBAL_LOG="$OUT/_matrix.log"

mkdir -p "$OUT" "$OUT/_state" "$OUT/_per_agent_logs"
echo "PROPHET hard-tier essentials → $OUT  (pruned for cost)" | tee -a "$GLOBAL_LOG"

# Pruned to essentials: GPT-5 baseline + 3 frontier (Opus, Gemini Pro) +
# user-requested GPT-5.2 + 2 open reasoning specialists.
declare -a AGENTS=(
  "openrouter:openai/gpt-5|1.5"
  "openrouter:anthropic/claude-opus-4.7|2.0"
  "openrouter:google/gemini-3.1-pro-preview|1.5"
  "openrouter:openai/gpt-5.2|1.5"
  "openrouter:deepseek/deepseek-r1|1.0"
  "openrouter:qwen/qwen3-235b-a22b-thinking-2507|0.8"
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

  echo "[RUN] $uri  (cap=\$$cap)" | tee -a "$GLOBAL_LOG"
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
export OUT N SEED CONC FAMILIES MAX_TOKENS DIFFICULTY GLOBAL_LOG

batch=()
for entry in "${AGENTS[@]}"; do
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

echo "Essentials done. Re-running leaderboard..." | tee -a "$GLOBAL_LOG"
.venv/bin/prophet analyze-compare "$OUT/runs" >> "$GLOBAL_LOG" 2>&1
.venv/bin/prophet leaderboard "$OUT/runs" >> "$GLOBAL_LOG" 2>&1
echo "Reports ready in $OUT" | tee -a "$GLOBAL_LOG"
