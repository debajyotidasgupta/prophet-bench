#!/usr/bin/env bash
# Complete the v2 multilingual run on agents not in the first batch.
# First batch had: gpt-5, gpt-5.2, opus-4.7, gemini-3.1-pro, gemini-3-flash,
# and R1 (errored at 38/40 due to 402). This adds R1 (rerun), Haiku 4.5,
# and V3.2 so the v2 panel has 8 agents.
#
# Pre-flight assumptions verified manually:
#   - data/seeds/multilingual_dict.json contains all 10 langs (v2 dict)
#   - results/multilingual_v2/runs/ exists (5 prior agents)
#   - .env has OPENROUTER_API_KEY restored
#   - PROPHET_MAX_RUN_COST_USD honored at the orchestrator level
#
# Cost estimate (per prior runs):
#   R1 ~ $0.50 (reasoning model)
#   Haiku 4.5 ~ $0.05
#   V3.2 ~ $0.02
#   Total ~ $0.60

set -euo pipefail
cd "$(dirname "$0")/.."

OUT_DIR=results/multilingual_v2/runs
mkdir -p "$OUT_DIR"

AGENTS=(
  "openrouter:deepseek/deepseek-r1"
  "openrouter:anthropic/claude-haiku-4.5"
  "openrouter:deepseek/deepseek-v3.2"
)

# Pre-flight: print plan
echo "==================== launch plan ===================="
echo "out-dir: $OUT_DIR"
echo "family: multilingual (v2 dict, 10 langs)"
echo "N: 40, seed: 42"
echo "per-agent cap: \$0.60"
echo "agents:"
for a in "${AGENTS[@]}"; do echo "  - $a"; done
echo "===================================================="

for AGENT in "${AGENTS[@]}"; do
  # Skip if already complete in the v2 dir (resumability)
  SLUG=$(echo "$AGENT" | sed 's|openrouter:||; s|/|_|g')
  if find "$OUT_DIR" -maxdepth 1 -type d -name "*$SLUG*" -print -quit | grep -q .; then
    echo "=== $AGENT: already has a run dir, skipping (delete to force rerun) ==="
    continue
  fi
  echo
  echo "=== $AGENT ==="
  # Pre-flight ping (~$0.0001) so a 402 / model-unavailable doesn't
  # silently fill a 40-task run with PASS outcomes.
  if ! .venv/bin/python scripts/preflight_agent_ping.py "$AGENT"; then
    echo "  preflight failed, skipping agent (no run dir created)"
    continue
  fi
  .venv/bin/prophet run \
    --agent "$AGENT" \
    --families multilingual \
    --n 40 \
    --seed 42 \
    --max-cost 0.60 \
    --out-dir "$OUT_DIR" \
    --concurrency 3 \
    --max-tokens 6000
done
echo
echo "v2 multilingual complete. Total agents in $OUT_DIR:"
ls "$OUT_DIR" | wc -l
