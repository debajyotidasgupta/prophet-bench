#!/usr/bin/env bash
# One-shot progress dump: agents done, leaderboard, budget so far.
set -uo pipefail
cd "$(dirname "$0")/.."

OUT="${OUT:-results/openrouter_matrix}"

echo "=== STATE ==="
ls "$OUT/_state/" 2>/dev/null | sort | sed 's/^/  /'
echo ""
echo "=== ACTIVE PROCESSES ==="
ps aux | grep -E "prophet run.*--agent" | grep -v grep \
  | awk '{for(i=1;i<=NF;i++) if($i=="--agent"){print "  "$(i+1)}}' | sort -u
echo ""
echo "=== RECENT EVENTS ==="
grep -E "^\[OK\]|^\[FAIL\]|^\[BATCH" "$OUT/_matrix.log" 2>/dev/null | tail -15
echo ""
echo "=== LEADERBOARD ==="
.venv/bin/prophet leaderboard "$OUT/runs" 2>/dev/null >/dev/null
cat "$OUT/leaderboard/leaderboard.md" 2>/dev/null | head -30
