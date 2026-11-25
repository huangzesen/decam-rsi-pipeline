#!/bin/bash
# One headless worker agent through the local relay (2026-11-22).
# Usage: bash doublecheck/launch.sh <pass>      MODEL=claude-opus-5-5 EFFORT=xhigh by default
#   task text   doublecheck/<pass>.task.md
#   trace       doublecheck/<pass>.trace.jsonl   (stream-json; its mtime shows whether the agent is alive)
#   hand-back   doublecheck/<pass>.handback.md   (written by the agent, at most 25 lines)
# The relay key is read by $RELAY_CLI and never printed.
set -u
A=$ALE_HOME/planet-search/rsi/candidates/allarch
p="$1"; cd "$A" || exit 2
[ -f "doublecheck/$p.task.md" ] || { echo "no task file for $p"; exit 2; }
export CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS=0 CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 DISABLE_AUTOUPDATER=1 OMP_NUM_THREADS=1
echo "agent $p started $(date +%T) on ${MODEL:-claude-opus-5-5} effort ${EFFORT:-xhigh}"
$RELAY_CLI -p "$(cat "doublecheck/$p.task.md")" --model "${MODEL:-claude-opus-5-5}" --effort "${EFFORT:-xhigh}" \
  --allowedTools Bash Read Write Edit Glob Grep --permission-mode acceptEdits \
  --output-format stream-json --verbose < /dev/null > "doublecheck/$p.trace.jsonl" 2> "doublecheck/$p.agent.err"
rc=$?
echo "agent $p exited rc=$rc at $(date +%T)"
if [ -s "doublecheck/$p.handback.md" ]; then echo "--- hand-back:"; head -40 "doublecheck/$p.handback.md"
else echo "--- no hand-back file; end of the trace:"; tail -1 "doublecheck/$p.trace.jsonl" | cut -c1-3000; tail -5 "doublecheck/$p.agent.err"; fi
