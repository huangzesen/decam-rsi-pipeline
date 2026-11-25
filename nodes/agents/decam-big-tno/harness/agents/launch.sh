#!/bin/bash
# One headless worker agent through the local relay (same pattern as planet-search/rsi/candidates/allarch/doublecheck/launch.sh).
# Usage: bash harness/agents/launch.sh <name>   task <name>.task.md, trace <name>.trace.jsonl, hand-back <name>.handback.md
set -u
D=$ALE_HOME/rsi/decam-big-tno; p="$1"; cd "$D" || exit 2
[ -f "harness/agents/$p.task.md" ] || { echo "no task file for $p"; exit 2; }
export CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS=0 CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 DISABLE_AUTOUPDATER=1 OMP_NUM_THREADS=1
echo "agent $p started $(date +%T) on ${MODEL:-claude-opus-5-5} effort ${EFFORT:-xhigh}"
$RELAY_CLI -p "$(cat "harness/agents/$p.task.md")" --model "${MODEL:-claude-opus-5-5}" --effort "${EFFORT:-xhigh}" \
  --allowedTools Bash Read Write Edit Glob Grep --permission-mode acceptEdits \
  --output-format stream-json --verbose < /dev/null > "harness/agents/$p.trace.jsonl" 2> "harness/agents/$p.agent.err"
rc=$?; echo "agent $p exited rc=$rc at $(date +%T)"
if [ -s "harness/agents/$p.handback.md" ]; then echo "--- hand-back:"; head -40 "harness/agents/$p.handback.md"
else echo "--- no hand-back; end of trace:"; tail -1 "harness/agents/$p.trace.jsonl" | cut -c1-3000; tail -5 "harness/agents/$p.agent.err"; fi
