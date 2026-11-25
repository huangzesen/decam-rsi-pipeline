#!/bin/zsh
# Keeps <team-site-url> current: every 30 min the candidate database (refresh.sh: gather, cards, LDT nights,
# catalogue page, Drive folder) and then this site (build.py --deploy). Stop: touch decam/STOP. Log: decam/loop.log.
cd "${0:A:h}"
while [[ ! -f STOP ]]; do
  echo "== $(date '+%m-%d %H:%M')"
  $ALE_HOME/rsi/decam-big-tno/candidates/refresh.sh 2>&1 | grep -v WARNING | tail -3
  CLOUDFLARE_AUTH_USE_KEYRING=false python3 -I build.py --deploy 2>&1 | grep -E "wrote|Deployed|Error|error|refusing" | head -4
  sleep 1800
done
echo "stopped $(date '+%m-%d %H:%M')"
