#!/bin/zsh
# Refresh the candidate database and the Promising New Bodies page (2026-11-19). gather.py picks up every run with lead checks (runs/<run>/lead_check/), so new searches join by themselves.
# Then publish progress/objects/index.html to <private-page-url> (Artifact tool, same URL).
set -e
cd "${0:A:h}/.."
# one refresh at a time (2026-11-22: the decam-search loop's refresh and TRKSUB01's collided twice in package.py's _drive):
# a second caller waits for the first, up to 30 min; a lock whose owner is gone is taken over
L=candidates/.refresh.lock
for _ in {1..360}; do
  if mkdir $L 2>/dev/null; then echo $$ > $L/pid; break; fi
  kill -0 "$(cat $L/pid 2>/dev/null)" 2>/dev/null || { rm -rf $L; continue; }
  sleep 5
done
[[ "$(cat $L/pid 2>/dev/null)" == "$$" ]] || { echo "refresh.sh: another refresh still running after 30 min, skipped"; exit 1; }
trap 'rm -rf $L' EXIT
python3 -I progress/objects/gather.py | sed -n 2p
python3 -I candidates/build.py
DBT_SITE=nersc ~/anaconda3/bin/python -I progress/objects/observe.py --sites G37 | tail -1
python3 -I progress/objects/build_page.py
DBT_SITE=nersc ~/anaconda3/bin/python -I candidates/package.py | tail -2   # private Drive copy, one folder per object (2026-11-19)
