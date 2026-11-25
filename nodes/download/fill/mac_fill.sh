#!/bin/bash
# DECam fill, Mac share (2026-11-20). Runs fill/fetch.py on
# blank_mac.tsv into the Crucial-8TB buffer in chunks of <= 0.1 TB while the buffer holds < 200 GB; relay_mac.py pushes
# finished files to Vista and deletes them. Stop: touch $B/log/STOP_ALL (and fetch's own log/STOP ends the current chunk).
B=$DATA_DRIVE/decam-fill-mac
F=$(cd "$(dirname "$0")" && pwd)/fetch.py
STREAMS=${STREAMS:-24}
# 2026-11-22: a python with astropy, so fetch.py can keep files whose listed md5 is stale but whose FITS DATASUMs verify
# (Homebrew python3 has no astropy: every such file failed 4 tries on the Mac, e.g. 2013 v1 files)
PY=${PY:-$HOME/anaconda3/bin/python}
while [ ! -e $B/log/STOP_ALL ]; do
  [ -d $DATA_DRIVE ] || { echo "$(date '+%m-%d %H:%M:%S') Crucial-8TB not mounted; waiting"; sleep 300; continue; }
  used=$(du -sk $B/img 2>/dev/null | cut -f1)
  if [ "${used:-0}" -lt 200000000 ]; then
    nice -n 10 "$PY" -I "$F" $B/blank_mac.tsv $B --streams $STREAMS --hours 1 --max-tb 0.1 >> $B/log/fetch_mac.log 2>&1
    rc=$?
    tail -3 $B/log/fetch_mac.log | grep -q "share finished" && { echo "$(date '+%m-%d %H:%M:%S') share finished"; break; }
    [ $rc = 3 ] && sleep 300                     # the archive kept refusing: wait before the next chunk
  else
    sleep 300
  fi
done
