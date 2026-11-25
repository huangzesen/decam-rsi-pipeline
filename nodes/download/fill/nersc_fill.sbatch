#!/bin/bash
# DECam fill, NERSC leg (2026-11-17; ,
# "i am ok to spend some cpu hour on nersc for downloading"). One job downloads part PART (k/N) of blank.tsv into
# $F/$PARTDIR/img/<YYMMDD>/ with fetch.py (md5-checked, resumable) in 1-hour chunks. fill/relay.py (on the Mac) copies
# finished files to Vista with Globus and deletes them here once the Globus task has succeeded (the third 5 %'s files,
# tag a15, stay until its detection is done).
# Space guard: a chunk starts only while the user's /pscratch use is below LOW_KB; during a chunk, use at HIGH_KB touches
# log/STOP (fetch.py starts no new file); the job then waits until use falls below LOW_KB again (the relay frees space).
# Limits can be changed while jobs run: write LOW_KB=...; HIGH_KB=... to $F/limits (2026-11-17 22:40).
# Stop all parts: touch $F/STOP_ALL (each job ends after its current file transfers).
# Free (login node):  sbatch -q xfer -C cron --mem=8G -t 48:00:00 -J fillx0 --export=ALL,PART=0/6,PARTDIR=p0 nersc_fill.sbatch
# CPU node (charged): sbatch -q regular -C cpu -N 1 -t 24:00:00 -J fillc4 --export=ALL,PART=4/6,PARTDIR=p4,STREAMS=48 nersc_fill.sbatch
#SBATCH -A $NERSC_CPU_ACCOUNT
#SBATCH --licenses=SCRATCH
#SBATCH -o $NERSC_SCRATCH/decam-fill/log/job-%j.out
F=$NERSC_SCRATCH/decam-fill; PY=$NERSC_SCRATCH/dbt/env/bin/python
P=$F/$PARTDIR; mkdir -p $P/log $P/img; cd $P || exit 1
LOW_KB=${LOW_KB:-18790481920}     # 17.5 TiB
HIGH_KB=${HIGH_KB:-19864223744}   # 18.5 TiB of the 20 TiB quota
STREAMS=${STREAMS:-24}
lim() { [ -f $F/limits ] && . $F/limits; }   # $F/limits (LOW_KB=..; HIGH_KB=..) overrides both, re-read every cycle
WALL=$(squeue -h -j $SLURM_JOB_ID -o %L 2>/dev/null); echo "job $SLURM_JOB_ID $(hostname) $(date); part $PART -> $P; $STREAMS streams; wall left $WALL"
END=$(( $(date +%s) + ${HOURS:-47} * 3600 ))
use() { lfs quota -q -u $NERSC_USER /pscratch 2>/dev/null | awk 'NR==1 {print $2}'; }
ok=0; for t in 1 2 3 4 5 6 7 8 9 10; do
  c=$(curl -s -o /dev/null -m 30 -w "%{http_code}" https://astroarchive.noirlab.edu/); echo "archive: http $c (try $t)"
  [ "$c" = "200" ] && { ok=1; break; }; sleep 30
done
[ $ok = 1 ] || { echo "no archive from this node"; exit 2; }
while [ $(( END - $(date +%s) )) -gt 1200 ]; do
  [ -f $F/STOP_ALL ] && { echo "$(date) STOP_ALL"; break; }
  lim; u=$(use)
  if [ -z "$u" ] || [ "$u" -ge "$LOW_KB" ]; then echo "$(date) scratch ${u:-?} KB >= $LOW_KB: waiting for the relay" >> log/guard.log; sleep 600; continue; fi
  rm -f log/STOP
  ( while sleep 120; do lim; v=$(use); if { [ -n "$v" ] && [ "$v" -ge "$HIGH_KB" ]; } || [ -f $F/STOP_ALL ]; then echo "$(date) scratch ${v:-?} KB: STOP" >> log/guard.log; touch log/STOP; break; fi; done ) &
  G=$!
  h=$(python3 -c "print(round(min(1.0, ($END - $(date +%s) - 900) / 3600), 3))")
  $PY -I $F/fetch.py $F/blank.tsv $P --part $PART --streams $STREAMS --hours $h >> log/fetch_${SLURM_JOB_ID}.log 2>&1; rc=$?
  kill $G 2>/dev/null; wait $G 2>/dev/null
  if tail -1 log/fetch_${SLURM_JOB_ID}.log | grep -q "share finished"; then echo "$(date) share finished"; touch log/SHARE_DONE; break; fi
  [ $rc = 3 ] && { echo "$(date) archive refusing, pause 15 min" >> log/guard.log; sleep 900; }
done
echo "end $(date)"
