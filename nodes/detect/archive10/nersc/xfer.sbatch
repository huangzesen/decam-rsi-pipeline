#!/bin/bash
# archive10 on NERSC (2026-11-14): one free xfer job (Perlmutter login node, charge 0 per NERSC's QOS table) downloads its share of
# new10_nersc.tsv (the 10 % selection's 45,073 exposures not in the 5 % search, best regions first) with fetch.py; no detection here.
# Guard: every 5 min the user's /pscratch use is read (lfs quota); at LIMIT_KB (default 19 TiB of the 20 TiB quota) it
# touches log/STOP, so fetch.py starts no new file (in-flight ones finish) and the job ends: the "hold".
# Resume later (space freed, e.g. after Globus to TACC): resubmit; fetch.py skips files already on disk.
# Submit from $A/a10n_k: sbatch --export=ALL,PART=k/4 ../xfer.sbatch   (manifest: new10_missing.tsv = not on NERSC already)
#SBATCH -q xfer
#SBATCH -C cron
#SBATCH --licenses=SCRATCH
#SBATCH --mem=8G
#SBATCH -A $NERSC_CPU_ACCOUNT
#SBATCH -t 48:00:00
#SBATCH -J a10x
#SBATCH -o log/xfer-%j.out
A=$NERSC_SCRATCH/archive10; PY=$NERSC_SCRATCH/dbt/env/bin/python
D=$SLURM_SUBMIT_DIR; cd $D || exit 1; mkdir -p log img det; rm -f log/FETCH_DONE log/STOP
LIMIT_KB=${LIMIT_KB:-20401094656}; STREAMS=${STREAMS:-24}
echo "xfer $(hostname) $(date); part $PART; $STREAMS streams; hold at $LIMIT_KB KB"
( while sleep 300; do u=$(lfs quota -q -u $NERSC_USER /pscratch 2>/dev/null | awk 'NR==1 {print $2}')
    if [ -n "$u" ] && [ "$u" -ge "$LIMIT_KB" ]; then echo "$(date) scratch $u KB >= $LIMIT_KB KB: hold (STOP)" | tee -a log/guard.log; touch log/STOP; break; fi
  done ) &
G=$!
ok=0; for t in 1 2 3 4 5 6 7 8 9 10; do
  c=$(curl -s -o /dev/null -m 30 -w "%{http_code}" https://astroarchive.noirlab.edu/); echo "archive: http $c (try $t)"
  [ "$c" = "200" ] && { ok=1; break; }; sleep 30
done
[ $ok = 1 ] || { echo "no archive from this node"; kill $G; exit 2; }
$PY -I $A/fetch.py $A/${MANIFEST:-new10_missing.tsv} $D --part $PART --streams $STREAMS --hours 47 > log/fetch_${SLURM_JOB_ID}.log 2>&1
echo "fetch ended $(date) (exit $?)"
kill $G 2>/dev/null; touch log/FETCH_DONE
