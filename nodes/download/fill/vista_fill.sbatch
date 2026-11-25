#!/bin/bash
# DECam fill, TACC leg (2026-11-17). One Vista gg node runs NPROC copies of
# fetch.py (md5-checked, resumable) over its parts of blank_vista.tsv into $TACC_SCRATCH/decam-fill/img.
# Download only: no detection here. Scratch purges files not accessed for 10 days.
# Submit from the folder: sbatch -A $VISTA_ACCOUNT -p gg -t 47:00:00 -J dfill_K --export=ALL,FIRST=..,NPROC=2,NPARTS=8,STREAMS=32,HOURS=46.5 vista_fill.sbatch
# Stop every node: touch $TACC_SCRATCH/decam-fill/log/STOP (fetch stops starting files).
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o log/job-%j.out
D=$TACC_SCRATCH/decam-fill
cd $D || exit 1
echo "node $(hostname) $(date); parts $FIRST..$((FIRST + NPROC - 1)) of $NPARTS, $STREAMS streams each, $HOURS h; $(python3 -V 2>&1)"
ok=0; for t in 1 2 3 4 5 6 7 8 9 10; do        # the archive answers slowly under load (2026-11-13), retry ~5 min
  c=$(curl -s -o /dev/null -m 30 -w "%{http_code}" https://astroarchive.noirlab.edu/); echo "archive: http $c (try $t)"
  [ "$c" = "200" ] && { ok=1; break; }; sleep 30
done
[ $ok = 1 ] || { echo "this node cannot reach the archive"; exit 2; }
for k in $(seq $FIRST $((FIRST + NPROC - 1))); do
  python3 -I fetch.py blank_vista.tsv $D --part $k/$NPARTS --streams $STREAMS --hours $HOURS \
    > log/fetch_${k}of${NPARTS}_${SLURM_JOB_ID}.log 2>&1 &
done
wait
echo "done $(date)"
