#!/bin/bash
# One node of the DECam bulk download to scratch (fetch.py; manifest from make_manifest.py).
# Submit with -A, -p, -t and --export=ALL,FIRST=0,NPROC=2,NPARTS=4,STREAMS=32,HOURS=47
# (this node runs parts FIRST..FIRST+NPROC-1 of NPARTS; nodes running at the same time share one NPARTS).
# Stop every node of a machine: touch $TACC_SCRATCH/decam-bulk/log/STOP (remove it before a restart).
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J decam-bulk
#SBATCH -o $TACC_SCRATCH/decam-bulk/log/job-%j.out
D=$TACC_SCRATCH/decam-bulk
cd $D || exit 1
echo "node $(hostname) $(date); parts $FIRST..$((FIRST + NPROC - 1)) of $NPARTS, $STREAMS streams each, $HOURS h"
curl -s -o /dev/null -m 30 -w "archive from this node: http %{http_code}\n" https://astroarchive.noirlab.edu/ \
  || { echo "this node cannot reach the archive"; exit 2; }
for k in $(seq $FIRST $((FIRST + NPROC - 1))); do
  python3 -I fetch.py manifest.tsv $D --part $k/$NPARTS --streams $STREAMS --hours $HOURS \
    > log/fetch_${k}of${NPARTS}_${SLURM_JOB_ID}.log 2>&1 &
done
wait
echo "done $(date)"
