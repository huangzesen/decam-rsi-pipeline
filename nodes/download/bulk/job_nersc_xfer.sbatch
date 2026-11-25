#!/bin/bash
# Relay test, leg 1 (2026-11-12):
# 13 pilot-group slices, 1.03 TB, from the NOIRLab archive to Perlmutter scratch in NERSC's free transfer queue.
# Two copies of fetch.py with 32 streams each. Stop: touch $NERSC_SCRATCH/decam-relay-test/log/STOP
#SBATCH --qos=xfer
#SBATCH -C cron
#SBATCH -A $NERSC_CPU_ACCOUNT
#SBATCH --time=06:00:00
#SBATCH --job-name=decam-relay-test
#SBATCH --licenses=SCRATCH
#SBATCH --mem=8GB
#SBATCH -o $NERSC_SCRATCH/decam-relay-test/log/job-%j.out
D=$NERSC_SCRATCH/decam-relay-test
cd $D || exit 1
module load python >/dev/null 2>&1
echo "node $(hostname), $(nproc) cpus, $(date)"
curl -s -o /dev/null -m 30 -w "archive from this node: http %{http_code}\n" https://astroarchive.noirlab.edu/ \
  || { echo "this node cannot reach the archive"; exit 2; }
for k in 0 1; do
  python -I fetch.py manifest.tsv $D --part $k/2 --streams 32 --hours 5.5 > log/fetch_${k}of2_${SLURM_JOB_ID}.log 2>&1 &
done
wait
echo "done $(date)"; du -sh $D/img | cut -f1
