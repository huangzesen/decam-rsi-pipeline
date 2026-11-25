#!/bin/bash
# archive10 step A: single-exposure detection lists (detect.py, S/N >= 4.5) for one share sa_k of the five step A regions
# on one whole CPU node (regular QOS, $NERSC_CPU_ACCOUNT, charged per node-hour). 56 workers, not 96: one exposure peaks at 6.5 GB
# (login-node test 2026-11-16 14:06), and the 96-worker jobs <slurm-job-id>/93 ran out of the node's 512 GB, which killed the
# whole pool. A rerun skips the lists already written, so a job that hits its wall is simply submitted again.
# Submit from $A: sbatch --export=ALL,DIR=$A/sa_0 -t 02:30:00 stepA_detect.sbatch
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -A $NERSC_CPU_ACCOUNT
#SBATCH -J a10sa
#SBATCH -o log/sa-%j.out
A=$NERSC_SCRATCH/archive10; PY=$NERSC_SCRATCH/dbt/env/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "detect $(hostname) $(date) dir $DIR workers ${WORKERS:-56}"
$PY -I $A/detect.py $DIR/img $DIR/det --workers ${WORKERS:-56} > $DIR/log/detect_${SLURM_JOB_ID}.out 2>&1
echo "done $(date); $(ls $DIR/det/*/*.npz 2>/dev/null | wc -l) detection lists; $(grep -c ERROR $DIR/det/detect.log) errors"
