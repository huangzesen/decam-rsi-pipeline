#!/bin/bash
# Step B of the second 5 % (2026-11-16): detection lists for one share (sb_<k>, made by stepB_farm.py)
# on one whole CPU node. Detection runs on CPU nodes ($NERSC_CPU_ACCOUNT), not GPU (2026-11-16).
# detect_pad.py = detect.py + the one-pixel border cut for CP V4.9 weight maps + the skip of a CCD that cannot be read.
# 56 workers: an exposure takes about 6.5 GB, the node has 512 GB. Measured in step A: 25 to 28 exposures a minute per
# node, so a share of 2,400 to 2,630 exposures takes about 1.6 to 1.8 h. A rerun skips the lists already written, so a
# job that hits its wall is simply submitted again.
# Submit from $NERSC_SCRATCH/archive10:  sbatch --export=ALL,DIR=$PWD/sb_0 -t 02:30:00 stepB_detect.sbatch
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -A $NERSC_CPU_ACCOUNT
#SBATCH -J a10sb
#SBATCH -o log/sb-%j.out
A=$NERSC_SCRATCH/archive10; PY=$NERSC_SCRATCH/dbt/env/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "detect $(hostname) $(date) dir $DIR workers ${WORKERS:-56}"
$PY -I $A/detect_pad.py $DIR/img $DIR/det --workers ${WORKERS:-56} > $DIR/log/detect_${SLURM_JOB_ID}.out 2>&1
echo "done $(date); $(find $DIR/det -name 'c4d_*.npz' | wc -l) detection lists; $(grep -c ERROR $DIR/det/detect.log) errors; $(grep -c unreadable $DIR/det/detect.log) exposures with an unreadable CCD"
