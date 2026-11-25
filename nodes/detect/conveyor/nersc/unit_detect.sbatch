#!/bin/bash
# conveyor unit detection for one share (DIR = <unit>/<share>), a copy of ldt5/l5_detect_dbg.sbatch. The coordinator picks
# the queue: CPU debug (30 min) while NERSC's other CPU queues wait, or CPU regular when it starts soon. detect_pad writes
# each list atomically, so a job stopped by its limit loses only the exposures in flight; the next one skips done lists.
# sbatch -q debug -t 00:30:00 -J cvd-<uid>-<share> --export=ALL,DIR=<unit>/<share>,TL=1680 unit_detect.sbatch
# sbatch -q regular -t 03:00:00 -J cvd-<uid>-<share> --export=ALL,DIR=<unit>/<share>,TL=10500 unit_detect.sbatch
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -A $NERSC_CPU_ACCOUNT
#SBATCH -o $NERSC_SCRATCH/units/slurm/detect-%j.out
A=$NERSC_SCRATCH/archive10; PY=$NERSC_SCRATCH/dbt/env/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p $DIR/log $DIR/det
echo "detect $(hostname) $(date) dir $DIR workers ${WORKERS:-56} limit ${TL:-1680} s"
timeout ${TL:-1680} $PY -I $A/detect_pad.py $DIR/img $DIR/det --workers ${WORKERS:-56} >> $DIR/log/detect.out 2>&1
echo "end $(date); $(find $DIR/det -name 'c4d_*.npz' ! -name '*.tmp.*' | wc -l) detection lists"
