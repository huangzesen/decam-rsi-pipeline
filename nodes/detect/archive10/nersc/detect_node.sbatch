#!/bin/bash
# archive10 on NERSC: single-exposure detection lists (detect.py, S/N >= 4.5) for one image folder on one whole CPU node
# (regular QOS, $NERSC_CPU_ACCOUNT group pool, charged per node-hour), 96 workers like the 5 % jobs. The shared-QOS quarter node
# (2026-11-14 12:42-13:52) lost 3 of 4 worker pools after ~1 h (BrokenProcessPool, most likely memory).
# Submit from $A: sbatch --export=ALL,DIR=<folder with img/ det/ log/> -t <wall> detect_node.sbatch
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -A $NERSC_CPU_ACCOUNT
#SBATCH -J a10det
#SBATCH -o log/detnode-%j.out
A=$NERSC_SCRATCH/archive10; PY=$NERSC_SCRATCH/dbt/env/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "detect $(hostname) $(date) dir $DIR"
$PY -I $A/detect.py $DIR/img $DIR/det --workers ${WORKERS:-96} ${WATCH:+--watch} > $DIR/log/detect_${SLURM_JOB_ID}.out 2>&1
echo "done $(date); $(ls $DIR/det/*/*.npz 2>/dev/null | wc -l) detection lists"
