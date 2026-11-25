#!/bin/bash
# archive10 on NERSC: single-exposure detection (detect.py --watch, S/N >= 4.5 lists) for the four download folders
# a10n_0-3, on a quarter of a CPU node (shared QOS, 32 cores, charged 0.25 node-h per hour to the group pool $NERSC_CPU_ACCOUNT).
# Each watcher ends when its folder's log/FETCH_DONE exists and every complete exposure has a list.
# Submit from $A: sbatch detect.sbatch
#SBATCH -q shared
#SBATCH -C cpu
#SBATCH -A $NERSC_CPU_ACCOUNT
#SBATCH -c 32
#SBATCH -t 48:00:00
#SBATCH -J a10d
#SBATCH -o log/detect-%j.out
A=$NERSC_SCRATCH/archive10; PY=$NERSC_SCRATCH/dbt/env/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "detect $(hostname) $(date)"
for k in 0 1 2 3; do
  $PY -I $A/detect.py $A/a10n_$k/img $A/a10n_$k/det --workers 8 --watch > $A/a10n_$k/log/detect_${SLURM_JOB_ID}.out 2>&1 &
done
wait
echo "done $(date); $(ls $A/a10n_*/det/*/*.npz 2>/dev/null | wc -l) detection lists"
