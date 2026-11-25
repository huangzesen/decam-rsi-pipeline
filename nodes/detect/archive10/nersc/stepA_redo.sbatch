#!/bin/bash
# Step A: second pass over the three shares with detect_pad.py, for the exposures whose weight map is 2 pixels larger
# than the image (CP V4.9 in the Legacy Surveys ls9 files; detect.py fails on them with a shape error). detect_pad.py
# is detect.py plus a trim of that one-pixel zero border. A rerun skips every exposure that already has a list, so
# this only does the failed ones (about 3 % of the exposures, a few minutes each share).
# Submit from $NERSC_SCRATCH/archive10 after the stepA_detect jobs have ended:  sbatch stepA_redo.sbatch
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -A $NERSC_CPU_ACCOUNT
#SBATCH -t 00:45:00
#SBATCH -J a10sr
#SBATCH -o log/sr-%j.out
A=$NERSC_SCRATCH/archive10; PY=$NERSC_SCRATCH/dbt/env/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for k in 0 1 2; do
  echo "redo sa_$k $(hostname) $(date)"
  $PY -I $A/detect_pad.py $A/sa_$k/img $A/sa_$k/det --workers 56 > $A/sa_$k/log/redo_${SLURM_JOB_ID}.out 2>&1
  echo "sa_$k: $(find $A/sa_$k/det -name '*.npz' | wc -l) detection lists"
done
echo "done $(date)"
