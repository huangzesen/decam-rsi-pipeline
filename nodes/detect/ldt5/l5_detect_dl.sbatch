#!/bin/bash
# ldt5: detection lists for one download part (p<k>, filled by ldt5_fetch.sbatch) as its exposures arrive (--watch),
# else as l5_detect.sbatch. Submit: sbatch --export=ALL,DIR=$PWD/p0 l5_detect_dl.sbatch
# Original header: detection lists for one share (l5_<k>, made by l5_farm.py), a copy of
# archive15/s15_detect.sbatch on half a GPU node: NERSC's CPU nodes are all waiting for a rolling reboot
# (2026-11-18). detect_pad.py is CPU code; the job
# uses the node share's 64 CPUs; 16 workers at ~6.5 GB each (the share gets ~120 GB with its 64 CPUs). A rerun skips the lists already written.
# Submit from $NERSC_SCRATCH/ldt5:  sbatch --export=ALL,DIR=$PWD/l5_0 l5_detect.sbatch
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -A $NERSC_GPU_ACCOUNT
#SBATCH --gpus 2
#SBATCH -c 64
#SBATCH -t 06:00:00
#SBATCH -J l5w
#SBATCH -o log/l5w-%j.out
A=$NERSC_SCRATCH/archive10; PY=$NERSC_SCRATCH/dbt/env/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "detect $(hostname) $(date) dir $DIR workers ${WORKERS:-16}"
( while [ ! -f $DIR/log/SHARE_DONE ]; do sleep 60; done; touch $DIR/log/FETCH_DONE ) &   # detect_pad --watch ends at FETCH_DONE; ldt5_fetch.sbatch writes SHARE_DONE
$PY -I $A/detect_pad.py $DIR/img $DIR/det --workers ${WORKERS:-16} --watch > $DIR/log/detect_${SLURM_JOB_ID}.out 2>&1
echo "done $(date); $(find $DIR/det -name 'c4d_*.npz' | wc -l) detection lists; $(grep -c ERROR $DIR/det/detect.log) errors; $(grep -c unreadable $DIR/det/detect.log) exposures with an unreadable CCD"
