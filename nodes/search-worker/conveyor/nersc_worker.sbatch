#!/bin/bash
# conveyor worker on NERSC Perlmutter (A100 40 GB). Queue, GPUs and wall come from the coordinator's sbatch line:
#   shared, 1 GPU:  sbatch -q shared --gpus-per-node=1 -c 32 -t 08:00:00 --export=ALL,NGPU=1,LANES=1,BIGOK=0,WALL_S=28800 nersc_worker.sbatch
#   shared, 2 GPUs: sbatch -q shared --gpus-per-node=2 -c 64 -t 08:00:00 --export=ALL,NGPU=2,LANES=1,BIGOK=1,WALL_S=28800 nersc_worker.sbatch
#     (one lane with two GPUs' host memory, ~112 GB, for regions over 1,000 exposures)
#   regular node:   sbatch -q regular -N 1 --gpus-per-node=4 -t 08:00:00 --export=ALL,NGPU=4,LANES=4,BIGOK=1,WALL_S=28800 nersc_worker.sbatch
# The GPUs must be asked for in both sbatch and srun (--gpus-per-node), or the step sees none (tested 2026-11-18).
#SBATCH -C gpu
#SBATCH -A $NERSC_GPU_ACCOUNT
#SBATCH -J cvw
#SBATCH -o $NERSC_SCRATCH/conveyor/slurm/w-%j.out
R=$NERSC_SCRATCH/conveyor
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
srun -N 1 -n 1 --cpu-bind=none --gpus-per-node=${NGPU:-1} bash $R/worker.sh $R $NERSC_SCRATCH/dbt/scen $NERSC_SCRATCH/dbt/env/bin/python \
     nersc ${LANES:-1} ${BIGOK:-0} ${WALL_S:-28800} ${IDLE_S:-900}
