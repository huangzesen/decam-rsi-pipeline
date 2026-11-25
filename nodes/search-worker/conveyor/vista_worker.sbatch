#!/bin/bash
# conveyor worker on TACC Vista: gh (1 GH200, H100-class, 72 Grace cores) or gb (4 GB200, 144 Grace cores, 960 GB).
# 1 SU per node-hour on either (15-minute minimum); the worker ends when it has had nothing to do for IDLE_S.
#   sbatch -p gh -t 12:00:00 --export=ALL,WALL_S=43200 vista_worker.sbatch
#   sbatch -p gb -t 12:00:00 --export=ALL,WALL_S=43200 vista_worker.sbatch
#SBATCH -N 1
#SBATCH -A $VISTA_ACCOUNT
#SBATCH -J cvw
#SBATCH -o $TACC_SCRATCH/conveyor/slurm/w-%j.out
R=$TACC_SCRATCH/conveyor
module load gcc/13.2.0 cuda/12.8 python3/3.11.8
L=$(nvidia-smi -L | wc -l); C=$(nproc)
export OMP_NUM_THREADS=$(( C / L - 4 )) OPENBLAS_NUM_THREADS=$(( C / L - 4 )) MKL_NUM_THREADS=$(( C / L - 4 ))
bash $R/worker.sh $R $TACC_SCRATCH/dbt/scen $TACC_WORK/vista/dbt/env/bin/python \
     vista-$SLURM_JOB_PARTITION $L 1 ${WALL_S:-43200} ${IDLE_S:-900}
