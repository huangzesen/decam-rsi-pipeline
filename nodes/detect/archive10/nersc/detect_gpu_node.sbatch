#!/bin/bash
# archive10: single-exposure detection lists on one whole GPU node with detect_gpu.py (4 x A100, 64 cores; regular QOS,
# $NERSC_GPU_ACCOUNT, charged per node-hour). 2026-11-16.
# Measured 2026-11-16 (job <slurm-job-id>): 85 exposures/min per node at 64 workers (96 workers: the same, the node's 64 cores
# decompressing the files are the limit), against 27/min on a CPU node. So one node does about 5,000 exposures per hour.
# DIR holds img/<YYMMDD>/ (the ooi, oow, ood trios or links to them), det/ and log/. A rerun skips lists already written,
# so a job that hits its wall is simply submitted again. A CCD that cannot be read is skipped and named in the log.
# NOT TO BE USED unless asked. User, later on 2026-11-16, on learning that CPU and GPU node-hours are separate pools and the GPU
# pool is the scarce one (2,594 node-h left for 16 people): "Oh they billed differently!! Let's use cpu then". Detection runs on
# CPU nodes ($NERSC_CPU_ACCOUNT) with the stepA_detect.sbatch pattern; this script is kept only as a tested alternative.
# Submit from $A: sbatch --export=ALL,DIR=$A/<share> -t 02:00:00 detect_gpu_node.sbatch
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH --gpus-per-node=4
#SBATCH -q regular
#SBATCH -A $NERSC_GPU_ACCOUNT
#SBATCH -J a10dg
#SBATCH -o log/dg-%j.out
A=$NERSC_SCRATCH/archive10; PY=$NERSC_SCRATCH/dbt/env/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p $DIR/det $DIR/log
echo "gpu detect $(hostname) $(date) dir $DIR workers ${WORKERS:-64}"
$PY -I $A/detect_gpu.py $DIR/img $DIR/det --workers ${WORKERS:-64} > $DIR/log/detect_gpu_${SLURM_JOB_ID}.out 2>&1
echo "done $(date); $(find $DIR/det -name 'c4d_*.npz' | wc -l) detection lists; $(grep -c ERROR $DIR/det/detect.log) errors; $(grep -c unreadable $DIR/det/detect.log) exposures with an unreadable CCD"
