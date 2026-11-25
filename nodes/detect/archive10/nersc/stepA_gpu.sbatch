#!/bin/bash
# archive10 step A: the GPU version of the detection (detect_gpu.py; 2026-11-16) on one
# whole GPU node (4 x A100, 64 cores, debug QOS, $NERSC_GPU_ACCOUNT, charged per node-hour).
#  1. the 86 exposures whose weight map is 2 px larger than the image (stepA/redo_img, links) -> sa_redo/det. This is the
#     second pass of step A, in place of the CPU job stepA_redo.sbatch.
#  2. a rate and agreement test: 320 exposures that the CPU jobs already did (stepA/gput_img_a, links) with 64 workers,
#     320 others (gput_img_b) with 96 workers, lists to stepA/gput_det_a|b, then cmp_lists.py against the CPU lists.
# Submit from $A: sbatch stepA_gpu.sbatch
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH --gpus-per-node=4
#SBATCH -q debug
#SBATCH -A $NERSC_GPU_ACCOUNT
#SBATCH -t 00:28:00
#SBATCH -J a10sg
#SBATCH -o log/sg-%j.out
A=$NERSC_SCRATCH/archive10; PY=$NERSC_SCRATCH/dbt/env/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "gpu detect $(hostname) $(date)"; nvidia-smi -L; nproc; free -g | sed -n 2p
( while true; do echo "$(date +%H:%M:%S) $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr '\n' ';') load $(cut -d' ' -f1 /proc/loadavg) free $(free -g | awk 'NR==2{print $7}') GB"; sleep 15; done ) > $A/stepA/gpu_smi_${SLURM_JOB_ID}.txt 2>&1 &
SMI=$!
run() {   # name img out workers
  t0=$(date +%s)
  $PY -I $A/detect_gpu.py $2 $3 --workers $4 > $A/stepA/gpu_${1}_${SLURM_JOB_ID}.out 2>&1
  t1=$(date +%s); n=$(find $3 -name 'c4d_*.npz' | wc -l)
  echo "$1: $n lists in $((t1 - t0)) s with $4 workers; $(grep -c ERROR $3/detect.log) errors; $(tail -1 $A/stepA/gpu_${1}_${SLURM_JOB_ID}.out)"
}
mkdir -p $A/sa_redo/det
run redo $A/stepA/redo_img $A/sa_redo/det 64
run a64 $A/stepA/gput_img_a $A/stepA/gput_det_a 64
run b96 $A/stepA/gput_img_b $A/stepA/gput_det_b 96
kill $SMI
echo "agreement with the CPU lists, set a:"; $PY -I $A/stepA/cmp_lists.py $A/sa_0/det $A/stepA/gput_det_a
echo "set b:"; $PY -I $A/stepA/cmp_lists.py $A/sa_1/det $A/stepA/gput_det_b
echo "redo against the CPU test of c4d_150225_055441:"; $PY -I $A/stepA/cmp_lists.py $A/stepA/test_pad $A/sa_redo/det
echo "done $(date)"
