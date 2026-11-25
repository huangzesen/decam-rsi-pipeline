#!/bin/bash
#SBATCH -J a5link
#SBATCH -p gb
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o $TACC_SCRATCH/archive5/link/link-%j.out
# archive5: link_gpu.py over a task list on one Vista gb node, four lanes (one per GB200), mkdir locks, resumable.
# Task line: TAG RID SEASON SPAN RMIN RMAX DGAMMA MINNIGHTS CORE_DEG [extra args]
#   near arm (one season, 30-80 au):   tag s<k>,   --rmin 30 --rmax 80 --dgamma 1e-4
#   far arm (two seasons, 80-1000 au): tag f<k>,   --rmin 80 --rmax 1000 --dgamma 1e-5 (sedna80's settings)
# Windows from prep_region.py in $SCRATCH/archive5/win; outputs $SCRATCH/archive5/runs/<TAG>_<RID>.{json,log,_links.npz}.
# Usage: sbatch -A $VISTA_ACCOUNT -t HH:MM:SS run_link.sbatch TASKS
module load gcc/13.2.0 cuda/12.8 python3/3.11.8
cd $WORK/linkbench/linker_s80
PY=$WORK/linkbench/venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TASKS=$1; D=$SCRATCH/archive5/win; O=$SCRATCH/archive5/runs; mkdir -p $O
COMMON="--fp32 --refine --polish --gdot-tol 4 --extend index"
LIM=$(squeue -h -j $SLURM_JOB_ID -o %L | awk -F'[:-]' '{if (NF==4) print $1*86400+$2*3600+$3*60+$4; else if (NF==3) print $1*3600+$2*60+$3; else print $1*60+$2}')
END=$(( $(date +%s) + LIM - 120 ))
date; hostname; nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader; echo "tasks $TASKS, $(wc -l < $TASKS) lines, ${LIM}s"
lane() {
  local k=$1
  while read -r tag rid sea span rmin rmax dg mn core extra; do
    local w=${tag}_${rid}
    [ -f $O/$w.json ] && continue
    mkdir $O/$w.lock 2>/dev/null || continue
    local left=$(( END - $(date +%s) ))
    if [ $left -lt 120 ]; then rmdir $O/$w.lock; break; fi
    local t0=$(date +%s)
    CUDA_VISIBLE_DEVICES=$k timeout $(( left + 60 )) $PY link_gpu.py $D/$rid.npz --season $sea --span $span --rmin $rmin --rmax $rmax \
      --dgamma $dg --minnights $mn --core $core $COMMON $extra --seed-mask $D/${rid}_seed.npz --budget $(( left - 30 )) \
      --save-links $O/${w}_links.npz --out $O/$w.json.part > $O/$w.log 2>&1
    local rc=$?
    if [ $rc -eq 0 ] && [ -f $O/$w.json.part ]; then mv $O/$w.json.part $O/$w.json; else echo "$w rc=$rc" >> $O/failed.txt; fi
    rmdir $O/$w.lock
    echo "== lane $k $w rc=$rc $(( $(date +%s) - t0 ))s"
  done < $TASKS
}
for k in 0 1 2 3; do lane $k & done
wait
date
