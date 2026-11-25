#!/bin/bash
#SBATCH -J dbtgrade
#SBATCH -p gb
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o $TACC_SCRATCH/dbt/jobs/grade-%j.out
# decam-big-tno: run one submission on every sealed scenario of LIST on one Vista gb node, four lanes (one GB200 each).
# Each scenario gets a fresh copy of the submission, no network (unshare -rn), the budget BUDGET seconds, killed at
# BUDGET + 30 s. Outputs $RUN/out/<sid>/candidates.json, logs $RUN/log/<sid>.log, seconds in $RUN/timing/<sid>.
# Truth is never on this machine: grading happens on the Mac from the copied outputs.
# Usage: sbatch -A $VISTA_ACCOUNT -t HH:MM:SS run_gb.sbatch RUN BUDGET     (RUN = $SCRATCH/dbt/runs/<run>/<gen>, holding sub/ and list.txt)
module load gcc/13.2.0 cuda/12.8 python3/3.11.8
PY=$TACC_WORK/vista/dbt/env/bin/python
RUN=$1; BUDGET=$2; S=$TACC_SCRATCH/dbt/scen
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $RUN/out $RUN/log $RUN/timing $RUN/work
date; hostname; nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "cuda inside unshare: $(unshare -rn $PY -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())' 2>&1 | tail -1)"
lane() {
  local k=$1
  while read -r sid; do
    [ -f $RUN/timing/$sid ] && continue
    mkdir $RUN/work/$sid.lock 2>/dev/null || continue
    local w=$RUN/work/$sid; rm -rf $w; mkdir -p $w; cp -r $RUN/sub $w/sub
    rm -rf $RUN/out/$sid; mkdir -p $RUN/out/$sid
    local t0=$(date +%s.%N)
    ( cd $w/sub && CUDA_VISIBLE_DEVICES=$k HOME=$w TMPDIR=$w timeout -k 10 $(( BUDGET + 30 )) \
        unshare -rn $PY finder.py --scen $S/$sid --out $RUN/out/$sid --budget-s $BUDGET ) > $RUN/log/$sid.log 2>&1
    local rc=$?
    echo "$(echo "$(date +%s.%N) - $t0" | bc) $rc" > $RUN/timing/$sid
    rm -rf $w; rmdir $RUN/work/$sid.lock 2>/dev/null
    echo "== lane $k $sid rc=$rc $(cat $RUN/timing/$sid)"
  done < $RUN/list.txt
}
for k in 0 1 2 3; do lane $k & done
wait
touch $RUN/DONE
date
