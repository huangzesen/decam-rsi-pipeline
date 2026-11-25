#!/bin/bash
# decam-big-tno: run submissions on the sealed regions on one Perlmutter GPU node (4 x A100 40 GB), four lanes, one A100
# each. ROOT/list.txt holds "TAG SID" lines; ROOT/TAG/sub is that submission. Each (TAG, SID) gets a fresh copy of the
# submission, no network (unshare -rn), BUDGET seconds, killed at BUDGET + 30 s. Outputs ROOT/TAG/out/SID/candidates.json,
# logs ROOT/TAG/log/SID.log, seconds and exit code in ROOT/TAG/timing/SID. Truth is never on NERSC: scoring is on the Mac.
# Usage (inside an allocation): srun -N 1 -n 1 bash grade_node.sh ROOT BUDGET
ROOT=$1; BUDGET=$2; D=$NERSC_SCRATCH/dbt; S=$D/scen; PY=$D/env/bin/python
# python: the conda env at $D/env (no module needed)
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
date; hostname; nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "cuda inside unshare: $(unshare -rn $PY -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())' 2>&1 | tail -1)"
mkdir -p $ROOT/locks
lane() {
  local k=$1
  while read -r tag sid; do
    [ -z "$sid" ] && continue
    mkdir -p $ROOT/$tag/out $ROOT/$tag/log $ROOT/$tag/timing $ROOT/$tag/work
    [ -f $ROOT/$tag/timing/$sid ] && continue
    mkdir $ROOT/locks/$tag.$sid 2>/dev/null || continue
    local w=$ROOT/$tag/work/$sid; rm -rf $w; mkdir -p $w; cp -r $ROOT/$tag/sub $w/sub
    rm -rf $ROOT/$tag/out/$sid; mkdir -p $ROOT/$tag/out/$sid
    local t0=$(date +%s.%N)
    ( cd $w/sub && CUDA_VISIBLE_DEVICES=$k HOME=$w TMPDIR=$w timeout -k 10 $(( BUDGET + 30 )) \
        unshare -rn $PY finder.py --scen $S/$sid --out $ROOT/$tag/out/$sid --budget-s $BUDGET ) > $ROOT/$tag/log/$sid.log 2>&1
    local rc=$?
    echo "$(awk -v a=$(date +%s.%N) -v b=$t0 'BEGIN{printf "%.3f", a-b}') $rc" > $ROOT/$tag/timing/$sid
    rm -rf $w
    echo "== lane $k $tag $sid rc=$rc $(cat $ROOT/$tag/timing/$sid)"
  done < $ROOT/list.txt
}
for k in 0 1 2 3; do lane $k & done
wait
touch $ROOT/DONE
date
