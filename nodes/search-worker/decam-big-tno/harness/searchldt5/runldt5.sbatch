#!/bin/bash
# searchldt5: run one program over LDT 5 % regions on a shared GPU node share (NERSC's regular GPU nodes are waiting for
# a rolling reboot): lanes = GPUs of the job (1 A100 each), node38.sh's lane logic (lock directories, a wall guard,
# no network, BUDGET seconds per region). ROOT/list.txt holds "TAG SID" lines, ROOT/TAG/sub the program.
# Submit from $D/searchldt5: sbatch --gpus G -c 32*G --export=ALL,ROOT=<dir>,BUDGET=<s>,WALL_S=<s> -t <wall> runldt5.sbatch
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -A $NERSC_GPU_ACCOUNT
#SBATCH -J dbt-runl5
#SBATCH -o log/runl5-%j.out
D=$NERSC_SCRATCH/dbt; S=$D/scen; PY=$D/env/bin/python
NG=$(nvidia-smi -L | wc -l)
W0=$(date +%s); while [ ! -f $D/searchldt5/BUILT ]; do   # scenarios are built after detection (buildldt5.sbatch)
  [ $(( $(date +%s) - W0 )) -gt ${WAIT_S:-5400} ] && { echo "no scenarios after ${WAIT_S:-5400} s; exit"; exit 0; }; sleep 60; done
T0=$(date +%s); LAST=$(( T0 + WALL_S - BUDGET - 120 ))
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "node $(hostname) $(date) root $ROOT budget $BUDGET wall_s $WALL_S gpus $NG"; nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "cuda inside unshare: $(unshare -rn $PY -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())' 2>&1 | tail -1)"
mkdir -p $ROOT/locks
lane() {
  local k=$1
  while read -r tag sid; do
    [ -z "$sid" ] && continue
    [ $(date +%s) -le $LAST ] || { echo "== lane $k stops: no time for another region"; break; }
    mkdir -p $ROOT/$tag/out $ROOT/$tag/log $ROOT/$tag/timing $ROOT/$tag/work
    [ -f $ROOT/$tag/timing/$sid ] && continue
    [ -f $S/$sid/exposures.json ] || continue
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
for k in $(seq 0 $(( ${LANES:-$NG} - 1 ))); do lane $k & done   # LANES=1 on a 2-GPU share gives one region ~112 GB
wait
date
