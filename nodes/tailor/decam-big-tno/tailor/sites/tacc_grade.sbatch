#!/bin/bash
#SBATCH -J dbthw
#SBATCH -n 1
# decam-big-tno hardware tailoring (tailor/, 2026-11-18): run one submission on the four graded regions on a TACC GPU
# partition (Vista gb / gh, Lonestar6 gpu-h100). Generalises harness/vista/run_gb.sbatch: LANES = nodes x GPUS_PER_NODE,
# one region per GPU at a time; lane j runs on node j / GPUS_PER_NODE with CUDA_VISIBLE_DEVICES = j % GPUS_PER_NODE.
# Each region gets a fresh copy of the submission, no network (unshare -rn; the job refuses to grade if a node cannot
# isolate), BUDGET seconds, killed at BUDGET + 30 s. Outputs ROOT/out/<sid>/candidates.json, logs ROOT/log/<sid>.log,
# seconds and exit code in ROOT/timing/<sid>, the hardware probe in ROOT/probe/<node>.txt. Truth is never on this
# machine: tailor.py scores the copied outputs on the Mac.
# Usage: sbatch -A ACCOUNT -p PARTITION -N NODES -t HH:MM:SS -o ROOT/job-%j.out tacc_grade.sbatch ROOT BUDGET GPUS_PER_NODE SCEN ENV "MODULES"
ROOT=$1; BUDGET=$2; GPN=$3; S=$4; ENV=$5; MODS=$6
set +u
[ -n "$MODS" ] && module load $MODS
PY=$ENV/bin/python
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $ROOT/out $ROOT/log $ROOT/timing $ROOT/work $ROOT/probe $ROOT/locks
NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST")); NN=${#NODES[@]}; LANES=$(( NN * GPN ))
date; echo "nodes ${NODES[*]}; $LANES lanes"

probe() {   # hardware facts for the agent's next round; also checks network isolation on the node
  {
    echo "host $(hostname)  date $(date -Is)"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap,clocks.max.sm,clocks.max.mem,power.limit --format=csv
    nvidia-smi topo -m 2>/dev/null | head -12
    lscpu | grep -E "Architecture|Model name|^CPU\(s\)|Thread|Core|Socket|NUMA node[0-9]* CPU|L3"
    free -g | head -2
    $PY -c "import sys, torch, numpy, scipy; print('python', sys.version.split()[0], 'torch', torch.__version__, 'cuda', torch.version.cuda, 'cudnn', torch.backends.cudnn.version(), 'numpy', numpy.__version__, 'scipy', scipy.__version__); [print('gpu', i, torch.cuda.get_device_name(i), 'capability', torch.cuda.get_device_capability(i), 'mem GB', round(torch.cuda.get_device_properties(i).total_memory/1e9,1), 'SMs', torch.cuda.get_device_properties(i).multi_processor_count) for i in range(torch.cuda.device_count())]" 2>&1
    g++ --version | head -1
    if unshare -rn true 2>/dev/null; then echo "isolation unshare -rn: ok"; else echo "isolation unshare -rn: FAILED"; fi
    echo "cuda inside unshare: $(unshare -rn $PY -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())' 2>&1 | tail -1)"
  } > $ROOT/probe/$(hostname).txt 2>&1
}
on() {   # run a command on node $1 (directly when the job has one node)
  local node=$1; shift
  if [ $NN -eq 1 ]; then bash -c "$*"; else srun -N 1 -n 1 -w $node --exact --overlap bash -c "$*"; fi
}
export -f probe; export ROOT PY
for n in "${NODES[@]}"; do on $n "$(declare -f probe); probe" & done; wait
if grep -l "isolation unshare -rn: FAILED" $ROOT/probe/*.txt >/dev/null 2>&1; then
  echo "a node cannot run without network (unshare -rn failed): not grading"; touch $ROOT/NO_ISOLATION; exit 3
fi

lane() {
  local j=$1 node=${NODES[$(( $1 / GPN ))]} gpu=$(( $1 % GPN ))
  while read -r sid; do
    [ -z "$sid" ] && continue
    [ -f $ROOT/timing/$sid ] && continue
    mkdir $ROOT/locks/$sid 2>/dev/null || continue
    local w=$ROOT/work/$sid; rm -rf $w; mkdir -p $w; cp -r $ROOT/sub $w/sub
    rm -rf $ROOT/out/$sid; mkdir -p $ROOT/out/$sid
    local t0=$(date +%s.%N)
    on $node "cd $w/sub && CUDA_VISIBLE_DEVICES=$gpu HOME=$w TMPDIR=$w timeout -k 10 $(( BUDGET + 30 )) unshare -rn $PY finder.py --scen $S/$sid --out $ROOT/out/$sid --budget-s $BUDGET" > $ROOT/log/$sid.log 2>&1
    local rc=$?
    echo "$(awk -v a=$(date +%s.%N) -v b=$t0 'BEGIN{printf "%.3f", a-b}') $rc $node $gpu" > $ROOT/timing/$sid
    rm -rf $w
    echo "== lane $j ($node gpu $gpu) $sid rc=$rc $(cat $ROOT/timing/$sid)"
  done < $ROOT/list.txt
}
for (( j=0; j<LANES; j++ )); do lane $j & done
wait
touch $ROOT/DONE
date
