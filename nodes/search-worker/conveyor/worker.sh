#!/bin/bash
# conveyor worker (2026-11-18).
# Runs inside one allocation on any site (NERSC A100, Vista GH200 / GB200). Each lane = one GPU. A lane pulls the
# highest-priority task from ROOT/queue that nobody has claimed (mkdir ROOT/claims/<key> is the claim; the Mac
# coordinator mirrors claims between sites), whose scenario is on this site and whose budget fits before the wall,
# copies the program the coordinator marked current at that moment (ROOT/best/CURRENT -> ROOT/best/<gen>/sub), runs it
# with no network where the node allows it, and writes ROOT/out/<unit>/<sid>/candidates.json, ROOT/log/<unit>/<sid>.log
# and ROOT/timing/<unit>/<sid> ("seconds rc gen site device isolation"). A lane with nothing to do for IDLE_S seconds ends.
# Task file ROOT/queue/<prio>_<unit>_<sid>: one line "unit sid budget_s class" (class normal | big).
# Usage (inside an allocation): bash worker.sh ROOT SCEN PY SITE LANES BIGOK WALL_S [IDLE_S]
ROOT=$1; SCEN=$2; PY=$3; SITE=$4; LANES=$5; BIGOK=$6; WALL_S=$7; IDLE_S=${8:-900}
T0=$(date +%s); END=$(( T0 + WALL_S - 180 ))
JOB=${SLURM_JOB_ID:-local}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-16} MKL_NUM_THREADS=${MKL_NUM_THREADS:-16}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $ROOT/claims $ROOT/out $ROOT/log $ROOT/timing $ROOT/work $ROOT/workers
DEV=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr ' ' '_')
if unshare -rn true 2>/dev/null; then ISO="unshare -rn"; ISON=none-network; else ISO=""; ISON=open; fi
echo "worker $JOB site $SITE host $(hostname) lanes $LANES big $BIGOK wall $WALL_S device $DEV isolation $ISON $(date)"
echo "$(hostname) $SITE $LANES $DEV $(date +%s)" > $ROOT/workers/$JOB

lane() {
  local k=$1 idle0=$(date +%s)
  while true; do
    local now=$(date +%s) picked=""
    [ -f $ROOT/STOP ] && { echo "== lane $k: STOP"; return; }
    for f in $(ls $ROOT/queue 2>/dev/null | sort); do
      read -r unit sid budget cls < $ROOT/queue/$f 2>/dev/null || continue
      [ -z "$sid" ] && continue
      local key=${unit}_${sid}
      [ -e $ROOT/claims/$key ] && continue
      [ "$cls" = big ] && [ "$BIGOK" != 1 ] && continue
      [ -f $SCEN/$sid/exposures.json ] || continue
      [ $(( now + budget + 120 )) -le $END ] || continue
      mkdir $ROOT/claims/$key 2>/dev/null || continue
      echo "$SITE $JOB $k $(hostname) $now" > $ROOT/claims/$key/LOCAL
      picked="$unit $sid $budget"; break
    done
    if [ -z "$picked" ]; then
      [ $(( now - idle0 )) -ge $IDLE_S ] && { echo "== lane $k: nothing to do for $IDLE_S s, ends $(date)"; return; }
      [ $(( now + 600 )) -ge $END ] && { echo "== lane $k: wall near, ends"; return; }
      sleep 60; continue
    fi
    set -- $picked; local unit=$1 sid=$2 budget=$3
    local gen=$(cat $ROOT/best/CURRENT)
    local w=$ROOT/work/$JOB.$k; rm -rf $w; mkdir -p $w $ROOT/out/$unit $ROOT/log/$unit $ROOT/timing/$unit
    cp -r $ROOT/best/$gen/sub $w/sub
    rm -rf $ROOT/out/$unit/$sid; mkdir -p $ROOT/out/$unit/$sid
    local t0=$(date +%s.%N)
    local dev=${GPUS[$k]:-$k}
    ( cd $w/sub && CUDA_VISIBLE_DEVICES=$dev HOME=$w TMPDIR=$w timeout -k 10 $(( budget + 30 )) \
        $ISO $PY finder.py --scen $SCEN/$sid --out $ROOT/out/$unit/$sid --budget-s $budget ) > $ROOT/log/$unit/$sid.log 2>&1
    local rc=$?
    echo "$(awk -v a=$(date +%s.%N) -v b=$t0 'BEGIN{printf "%.3f", a-b}') $rc $gen $SITE $DEV $ISON" > $ROOT/timing/$unit/$sid
    rm -rf $w
    echo "== lane $k $unit $sid $gen rc=$rc $(cut -d' ' -f1 $ROOT/timing/$unit/$sid) s"
    idle0=$(date +%s)
  done
}
GPUS=(${CUDA_VISIBLE_DEVICES//,/ })     # the allocation's own GPU ids when Slurm sets them; lane k uses the k-th
for k in $(seq 0 $(( LANES - 1 ))); do lane $k & done
wait
rm -f $ROOT/workers/$JOB
echo "worker $JOB ends $(date)"
