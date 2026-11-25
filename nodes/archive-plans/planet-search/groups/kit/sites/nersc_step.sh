#!/bin/bash
# Perlmutter side of kit/remote_run.sh (a copy lives in $SCRATCH/decam-groups/kit/). Two modes:
#   nersc_step.sh launch TAG SCEN_HASH BUDGET THREADS JOBID   on the login node: start the finder as a detached step of the held job
#   nersc_step.sh step   TAG SCEN_HASH BUDGET THREADS         on the compute node: run the finder the way the group job does
# The finder runs from a fresh copy in the node's /tmp, with THREADS threads on THREADS cores, no network when
# unprivileged namespaces work, killed 30 s after its budget. Writes runs/TAG/{out.json, finder.log, rc}.
RD=$NERSC_SCRATCH/decam-groups
PY=$NERSC_SCRATCH/rsi-planet/env/bin/python
mode=$1; tag=$2; sh=$3; budget=$4; thr=$5
d=$RD/runs/$tag
if [ "$mode" = launch ]; then
    cd "$d" || exit 2
    nohup setsid srun --jobid="$6" -N1 -n1 -c $((2 * thr)) --cpu-bind=cores \
        bash "$RD/kit/nersc_step.sh" step "$tag" "$sh" "$budget" "$thr" > "$d/srun.log" 2>&1 < /dev/null &
    exit 0
fi
tmp=$(mktemp -d "/tmp/dg_${tag}_XXXXXX")
cp -r "$d/program" "$tmp/sub"; mkdir "$tmp/home"
cd "$tmp/sub" || exit 2
ns=""; unshare -rn true 2>/dev/null && ns="unshare -rn"
t0=$(date +%s)
env -i PATH="$(dirname $PY):/usr/bin:/bin" HOME="$tmp/home" TMPDIR="$tmp/home" MPLCONFIGDIR="$tmp/home" \
    OMP_NUM_THREADS="$thr" MKL_NUM_THREADS="$thr" OPENBLAS_NUM_THREADS="$thr" CUDA_VISIBLE_DEVICES="" \
    PYTHONDONTWRITEBYTECODE=1 LD_LIBRARY_PATH="$(dirname "$(dirname $PY)")/lib" \
    timeout -s KILL $((budget + 30)) $ns $PY -I finder.py --scenario "$RD/scen/$sh" --out "$d/out.json" \
    --budget-s "$budget" > "$d/finder.log" 2>&1
rc=$?
echo "[remote] $(hostname) threads $thr budget $budget s, ran $(( $(date +%s) - t0 )) s, exit $rc, network $([ -n "$ns" ] && echo off || echo on)" >> "$d/finder.log"
rm -rf "$tmp"
echo $rc > "$d/rc"
