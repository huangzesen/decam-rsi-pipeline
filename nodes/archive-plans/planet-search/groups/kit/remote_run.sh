#!/bin/bash
# Run one finder test on the held Perlmutter CPU node instead of the Mac (the Mac is shared and slow).
#
#   groups/kit/remote_run.sh PROGRAM_DIR SCENARIO_DIR OUT.json [BUDGET_S=900] [THREADS=8]
#
# Copies PROGRAM_DIR and SCENARIO_DIR (links followed; a scenario is sent once, later runs send only changes) to
# $SCRATCH/decam-groups, runs   python -I finder.py --scenario SCEN --out OUT --budget-s BUDGET_S   there on THREADS
# cores (CPU only: no GPU on that node), waits, and copies back OUT.json and OUT.json.log (the finder's stdout and
# stderr, plus one "[remote]" line with the run time). Several runs can go at once; each is its own step on the node.
# Exit code: the finder's (137 = killed 30 s after its budget); 3 = no node is held, run on the Mac as before;
# 4 = copy failed. The job id of the held node is in kit/remote_job.txt (the session that holds it writes it).
set -u
export LC_ALL=C LANG=C
[ $# -ge 3 ] || { sed -n 2,12p "$0"; exit 2; }
KIT=$(cd "$(dirname "$0")" && pwd)
SOCK=$SSH_CM_DIR/nersc-$NERSC_USER.sock
HOST=$NERSC_USER@perlmutter.nersc.gov
RD=$NERSC_SCRATCH/decam-groups
prog=$(cd "$1" && pwd -P) || exit 2
scen=$(cd "$2" && pwd -P) || exit 2
out=$3; budget=${4:-900}; thr=${5:-8}
[ -f "$prog/finder.py" ] || { echo "remote_run: no finder.py in $prog"; exit 2; }
[ -f "$scen/exposures.csv" ] || { echo "remote_run: no exposures.csv in $scen"; exit 2; }
[ "$thr" -le 16 ] || thr=16
job=$(cat "$KIT/remote_job.txt" 2>/dev/null)
RSH="ssh -S $SOCK -o BatchMode=yes"
r() {   # a remote command through the shared login; retried when the login refuses one more session
    local i rc
    for i in 1 2 3 4 5 6 7 8; do
        $RSH "$HOST" "$@"; rc=$?
        [ $rc -ne 255 ] && return $rc
        sleep $((4 * i))
    done
    return 255
}
push() {   # rsync with retries
    local i
    for i in 1 2 3 4; do rsync -aL --exclude __pycache__ --exclude '*.pyc' -e "$RSH" "$@" && return 0; sleep $((5 * i)); done
    return 1
}
[ -n "$job" ] && [ "$(r "squeue -h -j $job -o %T" 2>/dev/null)" = RUNNING ] || { echo "remote_run: no node is held; run on the Mac"; exit 3; }
sh=$(printf %s "$scen" | shasum | cut -c1-12)
tag=$(printf %s "$prog" | shasum | cut -c1-6)_$(date +%H%M%S)_$$
lock=$KIT/.remote_locks/$sh
mkdir -p "$KIT/.remote_locks"
n=0; until mkdir "$lock" 2>/dev/null; do n=$((n + 1)); [ $n -gt 360 ] && break; sleep 5; done   # one upload of a scenario at a time
trap 'rmdir "$lock" 2>/dev/null' EXIT
t0=$(date +%s)
r "mkdir -p $RD/scen/$sh $RD/runs/$tag/program" || exit 4
push "$scen/" "$HOST:$RD/scen/$sh/" || { echo "remote_run: scenario copy failed"; exit 4; }
rmdir "$lock" 2>/dev/null; trap - EXIT
push "$prog/" "$HOST:$RD/runs/$tag/program/" || { echo "remote_run: program copy failed"; exit 4; }
t1=$(date +%s)
r "bash $RD/kit/nersc_step.sh launch $tag $sh $budget $thr $job" || exit 4
echo "remote_run: $tag started (copy $((t1 - t0)) s); scenario $RD/scen/$sh"
limit=$((budget + 1800))
while ! r "test -f $RD/runs/$tag/rc"; do
    [ $(( $(date +%s) - t1 )) -gt $limit ] && { echo "remote_run: gave up after $limit s (step log: $RD/runs/$tag/srun.log)"; exit 4; }
    [ "$(r "squeue -h -j $job -o %T" 2>/dev/null)" = RUNNING ] || { r "test -f $RD/runs/$tag/rc" || { echo "remote_run: the node was released before the run ended"; exit 3; }; }
    sleep 15
done
rc=$(r "cat $RD/runs/$tag/rc")
mkdir -p "$(dirname "$out")"
rsync -a -e "$RSH" "$HOST:$RD/runs/$tag/finder.log" "$out.log" 2>/dev/null
rsync -a -e "$RSH" "$HOST:$RD/runs/$tag/out.json" "$out" 2>/dev/null || echo "remote_run: the finder wrote no output"
r "rm -rf $RD/runs/$tag/program"
tail -3 "$out.log" 2>/dev/null
echo "remote_run: exit $rc, $(( $(date +%s) - t0 )) s in all; log $out.log"
exit "${rc:-4}"
