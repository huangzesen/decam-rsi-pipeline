#!/bin/bash
# Slices of one group on one Perlmutter GPU node (4 A100; 1 node-hour per node-hour, account $NERSC_GPU_ACCOUNT).
#   sbatch -J g099-017 -q regular -t 02:30:00 --export=ALL,GID=g099,SLICES=g099-017 groups/kit/sites/nersc_slices.sbatch
# AB=1 also runs the unchanged reference program on the same images (kept between the two runs, deleted after), into
# work_ref/<group>: the one-off comparison of a group's adapted program with the reference on real images.
# Inside a held allocation (salloc --no-shell), run it as a step instead:
#   srun --jobid=J -N1 -n1 --gpus=4 --export=ALL,NOSRUN=1,GID=g099,SLICES=g099-017,AB=1 bash kit/sites/nersc_slices.sbatch
# It writes logs/<slices>.done (exit codes of the runs) when it ends.
# The job fetches its own images (a compute node reached the archive at 150 MB/s with 32 streams, job <slurm-job-id>).
#SBATCH -A $NERSC_GPU_ACCOUNT
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -o slurm-%x-%j.out
set -uo pipefail
ROOT=${ROOT:-$SCRATCH/decam-groups}
export PYTHON=${PYTHON:-$NERSC_SCRATCH/rsi-planet/env/bin/python}
export GROUP_TMP=${TMPDIR:-/tmp}
KNOWN=$ROOT/_private/$GID/known.json
W=$ROOT/work/$GID; WR=$ROOT/work_ref/$GID
K=""; [ -f "$KNOWN" ] && K="--known $KNOWN"
RUN="srun -N1 -n1"; [ -n "${NOSRUN:-}" ] && RUN=""
mkdir -p "$ROOT/logs"; rc2=-
if [ "${AB:-0}" = 1 ]; then
    $RUN "$ROOT/$GID/run.sh" "$W" all --slices "$SLICES" $K --keep-images ${EXTRA:-}; rc1=$?
    IDX=$W/local_index_$SLURM_JOB_ID.txt
    find "$W/data" -name '*.fits.fz' > "$IDX"
    $RUN "$ROOT/$GID/run.sh" "$WR" all --slices "$SLICES" $K --program "$ROOT/kit/reference" --local-index "$IDX" ${EXTRA:-}; rc2=$?
    "$PYTHON" -I - "$ROOT/$GID" "$W/data" "$SLICES" <<'EOF'
import json, os, shutil, sys
g, data, slices = sys.argv[1], sys.argv[2], set(sys.argv[3].split(","))
for w in json.load(open(os.path.join(g, "windows.json")))["windows"]:
    if w.get("slice") in slices:
        shutil.rmtree(os.path.join(data, str(w["key"])), ignore_errors=True)
EOF
else
    $RUN "$ROOT/$GID/run.sh" "$W" all --slices "$SLICES" $K ${EXTRA:-}; rc1=$?
fi
echo "$(date '+%F %T') $SLICES program $rc1 reference $rc2 host $(hostname)" > "$ROOT/logs/${SLICES//,/_}.done"
