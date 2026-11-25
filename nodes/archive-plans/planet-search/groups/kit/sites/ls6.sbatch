#!/bin/bash
# One group on Lonestar6 (3 A100 per node, 1 SU per A100-hour). Not yet run: check first that a compute node reaches
# astroarchive.noirlab.edu (EXTRA="--limit 1" on one node, 15 minutes).
# Usage: sbatch -J g099 -N 2 -t 10:00:00 --export=ALL,GID=g099,ROLE=validation groups/kit/sites/ls6.sbatch
#SBATCH -p gpu-a100
#SBATCH -A $LS6_ACCOUNT
#SBATCH --ntasks-per-node=1
#SBATCH -o slurm-%x-%j.out
set -euo pipefail
ROOT=${ROOT:-$SCRATCH/decam-groups}        # kit/, <id>/, _private/<id>/known.json, work/<id>/ (copied from planet-search/groups)
export PYTHON=${PYTHON:-$TACC_WORK/ls6/rsi-planet/env/bin/python}
export GROUP_TMP=/tmp
KNOWN=$ROOT/_private/$GID/known.json
srun -N "$SLURM_NNODES" --ntasks-per-node=1 "$ROOT/$GID/run.sh" "$ROOT/work/$GID" "${ROLE:-all}" \
     $( [ -f "$KNOWN" ] && echo --known "$KNOWN" ) ${EXTRA:-}
