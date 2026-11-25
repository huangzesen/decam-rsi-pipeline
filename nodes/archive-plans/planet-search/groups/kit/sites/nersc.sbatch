#!/bin/bash
# One group on Perlmutter (4 A100 per node, 1 node-hour per node-hour). A compute node reached the archive at 150 MB/s
# with 32 streams (job <slurm-job-id>). Exposures of Legacy Surveys DR11 are already on disk under
# /global/cfs/cdirs/cosmo/staging/decam/: list them once into a text file and pass EXTRA="--local-index FILE".
# Usage: sbatch -J g099 -N 1 -t 12:00:00 -q regular --export=ALL,GID=g099,ROLE=validation groups/kit/sites/nersc.sbatch
#SBATCH -A $NERSC_GPU_ACCOUNT
#SBATCH -C gpu
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -o slurm-%x-%j.out
set -euo pipefail
ROOT=${ROOT:-$SCRATCH/decam-groups}
export PYTHON=${PYTHON:-$NERSC_SCRATCH/rsi-planet/env/bin/python}
export GROUP_TMP=${TMPDIR:-/tmp}
KNOWN=$ROOT/_private/$GID/known.json
srun -N "$SLURM_NNODES" --ntasks-per-node=1 "$ROOT/$GID/run.sh" "$ROOT/work/$GID" "${ROLE:-all}" \
     $( [ -f "$KNOWN" ] && echo --known "$KNOWN" ) ${EXTRA:-}
