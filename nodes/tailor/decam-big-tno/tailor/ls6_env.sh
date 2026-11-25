#!/bin/bash
# Build the decam-big-tno grading environment on Lonestar6 (tailor/, 2026-11-18): the same packages as NERSC's
# $SCRATCH/dbt/env (tailor/nersc_env_freeze.txt, read 2026-11-18: python 3.11, torch 2.11.0+cu128, triton 3.6.0,
# numpy 2.4.6, scipy 1.17.1, astropy 8.0.2, scikit-learn 1.9.1), at $TACC_WORK/ls6/dbt/env.
# Lonestar6 has no python3.11 module (python3/3.9.7 only), so uv provides a standalone CPython 3.11. The existing
# $WORK/rsi-planet/env (py 3.12, numpy 1.26, no scikit-learn) belongs to planet-search and is not reused.
# NOT RUN YET. Run on a Lonestar6 login node (it reaches PyPI and download.pytorch.org), from the Mac:
#   scp tailor/ls6_env.sh tailor/nersc_env_freeze.txt ls6:$TACC_WORK/ls6/dbt/  (via the SSH master)
#   ssh ls6 'bash $TACC_WORK/ls6/dbt/ls6_env.sh'
# Then check on a gpu-h100 node (the grader's probe does this): torch.cuda.is_available() inside `unshare -rn`.
set -euo pipefail
D=$TACC_WORK/ls6/dbt
ENV=$D/env
mkdir -p $D/bin
if ! command -v uv >/dev/null 2>&1 && [ ! -x $D/bin/uv ]; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=$D/bin UV_NO_MODIFY_PATH=1 sh
fi
UV=$(command -v uv || echo $D/bin/uv)
export UV_PYTHON_INSTALL_DIR=$D/python UV_CACHE_DIR=$D/uv-cache
$UV python install 3.11
[ -d $ENV ] && { echo "$ENV exists: remove it first to rebuild"; exit 1; }
$UV venv --python 3.11 $ENV
# Pinned like NERSC; torch from the cu128 index (its wheels bundle the CUDA 12.8 runtime; the H100 driver must be >= 12.8).
grep -v -E '^(pip|setuptools|wheel|nersc-pymon|torch|triton)==' $D/nersc_env_freeze.txt > $D/req_nersc.txt
$UV pip install --python $ENV/bin/python --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0+cu128 triton==3.6.0
$UV pip install --python $ENV/bin/python -r $D/req_nersc.txt
$ENV/bin/python -c "import sys, torch, numpy, scipy, astropy, sklearn; print(sys.version.split()[0], torch.__version__, torch.version.cuda, numpy.__version__, scipy.__version__, astropy.__version__, sklearn.__version__)"
$ENV/bin/python -m pip list --format=freeze > $D/env_freeze_ls6.txt 2>/dev/null || $UV pip freeze --python $ENV/bin/python > $D/env_freeze_ls6.txt
echo "built $ENV"
