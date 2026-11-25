#!/bin/bash
# Build the KBMOD environment for the planet-finder baseline on NERSC Perlmutter (login node; no GPU needed to compile).
# Self-contained conda env (its own libstdc++/libgomp in env_kbmod/lib, which finder_node.py puts on LD_LIBRARY_PATH).
source /etc/profile >/dev/null 2>&1
B=$NERSC_SCRATCH/rsi-planet
E=$B/env_kbmod
COMMIT=60b8c8216318ccd633eb9be05e8545c292ee27fa      # dirac-institute/kbmod main, 2026-11-02 (no release tag after v1.1.0 of 2023)
module load conda
module load cudatoolkit/12.9
module load gcc-native/13
set -eo pipefail
which nvcc gcc; nvcc --version | tail -2; gcc --version | head -1
cd $B/kbmod_build/src
git checkout -q $COMMIT
git submodule update --init --recursive
git log -1 --format='%H %cd %s'
if [ ! -x $E/bin/python ]; then
  conda create -y -p $E -c conda-forge --override-channels python=3.12 pip "numpy<2.3" scipy astropy pandas h5py numba \
     scikit-learn shapely pyarrow reproject joblib jplephem psutil tqdm pyyaml matplotlib-base libstdcxx libgomp
fi
$E/bin/python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
export CC=$(which gcc) CXX=$(which g++) CUDACXX=$(which nvcc)
$E/bin/python -m pip install -v . 2>&1 | tail -40
cd $B
$E/bin/python -c "import kbmod, kbmod.search as s; print('kbmod', kbmod.__version__, 'gpu', s.HAS_GPU if hasattr(s,'HAS_GPU') else '?')"
ls $E/lib/python3.12/site-packages/kbmod/ | head; find $E/lib/python3.12/site-packages -maxdepth 2 -name "search*.so" -exec ldd {} \;
echo BUILD_DONE
