#!/bin/bash
# archive5: prep_region.py for each region in $RIDS (comma list) on one Vista gg node: detection lists from both pilot
# folders -> $SCRATCH/archive5/win/<rid>.npz (linker format, static sources removed), with known-body labels.
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J a5prep
#SBATCH -o $TACC_SCRATCH/archive5/link/prep-%j.out
module purge >/dev/null 2>&1; module load gcc/14 cuda/12 python3/3.11.8
PY=$TACC_WORK/vista/archive5/env/bin/python
D=$TACC_SCRATCH/archive5
cd $D/link || exit 1
for r in ${RIDS//,/ }; do
  K=""; [ -f known/$r.json ] && K="--known known/$r.json"
  $PY -I prep_region.py $r regions_exposures.json $D/win $D/det $D/n2/det $K
done
