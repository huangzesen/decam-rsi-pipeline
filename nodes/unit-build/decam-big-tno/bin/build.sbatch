#!/bin/bash
# decam-big-tno: build scenarios on one Vista gg node: build_scen.py for each "RID SID" pair in $PAIRS (comma list of RID:SID).
# Scenarios -> $SCRATCH/dbt/scen/<sid>/ (read-only), truth -> $SCRATCH/dbt/truth_out/ (copied to the Mac, then deleted here).
# Also records whether `unshare -rn` (no network for the graded program) works on a compute node.
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J dbtbuild
#SBATCH -o $TACC_SCRATCH/dbt/jobs/build-%j.out
module purge >/dev/null 2>&1; module load gcc/14 cuda/12 python3/3.11.8
PY=$TACC_WORK/vista/archive5/env/bin/python
A5=$TACC_SCRATCH/archive5; D=$TACC_SCRATCH/dbt
mkdir -p $D/scen $D/truth_out
echo "node $(hostname) $(date)"
unshare -rn bash -c 'curl -s -m 5 -o /dev/null -w "%{http_code}" https://astroarchive.noirlab.edu/ || echo " no network"' 2>&1 | sed 's/^/unshare -rn: /'
for pr in ${PAIRS//,/ }; do
  rid=${pr%%:*}; sid=${pr##*:}
  $PY -I $D/bin/build_scen.py $rid $sid $A5/link/regions_exposures.json $A5/link/known/$rid.json $D/scen/$sid $D/truth_out \
      $A5/det $A5/n2/det && chmod -R a-w $D/scen/$sid
done
echo "done $(date)"
