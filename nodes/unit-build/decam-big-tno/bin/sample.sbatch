#!/bin/bash
# decam-big-tno: build the agents' unlabelled practice sample on one Vista gg node: region R005 without any exposure that
# belongs to a sealed region (REG_A, REG_B, REG_C). No truth is kept: the builder's truth file goes to a scratch folder
# that is deleted at the end. Output $SCRATCH/dbt/sample_build/sample/ (copied to harbor/.../environment/data/sample).
# Optional BOX="RA0 RA1 DEC0 DEC1" keeps exposures whose pointing centre is inside.
# Usage: sbatch -A $VISTA_ACCOUNT -p gg -t 00:20:00 sample.sbatch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J dbtsample
#SBATCH -o $TACC_SCRATCH/dbt/jobs/sample-%j.out
module purge >/dev/null 2>&1; module load gcc/14 cuda/12 python3/3.11.8
PY=$TACC_WORK/vista/archive5/env/bin/python
A5=$TACC_SCRATCH/archive5; D=$TACC_SCRATCH/dbt; B=$D/sample_build
rm -rf $B; mkdir -p $B/truth_tmp
echo "node $(hostname) $(date)"
$PY -I $D/bin/build_scen.py R005 sample $A5/link/regions_exposures.json $B/no_known.json $B/sample $B/truth_tmp \
    $A5/det $A5/n2/det --exclude-rid REG_A,REG_B,REG_C ${BOX:+--box $BOX}
rm -rf $B/truth_tmp
du -sh $B/sample
echo "done $(date)"
