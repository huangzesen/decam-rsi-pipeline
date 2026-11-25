#!/bin/bash
# search10b (step B of the second 5 %; 2026-11-16): build the scenario of each of the other 20 archive10
# regions (R061 in parts) on one NERSC CPU node from the step B detection lists (archive10/sb_0..7/det) and, for the 3,983
# exposures shared with the first 5 % search, that search's lists (archive5/*/det). A copy of search10b's build10b.sbatch.
# 4 builds at once, not 5: a 200 M-row region took 81 GB in step A (job <slurm-job-id>) and the node has 512 GB.
# Scenarios -> $D/scen/<sid>/ (read-only), known-body labels -> $D/search10b/truth/ (copied to the Mac, then deleted here).
# Submit from $D/search10b: sbatch build10b.sbatch  (reads list.txt: "RID SID NEXP" lines; 4 builds at once)
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -A $NERSC_CPU_ACCOUNT
#SBATCH -t 02:00:00
#SBATCH -J dbt-build10b
#SBATCH -o log/build10b-%j.out
D=$NERSC_SCRATCH/dbt; A10=$NERSC_SCRATCH/archive10; PY=$D/env/bin/python; S=$D/search10b
mkdir -p $S/truth $S/log $D/scen
echo "node $(hostname) $(date)"
A5=$NERSC_SCRATCH/archive5
DETS="$(for k in 0 1 2 3 4 5 6 7; do printf "%s " $A10/sb_$k/det; done)$A5/n0/det $A5/n1/det $A5/n2/det $A5/n3/det $A5/vista/r38_0/det $A5/vista/r38_1/det $A5/vista/r38_2/det $A5/vista/r38_3/det $A5/vista/det"
build() {
  rid=$1; sid=$2
  [ -f $D/scen/$sid/exposures.json ] && { echo "$rid $sid exists"; return; }
  rm -rf $D/scen/$sid.part
  $PY -I $S/build_scen.py $rid $sid $S/in/regions_exposures.json $S/in/known/$rid.json $D/scen/$sid.part $S/truth $DETS \
      > $S/log/build_$rid.log 2>&1 && mv $D/scen/$sid.part $D/scen/$sid && chmod -R a-w $D/scen/$sid
  echo "$rid $sid rc=$? $(tail -1 $S/log/build_$rid.log | cut -c1-120)"
}
export -f build; export D S PY DETS
# largest first, so the long ones start early
sort -k3 -n -r $S/list.txt | xargs -P 4 -L 1 bash -c 'build $0 $1'
echo "done $(date)"
