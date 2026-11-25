#!/bin/bash
# search10a (step A of the second 5 %; 2026-11-16): build the scenario of each of the five archive10
# regions on one NERSC CPU node from the step A detection lists (archive10/sa_0..2/det). A copy of search38's build38.sbatch.
# Scenarios -> $D/scen/<sid>/ (read-only), known-body labels -> $D/search10a/truth/ (copied to the Mac, then deleted here).
# Submit from $D/search10a: sbatch build10a.sbatch  (reads list.txt: "RID SID NEXP" lines; 5 builds at once)
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -A $NERSC_CPU_ACCOUNT
#SBATCH -t 02:00:00
#SBATCH -J dbt-build10a
#SBATCH -o log/build10a-%j.out
D=$NERSC_SCRATCH/dbt; A10=$NERSC_SCRATCH/archive10; PY=$D/env/bin/python; S=$D/search10a
mkdir -p $S/truth $S/log $D/scen
echo "node $(hostname) $(date)"
DETS="$A10/sa_0/det $A10/sa_1/det $A10/sa_2/det $A10/sa_redo/det $A10/sa_fix/det"
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
sort -k3 -n -r $S/list.txt | xargs -P 5 -L 1 bash -c 'build $0 $1'
echo "done $(date)"
