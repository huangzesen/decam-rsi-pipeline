#!/bin/bash
# search38: build the scenario of each 5 % search region (beyond the three sealed ones) on one NERSC CPU node, from the
# archive5 detection lists on NERSC scratch (NERSC's own share in archive5/n*/det, Vista's in archive5/vista/r38_*/det).
# Scenarios -> $D/scen/<sid>/ (read-only), known-body labels -> $D/search38/truth/ (copied to the Mac, then deleted here).
# 2026-11-13.
# Submit from $D/search38: sbatch build38.sbatch  (reads list.txt: "RID SID" lines; 6 builds at once)
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -A $NERSC_CPU_ACCOUNT
#SBATCH -t 04:00:00
#SBATCH -J dbt-build38
#SBATCH -o log/build38-%j.out
D=$NERSC_SCRATCH/dbt; A5=$NERSC_SCRATCH/archive5; PY=$D/env/bin/python; S=$D/search38
mkdir -p $S/truth $S/log $D/scen
echo "node $(hostname) $(date)"
DETS="$A5/n0/det $A5/n1/det $A5/n2/det $A5/n3/det $A5/vista/r38_0/det $A5/vista/r38_1/det $A5/vista/r38_2/det $A5/vista/r38_3/det $A5/vista/det"
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
sort -k3 -n -r $S/list.txt | xargs -P 6 -L 1 bash -c 'build $0 $1'
echo "done $(date)"
