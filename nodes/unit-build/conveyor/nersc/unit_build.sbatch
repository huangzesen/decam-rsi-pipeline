#!/bin/bash
# conveyor unit scenarios, a copy of rsi/decam-big-tno/harness/searchldt5/buildldt5.sbatch for any unit: U = the unit's
# NERSC folder (detection lists in U/*/det), B = its build folder dbt/units/<uid> (in/regions_exposures.json,
# in/known/<rid>.json, list.txt "RID SID NEXP"). Two builds at once on a CPU node; a cut-off build leaves <sid>.part,
# which the next job removes and redoes. Writes B/BUILT when every scenario of the list exists.
# sbatch -q debug -t 00:30:00 -J cvb-<uid> --export=ALL,U=...,B=... unit_build.sbatch
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -A $NERSC_CPU_ACCOUNT
#SBATCH -o $NERSC_SCRATCH/units/slurm/build-%j.out
D=$NERSC_SCRATCH/dbt; PY=$D/env/bin/python
mkdir -p $B/truth $B/log $D/scen
echo "node $(hostname) $(date) unit $U build $B"
DETS="$(ls -d $U/*/det | tr '\n' ' ')"
build() {
  rid=$1; sid=$2
  [ -f $D/scen/$sid/exposures.json ] && { echo "$rid $sid exists"; return; }
  rm -rf $D/scen/$sid.part
  $PY -I $D/units/build_scen.py $rid $sid $B/in/regions_exposures.json $B/in/known/$rid.json $D/scen/$sid.part $B/truth $DETS \
      > $B/log/build_$rid.log 2>&1 && mv $D/scen/$sid.part $D/scen/$sid && chmod -R a-w $D/scen/$sid
  echo "$rid $sid rc=$? $(tail -1 $B/log/build_$rid.log | cut -c1-120)"
}
export -f build; export D B PY DETS
# L = another list (early build of the regions already detected, 2026-11-20): never marks the unit BUILT
LIST=${L:-$B/list.txt}
sort -k3 -n -r $LIST | xargs -P 2 -L 1 bash -c 'build $0 $1'
n=0; m=0; while read -r rid sid nexp; do m=$((m+1)); [ -f $D/scen/$sid/exposures.json ] && n=$((n+1)); done < $LIST
echo "done $(date): $n of $m scenarios ($LIST)"; [ -z "$L" ] && [ $n -eq $m ] && touch $B/BUILT
