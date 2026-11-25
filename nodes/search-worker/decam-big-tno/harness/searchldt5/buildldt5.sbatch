#!/bin/bash
# searchldt5 (fourth 5 %, LDT-weighted; planet-search/ldt5): build the scenario of each region in LIST from ldt5's
# detection lists (ldt5/l5_*/det for the mirror exposures, ldt5/p0..p6/det for the downloaded ones). A copy of
# search10b/build10b.sbatch on a CPU debug node (30 min; NERSC's other CPU queues wait for the rolling reboot), two builds
# at once (a 200 M-row region took 81 GB; builds take 1-2 min each in search10b). A job cut off by its limit leaves a
# .part folder that the next job removes and redoes. When every scenario of LIST exists it writes $S/BUILT, which the
# search jobs (runldt5.sbatch) wait for. Scenarios -> $D/scen/<sid>/ (read-only), labels -> $D/searchldt5/truth/.
# Submit from $D/searchldt5: sbatch --export=ALL,LIST=<file of "RID SID NEXP" lines> buildldt5.sbatch
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -A $NERSC_CPU_ACCOUNT
#SBATCH -t 00:30:00
#SBATCH -J dbt-buildl5
#SBATCH -o log/buildl5-%j.out
D=$NERSC_SCRATCH/dbt; L5=$NERSC_SCRATCH/ldt5; PY=$D/env/bin/python; S=$D/searchldt5
mkdir -p $S/truth $S/log $D/scen
echo "node $(hostname) $(date) list $LIST"
DETS="$(for k in 0 1 2 3 4 5 6 7; do printf "%s " $L5/l5_$k/det; done)$(for k in 0 1 2 3 4 5 6; do printf "%s " $L5/p$k/det; done)"
build() {
  rid=$1; sid=$2
  [ -f $D/scen/$sid/exposures.json ] && { echo "$rid $sid exists"; return; }
  rm -rf $D/scen/$sid.part
  $PY -I $S/build_scen.py $rid $sid $S/in/regions_exposures.json $S/in/known/$rid.json $D/scen/$sid.part $S/truth $DETS \
      > $S/log/build_$rid.log 2>&1 && mv $D/scen/$sid.part $D/scen/$sid && chmod -R a-w $D/scen/$sid
  echo "$rid $sid rc=$? $(tail -1 $S/log/build_$rid.log | cut -c1-120)"
}
export -f build; export D S PY DETS
sort -k3 -n -r $LIST | xargs -P 2 -L 1 bash -c 'build $0 $1'
n=0; m=0; while read -r rid sid nexp; do m=$((m+1)); [ -f $D/scen/$sid/exposures.json ] && n=$((n+1)); done < $LIST
echo "done $(date): $n of $m scenarios"; [ $n -eq $m ] && touch $S/BUILT
