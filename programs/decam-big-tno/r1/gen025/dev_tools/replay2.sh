#!/bin/bash
# usage: replay.sh FINDER NAME BUDGET [ENV=VAL ...]
f=$1; name=$2; budget=$3; shift 3
env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 FINDER_RR=39.2,41.8 FINDER_DEVCACHE=/tmp/w/cache_p4.npz FINDER_PASSES=99 \
  FINDER_LOADRAW=/tmp/w/base/raw5.pkl FINDER_LOADGROWN=/tmp/w/base/grown.pkl FINDER_LOAD3=/tmp/w/base/g3.pkl FINDER_DEBUG=1 "$@" \
  python3 $f --scen /tmp/p4 --out /tmp/w/r_$name --budget-s $budget --procs 1 > /tmp/w/r_$name.log 2>&1
