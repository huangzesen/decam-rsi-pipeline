#!/bin/bash
# usage: replay.sh NAME BUDGET [ENV=VAL ...]   (growth cache and raw linkages of the baseline run; growth + carry only)
name=$1; budget=$2; shift 2
RAW=$(ls /tmp/w/base/raw*.pkl | sort | tail -1)
env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 FINDER_RR=39.2,41.8 FINDER_DEVCACHE=/tmp/w/cache_p4.npz FINDER_PASSES=99 \
  FINDER_LOADRAW=$RAW FINDER_LOADGROWN=/tmp/w/base/grown_full.pkl FINDER_LOAD3=/tmp/w/base/g3.pkl FINDER_DEBUG=1 "$@" \
  python3 /app/submission/finder.py --scen /tmp/p4 --out /tmp/w/r_$name --budget-s $budget --procs 1 > /tmp/w/r_$name.log 2>&1
