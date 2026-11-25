#!/bin/bash
# usage: stk.sh NAME SCENDIR BUDGET [ENV=VAL ...]   -- stackd alone on a fresh copy of submission
T=$RSI_SANDBOX/r1/gen118/tmp; N=$1; SC=$2; BU=$3; shift 3
rm -rf $T/s_$N; mkdir $T/s_$N; cp -R $RSI_SANDBOX/r1/gen118/workspace/submission/{finder.py,linker,stackd} $T/s_$N/
cd $T/s_$N/stackd && env OMP_NUM_THREADS=4 PYTORCH_ENABLE_MPS_FALLBACK=1 TNO_NULL_FRAC=0.35 "$@" $HOME/anaconda3/bin/python -I finder.py --scenario $SC --out $T/runs/stk_$N.json --budget-s $BU > $T/runs/stk_$N.log 2>&1
