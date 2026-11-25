#!/bin/bash
# usage: full.sh NAME SCENDIR BUDGET [ENV=VAL ...]   -- full wrapper on a fresh copy of submission
T=$RSI_SANDBOX/r1/gen118/tmp; N=$1; SC=$2; BU=$3; shift 3
rm -rf $T/f_$N; mkdir $T/f_$N; cp -R $RSI_SANDBOX/r1/gen118/workspace/submission/{finder.py,linker,stackd} $T/f_$N/
cd $T/f_$N && env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_ENABLE_MPS_FALLBACK=1 TNO_KEEP=1 "$@" $HOME/anaconda3/bin/python -I finder.py --scenario $SC --out $T/runs/b_$N.json --budget-s $BU > $T/runs/b_$N.log 2>&1
