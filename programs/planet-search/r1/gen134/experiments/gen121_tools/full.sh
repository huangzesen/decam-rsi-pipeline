#!/bin/bash
# usage: full.sh NAME SRCDIR SCENDIR BUDGET [ENV=VAL ...]   -- full wrapper on a fresh copy of SRCDIR
T=$RSI_SANDBOX/r1/gen121/tmp; N=$1; SRC=$2; SC=$3; BU=$4; shift 4
rm -rf $T/f_$N; mkdir $T/f_$N; cp -R $SRC/{finder.py,linker,stackd} $T/f_$N/
cd $T/f_$N && env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_ENABLE_MPS_FALLBACK=1 TNO_KEEP=1 "$@" $HOME/anaconda3/bin/python -I finder.py --scenario $SC --out $T/runs/b_$N.json --budget-s $BU > $T/runs/b_$N.log 2>&1
