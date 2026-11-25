#!/bin/bash
# usage: stk.sh NAME [ENV=VAL ...]
T=$RSI_SANDBOX/r1/gen020/tmp; N=$1; shift
rm -rf $T/s_$N; cp -R $RSI_SANDBOX/r1/gen020/workspace/submission $T/s_$N
cd $T/s_$N/stackd && env OMP_NUM_THREADS=4 PYTORCH_ENABLE_MPS_FALLBACK=1 TNO_XSEED=4 "$@" $HOME/anaconda3/bin/python -I finder.py --scenario $T/INJ --out $T/stk_$N.json --budget-s 330 > $T/stk_$N.log 2>&1
