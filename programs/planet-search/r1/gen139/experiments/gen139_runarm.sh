#!/bin/zsh
# usage: runarm.sh SUBDIR SCEN TAG [extra env...]
SUB=$1; SC=$2; TAG=$3; shift 3
cd $RSI_SANDBOX/r1/gen139/tmp/$SUB
env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_ENABLE_MPS_FALLBACK=1 TNO_FSTK_DMIN=40 TNO_FSTK_DMAX=44 TNO_FSTK_F=0.5 \
  TNO_FSTK_DUMP=$RSI_SANDBOX/r1/gen139/tmp/runs/fdump_$TAG.json "$@" \
  $HOME/anaconda3/bin/python -I finder.py --scenario $RSI_SANDBOX/r1/gen139/tmp/$SC \
  --out $RSI_SANDBOX/r1/gen139/tmp/runs/out_$TAG.json --budget-s 900 > $RSI_SANDBOX/r1/gen139/tmp/runs/log_$TAG.txt 2>&1
